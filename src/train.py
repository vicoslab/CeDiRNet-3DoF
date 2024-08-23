import collections
import datetime
import json
import os
import cv2
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import get_config_args
from criterions import get_criterion
from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils import transforms as my_transforms
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.hard_sampler import HardExamplesBatchSampler, DistributedRandomSampler, distributed_sync_dict
from utils.utils import AverageMeter, Logger, variable_len_collate
from utils.visualize import get_visualizer

from models.multitask_model import MultiTaskModel
from criterions.loss_weighting.weight_methods import get_weight_method


class Trainer:
    def __init__(self, local_rank, rank_offset, world_size, args, use_distributed_data_parallel=True):
        self.args = args
        self.world_size = world_size
        self.world_rank = rank_offset + local_rank
        self.local_rank = local_rank

        self.use_distributed_data_parallel = use_distributed_data_parallel and world_size > 1

        if args['save'] and self.world_rank == 0:
            if not os.path.exists(args['save_dir']):
                os.makedirs(args['save_dir'])

            # save parameters
            with open(os.path.join(args['save_dir'],'params.json'), 'w') as file:
                file.write(json.dumps(args, indent=4, sort_keys=True,default=lambda o: '<not serializable>'))

        if args['display']:
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

    def initialize_data_parallel(self, init_method=None):
        ###################################################################################################
        # set device
        if self.use_distributed_data_parallel and init_method is not None:
            self.use_distributed_data_parallel = True

            self.device = torch.device("cuda:%d" % self.local_rank)

            # if not master, then wait at least 5 sec to give master a chance for starting up first
            if self.world_rank != 0:
                time.sleep(10+np.random.randint(0,10))

            # initialize the process group
            torch.distributed.init_process_group("nccl", init_method=init_method, timeout=datetime.timedelta(hours=1),
                                                 rank=self.world_rank, world_size=self.world_size)

            print('Waiting for all nodes (ready from rank=%d/%d)' % (self.world_rank, self.world_size))
            sys.stdout.flush()
            torch.distributed.barrier(device_ids=[self.local_rank])
        else:
            self.use_distributed_data_parallel = False
            self.device = torch.device("cuda" if self.args['cuda'] else "cpu")

        #torch.backends.cudnn.benchmark = True

    def cleanup(self):
        if self.use_distributed_data_parallel:
            torch.distributed.destroy_process_group()

    def _to_data_parallel(self, X, **kwargs):
        if self.use_distributed_data_parallel:
            X = torch.nn.parallel.DistributedDataParallel(X.to(self.device), device_ids=[self.local_rank], find_unused_parameters=True, **kwargs)
        else:
            X = torch.nn.DataParallel(X.to(self.device), device_ids=[i for i in range(self.world_size)], **kwargs)
        return X

    def _synchronize_dict(self, array):
        if self.use_distributed_data_parallel:
            array = distributed_sync_dict(array, self.world_size, self.world_rank, self.device)
        return array

    def initialize(self):
        args = self.args
        device = self.device

        ###################################################################################################
        # train dataloader
        dataset_workers = args['train_dataset']['workers'] if 'workers' in args['train_dataset'] else 0
        dataset_batch = args['train_dataset']['batch_size'] if 'batch_size' in args['train_dataset'] else 1
        dataset_shuffle = args['train_dataset']['shuffle'] if 'shuffle' in args['train_dataset'] else True
        dataset_hard_sample_size = args['train_dataset'].get('hard_samples_size')

        self.accumulate_grads_iter = args['model'].get('accumulate_grads_iter',1)
        if self.accumulate_grads_iter:
            dataset_batch = dataset_batch // self.accumulate_grads_iter


        # in distributed settings we need to manually reduce batch size
        if self.use_distributed_data_parallel:
            dataset_batch = dataset_batch // self.world_size
            dataset_workers = 0 # ignore workers request since already using separate processes for each GPU
            if dataset_hard_sample_size:
                dataset_hard_sample_size = dataset_hard_sample_size // self.world_size

        def deepupdate(orig_dict, new_dict):
            for key, val in new_dict.items():
                if isinstance(val, collections.Mapping):
                    tmp = deepupdate(orig_dict.get(key, {}), val)
                    orig_dict[key] = tmp
                elif isinstance(val, list):
                    orig_dict[key] = (orig_dict.get(key, []) + val)
                else:
                    orig_dict[key] = new_dict[key]
            return orig_dict

        def create_dataset_with_batch_sampler(extra_kwargs=None, centerdir_groundtruth_op=None, name=args['train_dataset']['name'], batch_size=dataset_batch):
            db_args = args['train_dataset']['kwargs']
            db_args = deepupdate(deepcopy(db_args),extra_kwargs) if extra_kwargs is not None else db_args

            train_dataset, centerdir_groundtruth_op = get_centerdir_dataset(name, db_args,
                                                                            args['train_dataset'].get('centerdir_gt_opts'), centerdir_groundtruth_op)

            # prepare hard-examples sampler for dataset
            if dataset_shuffle:
                if self.use_distributed_data_parallel:
                    default_sampler = DistributedRandomSampler(train_dataset, device=self.device)
                else:
                    default_sampler = torch.utils.data.RandomSampler(train_dataset)
            else:
                default_sampler = torch.utils.data.SequentialSampler(train_dataset)

            batch_sampler = HardExamplesBatchSampler(train_dataset,
                                                     default_sampler,
                                                     batch_size=batch_size,
                                                     hard_sample_size=dataset_hard_sample_size,
                                                     drop_last=True,
                                                     hard_samples_selected_min_percent=args['train_dataset'].get('hard_samples_selected_min_percent'),
                                                     hard_samples_only_min_selected_when_empty=args['train_dataset'].get('hard_samples_only_min_selected_when_empty'),
                                                     device=self.device, world_size=self.world_size, rank=self.world_rank,
                                                     is_distributed=self.use_distributed_data_parallel)

            return train_dataset, batch_sampler, centerdir_groundtruth_op

    
        train_dataset, batch_sampler, centerdir_groundtruth_op = create_dataset_with_batch_sampler()

        train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler,
                                                    num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
                                                    collate_fn=variable_len_collate)

        ###################################################################################################
        # set model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'] if 'num_vector_fields' in args else args['loss_opts']['num_vector_fields'])

        # set center prediction head model
        center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                          is_learnable=args['center_model']['use_learnable_center_estimation'] if 'use_learnable_center_estimation' in args['center_model'] else True)
        center_model.init_output(args['num_vector_fields'] if 'num_vector_fields' in args else args['loss_opts']['num_vector_fields'])

        # set criterion
        criterion = get_criterion(args.get('loss_type'), args.get('loss_opts'), model, center_model)

        # set weight method
        multitask_weighting = None
        if args.get('multitask_weighting') and args['multitask_weighting']['name'] != 'off':

            # model must implement MultiTaskModel interface
            assert isinstance(model, MultiTaskModel)

            multitask_weighting = get_weight_method(args['multitask_weighting']['name'], device=device,
                                                    **args['multitask_weighting']['kwargs'])

        ############################################################
        ########## Prepare data/modules for distributed operations

        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = self._to_data_parallel(centerdir_groundtruth_op)

        model = self._to_data_parallel(model, dim=0)
        center_model = self._to_data_parallel(center_model, dim=0)
        criterion = self._to_data_parallel(criterion, dim=0)

        def get_optimizer(model_, args_):
            if args_ is None or args_.get('disabled'):
                return None, None
            if 'optimizer' not in args_ or args_['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model_.parameters(),lr=args_['lr'],
                                             weight_decay=args_['weight_decay'])
            elif args_['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model_.parameters(),lr=args_['lr'],
                                            momentum=args_['momentum'],
                                            weight_decay=args_['weight_decay'])
            # use custom lambda_scheduler_fn function that can pass args if available
            lr_lambda = args_['lambda_scheduler_fn'](args) if 'lambda_scheduler_fn' in args_ else args_['lambda_scheduler']
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            return optimizer, scheduler

        # set optimizer for model and for center model
        optimizer, scheduler = get_optimizer(model, args['model'])
        center_optimizer, center_scheduler = get_optimizer(center_model, args['center_model'])

        if args.get('resume') and not args.get('resume_path'):
            args['resume_path'] = self._get_last_checkpoint()

        ########################################################################################################################
        # Visualizer

        # default visualizer
        viz_name = 'CentersVisualizeTrain'
        viz_opts = dict(tensorboard_dir=os.path.join(args['save_dir'], 'tensorboard'),
                        tensorboard_continue=args['resume_path'] is not None)

        # newer version
        if args.get('visualizer'):
            viz_name = args['visualizer']['name']
            viz_opts.update(args['visualizer']['opts'] if 'opts' in args['visualizer'] else dict())

        # disable saving for non-master tasks
        if self.world_rank != 0:
            viz_opts['tensorboard_dir'] = None

        visualizer = get_visualizer(viz_name, viz_opts)

        ########################################################################################################################
        # Logger
        self.logger = Logger(('train', ), 'loss')

        self.sample_loss_history = {}
        self.sample_centerdir_loss_history = {}
        self.sample_hard_neg_selection = {}

        # resume
        self.start_epoch = 0
        if args['resume_path'] is not None and os.path.exists(args['resume_path']):
            print('Resuming model from {}'.format(args['resume_path']))
            state = torch.load(args['resume_path'])
            self.start_epoch = state['epoch'] + 1
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if 'optim_state_dict' in state and optimizer: optimizer.load_state_dict(state['optim_state_dict'])
            if 'criterion_state_dict' in state: criterion.load_state_dict(state['criterion_state_dict'])
            if 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                center_model.load_state_dict(state['center_model_state_dict'], strict=True)
            if 'center_optim_state_dict' in state and center_optimizer: center_optimizer.load_state_dict(state['center_optim_state_dict'])
            self.logger.data = state['logger_data']

        if args.get('pretrained_model_path') is not None and os.path.exists(args['pretrained_model_path']):
            print('Loading pre-trained model from {}'.format(args['pretrained_model_path']))
            state = torch.load(args['pretrained_model_path'])
            if 'model_state_dict' in state:
                INPUT_WEIGHTS_KEY = 'module.model.encoder.model.stem_0.weight'
                if INPUT_WEIGHTS_KEY in state['model_state_dict']:                
                    checkpoint_input_weights = state['model_state_dict'][INPUT_WEIGHTS_KEY]
                    model_input_weights = model.module.model.encoder.model.stem_0.weight
                    if checkpoint_input_weights.shape[1] < model_input_weights.shape[1]:
                        weights = torch.zeros_like(model_input_weights)
                        weights[:, :checkpoint_input_weights.shape[1], :, :] = checkpoint_input_weights
                        state['model_state_dict'][INPUT_WEIGHTS_KEY] = weights

                        print('WARNING: #####################################################################################################')
                        print(f'WARNING: pretrained model input shape mismatch - will load weights for only the first {checkpoint_input_weights.shape[1]} channels, is this correct ?!!!')
                        print('WARNING: #####################################################################################################')

                missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
                if len(missing) > 0 or len(unexpected) > 0:
                    print('WARNING: #####################################################################################################')
                    print('WARNING: Current model differs from the pretrained one, loading weights using strict=False')
                    print('WARNING: #####################################################################################################')

            if 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                center_model.load_state_dict(state['center_model_state_dict'], strict=True)

        if args.get('pretrained_center_model_path') is not None and os.path.exists(args['pretrained_center_model_path']):
            print('Loading pre-trained center model from {}'.format(args['pretrained_center_model_path']))
            state = torch.load(args['pretrained_center_model_path'])

            INPUT_WEIGHTS_KEY = 'module.instance_center_estimator.conv_start.0.weight'
            if INPUT_WEIGHTS_KEY in state['center_model_state_dict']:
                checkpoint_input_weights = state['center_model_state_dict'][INPUT_WEIGHTS_KEY]
                center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                if checkpoint_input_weights.shape != center_input_weights.shape:
                    state['center_model_state_dict'][INPUT_WEIGHTS_KEY] = checkpoint_input_weights[:, :2, :, :]

                    print('WARNING: #####################################################################################################')
                    print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                    print('WARNING: #####################################################################################################')

            center_model.load_state_dict(state['center_model_state_dict'], strict=False)

        denormalize_args = None

        # get prepare values/functions needed for display
        if 'transform' in args['train_dataset']['kwargs']:
            transforms = args['train_dataset']['kwargs']['transform']
            if isinstance(transforms,my_transforms.Compose):
                denormalize_args = [(t.mean[t.keys == 'image'], t.std[t.keys == 'image'])
                                    for t in transforms.transforms if type(t) == my_transforms.Normalize and 'image' in t.keys]
                denormalize_args = denormalize_args[0] if len(denormalize_args) > 0 else None
            elif isinstance(transforms,list):
                denormalize_args = [(t['opts']['mean'][t['opts']['keys'] == 'image'], t['opts']['std'][t['opts']['keys'] == 'image'])
                                    for t in transforms if t['name'] == 'Normalize' and 'image' in t['opts']['keys']]
                denormalize_args = denormalize_args[0] if len(denormalize_args) > 0 else None




        log_r_fn = criterion.module.log_r_fn if hasattr(criterion.module, "use_log_r") and criterion.module.use_log_r else None
        if 'learnable_center_loss' in args['loss_opts'] and args['loss_opts']['learnable_center_loss'] == 'cross-entropy':
            center_conv_resp_fn = lambda x: torch.sigmoid(x)
        else:
            center_conv_resp_fn = lambda x: x #torch.relu(x)

        self.device = device
        self.batch_sampler, self.train_dataset_it, self.dataset_batch = batch_sampler, train_dataset_it, dataset_batch
        self.model, self.center_model = model, center_model
        self.scheduler, self.center_scheduler = scheduler, center_scheduler
        self.optimizer, self.center_optimizer = optimizer, center_optimizer
        self.criterion, self.centerdir_groundtruth_op  = criterion, centerdir_groundtruth_op
        self.multitask_weighting = multitask_weighting

        self.center_conv_resp_fn, self.log_r_fn = center_conv_resp_fn, log_r_fn
        self.visualizer, self.denormalize_args = visualizer, denormalize_args

    def do_tf_logging(self, iter, type):
        if self.world_rank != 0:
            return False

        args = self.args
        if 'tf_logging' not in args:
            return False

        if 'tf_logging' in args and type not in args['tf_logging']:
            return False

        if 'tf_logging_iter' in args and iter % args['tf_logging_iter'] != 0:
            return False

        return True

    def print(self, *kargs, **kwargs):
        if self.world_rank == 0:
            print(*kargs, **kwargs)

    def train(self, epoch):
        args = self.args

        device = self.device
        batch_sampler, train_dataset_it, dataset_batch = self.batch_sampler, self.train_dataset_it, self.dataset_batch
        model, center_model = self.model, self.center_model
        optimizer, center_optimizer = self.optimizer, self.center_optimizer
        criterion, centerdir_groundtruth_op = self.criterion, self.centerdir_groundtruth_op

        # sum over channels and spatial location
        reduction_dim = (1,2,3)

        # put model into training mode
        model.train()
        center_model.train()
       
        # define meters
        loss_meter = AverageMeter()

        if optimizer:
            for param_group in optimizer.param_groups:
                self.print('learning rate (model): {}'.format(param_group['lr']))
        if center_optimizer:
            for param_group in center_optimizer.param_groups:
                self.print('learning rate (center_model): {}'.format(param_group['lr']))


        iter=epoch*len(train_dataset_it)

        all_samples_metrics = {}
        tqdm_iterator = tqdm(train_dataset_it, desc="Training epoch #%d/%d" % (epoch,args['n_epochs']),dynamic_ncols=True) if self.world_rank == 0 else None

        for i, sample in enumerate(tqdm_iterator if tqdm_iterator is not None else train_dataset_it):

            # call centerdir_groundtruth_op first which will create any missing centerdir_groundtruth (using GPU) and add synthetic output
            if centerdir_groundtruth_op is not None:
                sample = centerdir_groundtruth_op(sample, torch.arange(0, dataset_batch).int())

            im = sample['image']

            instances = sample['instance'].squeeze(dim=1)
            ignore = sample.get('ignore')
            centerdir_gt = sample.get('centerdir_groundtruth')

            from models.center_groundtruth import CenterDirGroundtruth

            loss_ignore = None
            if ignore is not None:
                # treat any type of ignore objects (truncated, border, etc) as ignore during training
                # (i.e., ignore loss and any groundtruth objects at those pixels)
                loss_ignore = ignore > 0

            # get difficult mask based on ignore flags (VALUE of 8 == difficult flag and VALUE of 2 == truncated flag )
            difficult = (((ignore & 8) | (ignore & 2)) > 0).squeeze(dim=1) if ignore is not None else torch.zeros_like(instances)

            # get gt_centers from centerdir_gt and convert them to dictionary (filter-out non visible and ignored examples)
            gt_centers = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_centers'])
            gt_centers_dict = CenterDirGroundtruth.convert_gt_centers_to_dictionary(gt_centers,
                                                                                    instances=instances,
                                                                                    ignore=loss_ignore)

            # retrieve and set random seed for hard examples from previous epoch
            # (will be returned as None if sample does not exist or is not hard-sample)
            sample['seed'] = batch_sampler.retrieve_hard_sample_storage_batch(sample['index'],'seed')

            output = model(im)

            # call center prediction model
            center_output = center_model(output, **sample)
            output, center_pred, center_heatmap, pred_mask = [center_output[k] for k in ['output','center_pred','center_heatmap','pred_mask']]

            # get losses
            losses = criterion(output, sample,
                            centerdir_responses=(center_pred, center_heatmap), centerdir_gt=centerdir_gt, ignore_mask=loss_ignore,
                            difficult_mask=difficult, reduction_dims=reduction_dim, epoch_percent=epoch/float(args['n_epochs']), **args['loss_w'])

            # since each GPU will have only portion of data it will not use correct batch size for averaging - do correction for this here
            if self.world_size > 1 and not self.use_distributed_data_parallel:
                losses = [l/float(self.world_size) for l in losses]

            sample_metrics = None
            if batch_sampler.has_hard_samples():
                # evaluate predictions to get metrics needed for difficulty score
                sample_metrics = self._evaluate_batch_predictions(center_pred, gt_centers_dict, ignore, difficult)

                # calc difficulty score from losses and metrics
                sample_difficulty_score = self._calc_sample_difficulty(losses, sample_metrics)

                # pass losses to hard-samples batch sampler
                batch_sampler.update_difficulty_score(sample, sample_difficulty_score, index_key='index', storage_keys=['seed'])

            # save losses and metrics from this batch to common storage for this epoch
            all_samples_metrics = self._updated_per_epoch_sample_metrics(all_samples_metrics, sample['index'],
                                                                         losses, sample_metrics)
            loss = losses[0].sum()

            # we can simply sum the final loss since average is already calculated through weighting
            if self.multitask_weighting is None:
                bp_loss = loss / self.accumulate_grads_iter
                bp_loss.backward()
            else:
                losses_groups = criterion.module.get_loss_dict(losses)
                assert 'losses_tasks' in losses_groups, \
                    'Criterion is not compatible with multitask weighting (missing "task_losses" key in returned get_loss_dict())'

                # get individual tasks losses, sum them over batches and apply accumulate_grads_iter factor
                losses_tasks = [t_loss.sum() for t_name, t_loss in losses_groups['losses_tasks'].items()]
                losses_tasks = [t_loss / self.accumulate_grads_iter for t_loss in losses_tasks]

                bp_loss, extra_outputs = self.multitask_weighting(
                    losses=torch.stack(losses_tasks),
                    shared_parameters=list(model.module.shared_parameters()),
                    task_specific_parameters=list(model.module.task_specific_parameters()),
                    last_shared_parameters=list(model.module.last_shared_parameters()),
                    representation=None, # TODO: add representation if needed at all
                )



            if self.do_tf_logging(iter, 'weights'):
                self.visualizer.log_conv_weights(model, iter=iter)

            if ((i + 1) % self.accumulate_grads_iter == 0) or (i + 1 == len(train_dataset_it)):
                if optimizer:
                    optimizer.step()
                    optimizer.zero_grad() # set_to_none=False for prior to v2.0 and set_to_none=True after v2.0

                if center_optimizer:
                    center_optimizer.step()
                    center_optimizer.zero_grad()

            if self.do_tf_logging(iter, 'loss'):
                self.visualizer.log_grouped_scalars_tf(grouped_dict=criterion.module.get_loss_dict(losses), iter=iter,
                                                       name='losses')

            if self.world_rank == 0 and args['display'] and i % args['display_it'] == 0:

                center_conv_resp = self.center_conv_resp_fn(center_heatmap[:,0]) if center_heatmap is not None else None

                self.visualizer(im, output, pred_mask, center_conv_resp, centerdir_gt, gt_centers_dict, difficult,
                                self.log_r_fn, (i//args['display_it'])%len(im), device, self.denormalize_args)

            loss_meter.update(loss.item())

            if tqdm_iterator is not None:
                loss_dict = criterion.module.get_loss_dict(losses)
                
                from collections import OrderedDict
                tqdm_plots = OrderedDict(loss=loss.item())
                
                tqdm_plots.update({n: l.cpu().item() for n,l in loss_dict['losses_tasks' if 'losses_tasks' in loss_dict else 'losses_groups'].items()})

                tqdm_iterator.set_postfix(**tqdm_plots)

            iter+=1

        all_samples_metrics = self._synchronize_dict(all_samples_metrics)
        all_samples_total_loss = {k:v['loss'] for k,v in all_samples_metrics.items()}

        if epoch == args['n_epochs']:
            self.print('end')

        return np.array(list(all_samples_total_loss.values())).mean() * dataset_batch

    def _calc_sample_difficulty(self, losses, metrics):
        loss_total = losses[0]

        sample_difficulty_score = torch.zeros((len(losses[0]),), dtype=torch.float, device=losses[0].device)

        for b in range(len(loss_total)):
            FP, FN = metrics[b]['FP'], metrics[b]['FN']

            # multiply loss with number of FP and FN for hard neg
            sample_difficulty_score[b] = loss_total[b].sum() * (FP + 2 * FN + 1) ** 2
            # sample_difficulty_score[b] = loss_total[b].sum() * (FN) ** 2

        return sample_difficulty_score

    def _evaluate_batch_predictions(self, center_pred, gt_centers_dict, ignore, difficult):
        metrics = [{} for _ in range(len(center_pred))]

        for b in range(len(center_pred)):
            # calc FP and FN
            if center_pred is not None:
                center_eval = CenterGlobalMinimizationEval()
                valid_pred = center_pred[b, center_pred[b, :, 0] != 0, :]
                valid_pred = valid_pred[ignore[b, 0, valid_pred[:, 2].long(), valid_pred[:, 1].long()] == 0, :] if ignore is not None else valid_pred
                center_eval.add_image_prediction(None, None, None, valid_pred[:, 1:3].cpu().numpy(), None, None,
                                                 gt_centers_dict[b], difficult[b], None)
                FP, FN = center_eval.metrics['FP'][0], center_eval.metrics['FN'][0]
            else:
                FP, FN = 0, 0

            metrics[b] = dict(FP=FP, FN=FN)

        return metrics

    def _updated_per_epoch_sample_metrics(self, stored_results, sample_indexes, losses, metrics=None):
        if len(losses) > 8:
            loss_total, loss_cls, loss_centerdir_total, loss_centers, loss_sin, loss_cos, loss_r, loss_magnitude_reg, _ = losses[:9]
        else:
            loss_total, loss_cls, loss_centerdir_total, loss_centers, loss_sin, loss_cos, loss_r, loss_magnitude_reg = losses[:8]

        for b in range(len(loss_total)):
            index = sample_indexes[b].item()

            # add losses
            stored_results[index] = dict(loss=loss_total[b].sum().item(),
                                         loss_centerdir=loss_centerdir_total[b].sum().item() + loss_cls[b].sum().item())

            if metrics is not None:
                stored_results[index].update(metrics[b])

        return stored_results

    def save_checkpoint(self, state, is_best=False, name='checkpoint.pth'):
        args = self.args

        print('=> saving checkpoint')
        file_name = os.path.join(args['save_dir'], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(args['save_dir'], 'best_iou_model.pth'))
        if state['epoch'] % args.get('save_interval',10) == 0:
            shutil.copyfile(file_name, os.path.join(args['save_dir'], 'checkpoint_%03d.pth' % state['epoch']))

    def should_skip_training(self):
        last_interval = self.args['n_epochs'] - self.args.get('save_interval',10)
        last_checkpoint = os.path.join(self.args['save_dir'], 'checkpoint_%03d.pth' % last_interval)

        return self.args.get('skip_if_exists') and os.path.exists(last_checkpoint)

    def _get_last_checkpoint(self):
        valid_last_checkpoint = None

        recent_checkpoints = [os.path.join(self.args['save_dir'], 'checkpoint.pth')]
        recent_checkpoints += [os.path.join(self.args['save_dir'], 'checkpoint_%03d.pth' % epoch) for epoch in list(range(self.args['n_epochs']))[::-1]]

        for last_checkpoint in recent_checkpoints:
            if os.path.exists(last_checkpoint):
                valid_last_checkpoint = last_checkpoint
                break
        return valid_last_checkpoint

    def run(self):
        args = self.args

        for epoch in range(self.start_epoch, args['n_epochs']):

            train_loss = self.train(epoch)

            if self.world_rank == 0: print('Starting epoch {}'.format(epoch))
            if self.scheduler: self.scheduler.step()
            if self.center_scheduler: self.center_scheduler.step()

            if self.world_rank == 0:
                print('===> train loss: {:.2f}'.format(train_loss))

                self.logger.add('train', train_loss)
                self.logger.plot(save=args['save'], save_dir=args['save_dir'])

                if args['save'] and (epoch % args.get('save_interval',10) == 0 or epoch + 1 == args['n_epochs']):
                    state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict() if self.model is not None else None,
                        'optim_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                        'criterion_state_dict': self.criterion.state_dict() if self.criterion is not None else None,
                        'center_model_state_dict': self.center_model.state_dict() if self.center_model is not None else None,
                        'center_optim_state_dict': self.center_optimizer.state_dict() if self.center_optimizer is not None else None,
                        'logger_data': self.logger.data,
                    }
                    self.save_checkpoint(state)

def main(local_rank, rank_offset, world_size, init_method=None):

    args = get_config_args(dataset=os.environ.get('DATASET'), type='train')

    trainer = Trainer(local_rank, rank_offset, world_size, args, use_distributed_data_parallel=init_method is not None)

    if trainer.should_skip_training():
        print('Skipping due to already existing checkpoints (and requested to skip if exists) !!')
        return

    trainer.initialize_data_parallel(init_method)

    trainer.initialize()
    trainer.run()

    trainer.cleanup()

import torch.multiprocessing as mp

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    world_size = int(os.environ.get('WORLD_SIZE',default=n_gpus))
    rank_offset = int(os.environ.get('RANK_OFFSET',default=0))

    args = get_config_args(dataset=os.environ.get('DATASET'), type='train')

    if world_size <= 1 or args.get('disable_distributed_training'):
        main(0, 0, n_gpus)
    else:
        spawn = None
        try:
            print("spawning %d new processes" % n_gpus)
            spawn = mp.spawn(main,
                             args=(rank_offset,world_size,'env://'),
                             nprocs=n_gpus,
                             join=False)
            while not spawn.join():
                pass
        except KeyboardInterrupt:
            if spawn is not None:
                for pid in spawn.pids():
                    os.system("kill %s" % pid)
            torch.distributed.destroy_process_group()
