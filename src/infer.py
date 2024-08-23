#!/usr/bin/python
import os, time

from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import scipy

import torch

from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils.utils import variable_len_collate

class Inferencce:
    def __init__(self, args):
        # if args['display'] and not args.get('display_to_file_only'):
        if True:
            # plt.switch_backend('TkAgg')
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        args = self.args

        ###################################################################################################
        # set dataset and model
        self.dataset_it, self.model, self.center_model = self._construct_dataset_and_processing(args, self.device)


    #@classmethod
    def _construct_dataset_and_processing(self, args, device):

        ###################################################################################################
        # dataloader
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1

        from utils import transforms as my_transforms
        args['dataset']['kwargs']['transform'] = my_transforms.get_transform([
            { 'name': 'Padding', 'opts': { 'keys': ('image',), 'pad_to_size_factor': 32 } },
            { 'name': 'ToTensor', 'opts': { 'keys': ('image',), 'type': (torch.FloatTensor) } },
        ])

        dataset, _ = get_centerdir_dataset(args['dataset']['name'], args['dataset']['kwargs'], no_groundtruth=True)

        dataset_it = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch, shuffle=False, drop_last=False,
                                                 num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
                                                 collate_fn=variable_len_collate)

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])
        model = torch.nn.DataParallel(model).to(device)

        # prepare center_model and center_estimator based on number of center_checkpoint_path that will need to be processed

        center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
        center_checkpoint_path = args.get('center_checkpoint_path')

        center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                        is_learnable=args['center_model'].get('use_learnable_center_estimation'),
                                        use_fast_estimator=True)

        center_model.init_output(args['num_vector_fields'])
        center_model = torch.nn.DataParallel(center_model).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                center_model.load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if args['center_model'].get('use_learnable_center_estimation') and len(center_checkpoint_name) > 0:
            if os.path.exists(center_checkpoint_path):
                print('Loading center model from "%s"' % center_checkpoint_path)
                state = torch.load(center_checkpoint_path)
                if 'center_model_state_dict' in state:
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                            print('WARNING: #####################################################################################################')
                            print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                            print('WARNING: #####################################################################################################')

                    center_model.load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('checkpoint_path {} does not exist!'.format(center_checkpoint_path))

        return dataset_it, model, center_model

    #########################################################################################################
    ## MAIN RUN FUNCTION
    def run(self):
        args = self.args

        time_array = dict(model=[],center=[],post=[],total=[])
        with torch.no_grad():
            model = self.model
            center_model = self.center_model
            dataset_it = self.dataset_it

            assert dataset_it.batch_size == 1

            model.eval()
            center_model.eval()

            im_image = 0
            while im_image < 1000:

                for sample in self.dataset_it:
                    im_image += 1

                    torch.cuda.synchronize()
                    start_model = time.time()
                    # run main model
                    output_batch_ = model(sample['image'])

                    torch.cuda.synchronize()
                    start_center = time.time()

                    # run center detection model
                    center_pred, times = center_model(output_batch_)

                    #predictions = center_pred[0]

                    # make sure data is copied
                    torch.cuda.synchronize()
                    end = time.time()

                    time_model = start_center - start_model
                    time_center_total = end - start_center
                    time_center_preprocess = times[0]
                    time_center_only = times[1]
                    time_center_postprocess = times[2]
                    time_total = end - start_model

                    time_array['model'].append(time_model)
                    time_array['center'].append(time_center_only+time_center_preprocess)
                    time_array['post'].append(time_center_postprocess)
                    time_array['total'].append(time_total)

                    
                    print('time total: %.1f ms with model=%.1f ms and center=%.1f ms (pre=%.1f ms, cent=%.1f ms, post=%.1f ms)' %
                          (time_total*1000, time_model*1000, time_center_total*1000,
                           time_center_preprocess*1000, time_center_only*1000, time_center_postprocess*1000,))

                    if im_image > 1000:
                        break

        times_model = np.array(time_array['model'])[100::100]
        times_center = np.array(time_array['center'])[100::100]
        times_post = np.array(time_array['post'])[100::100]
        times_total = np.array(time_array['total'])[100::100]
        print('-------------------------------------------------------------')
        print('TIMES:')
        print('model: avg %.1f ms, std %.1f ms' % (times_model.mean()*1000,times_model.std()*1000 ))
        print('center: avg %.1f ms, std %.1f ms' % (times_center.mean() * 1000, times_center.std() * 1000))
        print('post: avg %.1f ms, std %.1f ms' % (times_post.mean() * 1000, times_post.std() * 1000))
        print('total: avg %.1f ms, std %.1f ms' % (times_total.mean() * 1000, times_total.std() * 1000))

def main():
    from config import get_config_args

    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')

    infer = Inferencce(args)
    infer.initialize()
    infer.run()

if __name__ == "__main__":
    main()