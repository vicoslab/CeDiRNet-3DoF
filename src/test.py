#!/usr/bin/python
import os
import copy
from matplotlib import pyplot as plt

import numpy as np
import scipy
import json
import torch

from datasets import get_centerdir_dataset
from models import get_model, get_center_model
from utils.utils import tensor_mask_to_ids, variable_len_collate
from utils.visualize import get_visualizer
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.evaluation.center import CenterEvaluation
from utils.evaluation.orientation import OrientationEval
from utils.evaluation.multivariate import MultivariateEval

from inference.processing import CenterDirProcesser

class Evaluator:
    def __init__(self, args):
        plt.ion()

        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        args = self.args

        ###################################################################################################
        # Visualizer

        # default class (legacy support)
        viz_name = 'CentersVisualizeTest'
        viz_opts = dict(tensorboard_dir=os.path.join(args['save_dir'], 'tensorboard'),
                        to_file_only=args.get('display_to_file_only'),
                        autoadjust_figure_size=bool(args.get('autoadjust_figure_size')))

        # newer version
        if args.get('visualizer'):
            viz_name = args['visualizer']['name']
            viz_opts.update(args['visualizer']['opts'] if 'opts' in args['visualizer'] else dict())

        self.visualizer = get_visualizer(viz_name, viz_opts)

        ###################################################################################################
        # set dataset

        self.dataset_it, self.centerdir_groundtruth_op, self.processed_image_iter = self._construct_dataset_and_processing(args, self.device)


    def _construct_dataset_and_processing(self, args, device):

        ###################################################################################################
        # dataloader
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1


        groundtruth_loading = args.get('groundtruth_loading')
        if groundtruth_loading is not None and groundtruth_loading != 'minimal':
            groundtruth_loading = True

        dataset, centerdir_groundtruth_op = get_centerdir_dataset(args['dataset']['name'], args['dataset']['kwargs'],
                                                                  args['dataset'].get('centerdir_gt_opts') if groundtruth_loading else None)

        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = torch.nn.DataParallel(centerdir_groundtruth_op).to(device)

        dataset_it = torch.utils.data.DataLoader(dataset, batch_size=dataset_batch, shuffle=False, drop_last=False,
                                                 num_workers=dataset_workers, pin_memory=True if args['cuda'] else False,
                                                 collate_fn=variable_len_collate)

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output(args['num_vector_fields'])
        model = torch.nn.DataParallel(model).to(device)

        center_model_list = []

        def get_center_fn():
            return get_center_model(args['center_model']['name'], args['center_model']['kwargs'],
                                    is_learnable=args['center_model'].get('use_learnable_center_estimation'))

        # prepare center_model and center_estimator based on number of center_checkpoint_path that will need to be processed
        if args.get('center_checkpoint_path') and isinstance(args['center_checkpoint_path'],list):
            assert 'center_checkpoint_name_list' in args and isinstance(args['center_checkpoint_name_list'],list)
            assert len(args['center_checkpoint_name_list']) == len(args['center_checkpoint_path'])

            for center_checkpoint_name, center_checkpoint_path in zip(args['center_checkpoint_name_list'], args['center_checkpoint_path']):
                center_model = get_center_fn()

                center_model.init_output(args['num_vector_fields'])
                center_model_list.append(dict(name=center_checkpoint_name,
                                              checkpoint=center_checkpoint_path,
                                              model=center_model))
        else:
            center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
            center_checkpoint_path = args.get('center_checkpoint_path')

            center_model = get_center_fn()

            center_model.init_output(args['num_vector_fields'])
            center_model_list.append(dict(name=center_checkpoint_name,
                                          checkpoint=center_checkpoint_path,
                                          model=center_model))

        for center_model_desc in center_model_list:
            center_model_desc['model'] = torch.nn.DataParallel(center_model_desc['model']).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state and args['center_model'].get('use_learnable_center_estimation'):
                for center_model_desc in center_model_list:
                    center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if args['center_model'].get('use_learnable_center_estimation'):
            for center_model_desc in center_model_list:
                if center_model_desc['checkpoint'] is None:
                    continue
                if os.path.exists(center_model_desc['checkpoint']):
                    print('Loading center model from "%s"' % center_model_desc['checkpoint'])
                    state = torch.load(center_model_desc['checkpoint'])
                    if 'center_model_state_dict' in state:
                        if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                            checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                            center_input_weights = center_model_desc['model'].module.instance_center_estimator.conv_start[0].weight
                            if checkpoint_input_weights.shape != center_input_weights.shape:
                                state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                                print('WARNING: #####################################################################################################')
                                print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                                print('WARNING: #####################################################################################################')

                        center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
                else:
                    raise Exception('checkpoint_path {} does not exist!'.format(center_model_desc['checkpoint']))

        ###################################################################################################
        # MAIN PROCESSING PIPELINE:
        processed_image_iter = CenterDirProcesser(model, center_model_list, device)

        return dataset_it, centerdir_groundtruth_op, processed_image_iter

    def compile_evaluation_list(self):
        args = self.args

        center_checkpoint_name = args.get('center_checkpoint_name_list')
        if center_checkpoint_name is None:
            center_checkpoint_name = [args.get('center_checkpoint_name')]
        if center_checkpoint_name[0] is None:
            center_checkpoint_name = ['']

        evaluation_lists_per_center_model = {}

        for center_model_name in center_checkpoint_name:

            #########################################################################################################
            ## PREPARE EVALUATION CONFIG/ARGS


            if type(args.get('eval')) == dict:
                args_eval = args['eval']
                import itertools, functools

                ##########################################################################################
                # Scoring function based on provided list of scores that we want to use

                # index should match what is returned as scores in predictions (after x,y locations)
                scoring_index = {'mask': 0, 'center': 1, 'hough_energy': 2, 'edge_to_area_ratio_of_mask': 3,
                                 'avg(mask_pix)': 4, 'avg(hough_pix)': 5, 'avg(projected_dist_pix)': 6,
                                 'avg(mask_dir)': 7, 'std(mask_dir)': 8, 'orientation_confidence': 7 }

                # scoring function that extracts requested scores and multiply them to get the final score
                # scoring_fn = lambda scores: np.multiply([scores[:,scoring_index[t]] for t in args_eval['final_score_combination']])

                # function that creates new dictionaries by combinting every value in dictionary if value is a list of values
                def combitorial_args_fn(X):
                    # convert to list of values
                    X_vals = [[val] if type(val) not in [list, tuple] else val for val in X.values()]
                    # apply product to a list of list of values
                    for vals in itertools.product(*X_vals):
                        yield dict(zip(X.keys(), vals))

                # function that creates final score by multiplying specific scores (i.e. multiplying columns based on score_types names)
                def _scoring_fn(scores, score_types):
                    selected_scores = [scores[:, scoring_index[t]] for t in score_types]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else selected_scores[0]

                # function that thresholds predictions based on selected columns and provided thr (as key-value pair in score_thr)
                def _scoring_thrs_fn(scores, score_thr_dict):
                    selected_scores = [scores[:, scoring_index[t]] > thr for t, thr in score_thr_dict.items() if
                                       thr is not None]
                    return np.multiply.reduce(selected_scores) if len(selected_scores) > 1 else selected_scores[0]

                ##########################################################################################
                # Create all evaluation classes based on combination of thresholds and eval arguments
                evaluation_lists = []

                score_combination_and_thr_list = args_eval.get('score_combination_and_thr')

                if type(score_combination_and_thr_list) not in [list, tuple]:
                    score_combination_and_thr_list = [score_combination_and_thr_list]

                ignore_in_final_score = args_eval.get('ignore_in_final_score')
                if not ignore_in_final_score:
                    ignore_in_final_score = []

                # iterate over different scoring types
                for score_combination_and_thr in score_combination_and_thr_list:
                    # create scoring function based on which scores are requested
                    scoring_fn = functools.partial(_scoring_fn, score_types=set(score_combination_and_thr.keys())-set(ignore_in_final_score))
                    # iterate over different combination of thresholds for scores that are used
                    for scoring_thrs in combitorial_args_fn(score_combination_and_thr):
                        # create thresholding function based on which thresholds are requested for each score
                        scoring_thrs_fn = functools.partial(_scoring_thrs_fn, score_thr_dict=scoring_thrs)
                        # iterate over all final_score_thr that have been requested
                        for final_score_thr in args_eval.get('score_thr_final', -np.inf):
                            scoring_str = "+".join(scoring_thrs.keys())
                            scoring_thr = ["%s=%.2f" % (t, thr) for t, thr in scoring_thrs.items() if thr is not None]
                            if args_eval.get('top_k_predictions'):
                                exp_name = "%s-final_score_thr=%.2f-%s-top_k_predictions=%d" % (scoring_str,final_score_thr,"-".join(scoring_thr),int(args_eval['top_k_predictions']))
                            else:
                                exp_name = "%s-final_score_thr=%.2f-%s" % (scoring_str,final_score_thr,"-".join(scoring_thr))
                            exp_attributes = dict(scoring=scoring_str,scoring_thr=scoring_thr, final_score_thr=final_score_thr)

                            center_eval = []
                            if args_eval.get('centers_global_minimization') is not None:
                                ##########################################################################################
                                ## Evaluation class for counting metric based on best-fit detection minimization
                                center_eval += [CenterGlobalMinimizationEval(exp_name=exp_name, exp_attributes=exp_attributes, **args)
                                                    for args in combitorial_args_fn(args_eval['centers_global_minimization'])]

                            if not args_eval.get('skip_center_eval'):
                                ##########################################################################################
                                ## Evaluation class for centers using only AP
                                center_eval += [CenterEvaluation()]


                            if args_eval.get('orientation'):
                                ##########################################################################################
                                ## Evaluation class for orientation

                                orientation_eval = [OrientationEval(exp_name=exp_name, exp_attributes=exp_attributes, **args)
                                                        for args in combitorial_args_fn(args_eval['orientation'])]
                            else:
                                orientation_eval = []

                            add_multivariate_eval_fn = lambda eval_obj: MultivariateEval(eval_obj, **args['eval']['enable_multivariate_eval']) \
                                                                        if args['eval'].get('enable_multivariate_eval') else eval_obj

                            evaluation_lists.append(dict(
                                scoring_fn=scoring_fn,
                                scoring_thrs_fn=scoring_thrs_fn,
                                final_score_thr=final_score_thr,
                                center_eval=[add_multivariate_eval_fn(c) for c in center_eval],
                                orientation_eval=[add_multivariate_eval_fn(o) for o in orientation_eval],
                                top_k_predictions=args_eval.get('top_k_predictions')
                            ))

            # Check if evaluations already exists and return empty list if needed
            if args.get('skip_if_exists') and args['save_dir'] is not None:
                if self._check_if_results_exist(evaluation_lists, self._get_save_dir(center_model_name)):
                    evaluation_lists = []

            evaluation_lists_per_center_model[center_model_name] = evaluation_lists

        return evaluation_lists_per_center_model

    def _get_checkpoint_timestamp(self):
        get_date_fn = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0
        args = self.args

        mod_date = get_date_fn(args['checkpoint_path'])

        return mod_date
    
    def _check_if_results_exist(self, evaluation_lists, save_dir):
        # consider results invalid if they are older than checkpoint modification time
        checkpoint_time = self._get_checkpoint_timestamp()

        # checkpoint does not exist so just return false
        if checkpoint_time == 0:
            return False

        c_eval_exists, o_eval_exists, i_eval_exists = [], [], []
        for eval_args in evaluation_lists:
            C = [c_eval.get_results_timestamp(save_dir) >= checkpoint_time for c_eval in eval_args['center_eval']]
            O = [o_eval.get_results_timestamp(save_dir) >= checkpoint_time for o_eval in eval_args['orientation_eval']]

            c_eval_exists.append(all(C))
            o_eval_exists.append(all(O))

        return all(c_eval_exists) and all(o_eval_exists) and all(i_eval_exists)
    
    def _get_save_dir(self, center_model_name):
        MARKER = self.args['center_checkpoint_name'] if 'center_checkpoint_name' in self.args else '##CENTER_MODEL_NAME##'

        if MARKER in self.args['save_dir']:
            return self.args['save_dir'].replace(MARKER, center_model_name)
        else:
            return self.args['save_dir']

    def _visualize_prediction(self, sample, result, predictions, predictions_score, eval_obj, eval_res,
                              difficult, impath, save_root):
        args = self.args
        if type(eval_obj) == MultivariateEval:
            eval_obj = eval_obj.eval_obj

        if eval_res is None:
            # skip visualization if no output from evaluator
            return
        elif type(eval_obj) in [CenterEvaluation, CenterGlobalMinimizationEval, OrientationEval]:
            # parse results
            gt_missed, pred_missed, pred_gt_match, filename_suffix, pred_gt_match_idx = eval_res

            # we do not have mask output from those evaluator so use
            pred_polygon = {}
        else:
            raise Exception("Unsupported type of evaluator for visualization")

        if args['display'] is True or \
                type(args['display']) is str and args['display'].lower() == 'all' or \
                type(args['display']) is str and args['display'].lower() == 'error_gt' and gt_missed or \
                type(args['display']) is str and args['display'].lower() == 'error' and (gt_missed or pred_missed):

            visualize_to_folder = os.path.join(save_root, eval_obj.exp_name, eval_obj.save_str())
            os.makedirs(visualize_to_folder, exist_ok=True)

            if len(predictions) > 0:
                # predictions (x,y) for ploting
                plot_predictions = np.array(predictions)[:, :2]

                # matched gt info
                plot_predictions_gt_match = np.concatenate((pred_gt_match[:, :1],
                                                            pred_gt_match_idx[:, :1]), axis=1)
            else:
                plot_predictions = []
                plot_predictions_gt_match = []

            base = self.visualizer.impath2name_fn(impath) if impath is not None else None

            self.visualizer(sample, result, plot_predictions, predictions_score, pred_polygon.values(),
                            plot_predictions_gt_match, difficult, filename_suffix+base, visualize_to_folder)

    #########################################################################################################
    ## MAIN RUN FUNCTION
    def run(self, evaluation_lists_per_center_model):
        args = self.args

        with torch.no_grad():

            #########################################################################################################
            ## PROCESS EACH IMAGE and DO EVALUATION
            for im_index,(sample,result) in enumerate(self.processed_image_iter(self.dataset_it, self.centerdir_groundtruth_op)):

                im = sample.get('image')

                im_shape = sample.get('im_shape')
                im_name = sample['im_name']

                instances = sample.get('instance')
                instances_ids = sample.get('instance_ids')
                centerdir_gt = sample.get('centerdir_groundtruth')
                ignore_flags = sample.get('ignore')
                gt_centers_dict = sample.get('center_dict')

                predictions_ = result['predictions']
                pred_angle_ = result.get('pred_angle')
                center_model_name = result['center_model_name']

                if im_shape is None:
                    assert im is not None, 'ERROR: cannot evaluate without "im" or "im_shape"'
                    im_shape = im.shape[-2:]

                # get difficult mask based on ignore flags (VALUE of 8 == difficult flag)
                difficult = (ignore_flags & 8 > 0).squeeze() if ignore_flags is not None else torch.sparse_coo_tensor(size=im_shape)


                # convert masks to instance 1D indexes that allow faster overlap calculation if not done yet
                if instances_ids is None:
                    assert instances is not None, 'ERROR: cannot evaluate without "instances" or "instances_ids"'
                    instances_ids = tensor_mask_to_ids(instances)

                all_scores = predictions_[:, 2:] if len(predictions_) > 0 else []

                assert center_model_name in evaluation_lists_per_center_model

                save_vis_root = self._get_save_dir(center_model_name)
                
                # evaluate for different combination of scoring, thresholding and other eval arguments
                for eval_args in evaluation_lists_per_center_model[center_model_name]:
                    scoring_fn = eval_args['scoring_fn']
                    scoring_thrs_fn = eval_args['scoring_thrs_fn']
                    final_score_thr = eval_args['final_score_thr']
                    center_eval = eval_args['center_eval']
                    orientation_eval = eval_args['orientation_eval']

                    if len(all_scores) > 0:
                        # 1. apply scoring function
                        predictions_score = scoring_fn(all_scores)

                        # 2. filter based on specific scoring thresholds
                        selected_pred_idx = np.where((predictions_score > final_score_thr) *
                                                     scoring_thrs_fn(all_scores) *
                                                     (predictions_.sum(axis=1) != 0))[0]
                        predictions = predictions_[selected_pred_idx,:]
                        predictions_score = predictions_score[selected_pred_idx]

                        if pred_angle_ is not None:
                            pred_angle = pred_angle_[selected_pred_idx,:]


                        if eval_args.get('top_k_predictions'):
                            top_k = eval_args['top_k_predictions']
                            k = min(top_k, len(predictions))
                            
                            predictions = predictions[:k]
                            predictions_score = predictions_score[:k]
                            
                            if pred_angle_ is not None:
                                pred_angle = pred_angle[:k]

                    else:
                        predictions = []
                        predictions_score = []
                        pred_angle = []

                    # 3. do evaluation and visualization for every evaluator in center_eval and instance_eval

                    # collected metrics for per-center evaluation
                    for c_eval in center_eval:
                        center_eval_res = c_eval.add_image_prediction(im_name, im_index, im_shape,
                                                                      predictions, predictions_score,
                                                                      instances_ids, gt_centers_dict, difficult, centerdir_gt,
                                                                      return_matched_gt_idx=True)

                        self._visualize_prediction(sample, result, predictions, predictions_score,
                                                   c_eval, center_eval_res, difficult, im_name, save_vis_root )

                    # collected metrics for orientation evaluation
                    for o_eval in orientation_eval:
                        orient_eval_res = o_eval.add_image_prediction(im_name, im_index, im_shape,
                                                                      predictions, predictions_score, pred_angle,
                                                                      instances_ids, gt_centers_dict, difficult, centerdir_gt,
                                                                      return_matched_gt_idx=True)

                        self._visualize_prediction(sample, result, predictions, predictions_score,
                                                   o_eval, orient_eval_res, difficult, im_name, save_vis_root)

            ########################################################################################################
            # FINALLY, OUTPUT RESULTS TO DISPLAY and FILE

            if 'eval' not in args or args['eval']:
                for center_model_name, evaluation_lists in evaluation_lists_per_center_model.items():
                    save_dir = self._get_save_dir(center_model_name)

                    for eval_args in evaluation_lists:
                        center_eval = eval_args['center_eval']
                        orientation_eval = eval_args['orientation_eval']

                        ########################################################################################################
                        ## Evaluation based on center point only and on orientations
                        for c_eval in center_eval + orientation_eval:
                            metrics = c_eval.calc_and_display_final_metrics(self.dataset_it, save_dir=save_dir)

                            # store parameters for reference
                            if save_dir is not None and args.get('save_eval_args'):
                                self.save_args(os.path.join(save_dir, c_eval.exp_name, c_eval.save_str()), c_eval.get_attributes())

    def save_args(self, save_dir, extra_args=None):
        if extra_args is not None:
            args = copy.deepcopy(self.args)
            args.update(extra_args)
        else:
            args = self.args

        with open(os.path.join(save_dir, 'eval_params.json'), 'w') as file:
            file.write(json.dumps(args, indent=4, sort_keys=True, default=lambda o: '<not serializable>'))
def main():
    from config import get_config_args

    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')

    eval = Evaluator(args)

    # get list of all evaluations that will need to be performed (based on different combinations of thresholds etc)
    evaluation_lists = eval.compile_evaluation_list()

    # continue with initialization and running all eval, unless evaluation_lists is empty
    if any([len(e) > 0 for e in evaluation_lists.values()]):
        # initialize after checking for valid list of evaluations
        eval.initialize()
        # finally run all evaluations
        eval.run(evaluation_lists)
    else:
        print('Skipping due to already existing output')

if __name__ == "__main__":
    main()