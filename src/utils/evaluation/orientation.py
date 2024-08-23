import json
import os

import numpy as np
import torch

from models.center_groundtruth import CenterDirGroundtruth

from utils.evaluation import NumpyEncoder
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval

class OrientationEval(CenterGlobalMinimizationEval):

    def __init__(self, *args, use_gt_centers=False, append_orientation_to_display_name=True, **kwargs):
        super(OrientationEval, self).__init__(*args, **kwargs)

        self.use_gt_centers = use_gt_centers
        self.metrics.update(dict(rotation=[],translation=[], rot_x=[], rot_y=[], rot_z=[]))
        self.append_orientation_to_display_name = append_orientation_to_display_name


    def add_image_prediction(self, im_name, im_index, im_shape, predictions, predictions_score, pred_angles,
                             gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, return_matched_gt_idx=False,
                             **kwargs):

        # use parent class for center prediction matching
        ret = super(OrientationEval, self).add_image_prediction(
            im_name, im_index, im_shape, predictions, predictions_score,
            gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, return_matched_gt_idx=True)

        gt_missed, pred_missed, pred_gt_match_by_center, filename_suffix, pred_gt_match_by_center_idx = ret

        gt_maps_keys = ['gt_orientation_sin', 'gt_orientation_cos']
        gt_sin_orientation, gt_cos_orientation = CenterDirGroundtruth.parse_single_batch_groundtruth_map(centerdir_gt,
                                                                                                         keys=gt_maps_keys)

        if filename_suffix is None:
            filename_suffix = ''

        if pred_gt_match_by_center_idx.shape[0] != 0:
            gt_selected = np.array([gt_centers_dict[np.int16(i)][::-1] for i in pred_gt_match_by_center_idx[:,0] if i >= 0])

            if len(gt_selected) > 0:
                assert len(pred_angles) == len(pred_gt_match_by_center)
                assert len(predictions) == len(pred_gt_match_by_center)
                assert len(gt_selected) == sum(pred_gt_match_by_center[:,0] != 0)

                trans_err = np.abs(predictions[pred_gt_match_by_center[:,0] != 0, :2] - gt_selected)

                angle_err = []

                num_orientation_dim = pred_angles.shape[1]
                pred_angles = pred_angles[pred_gt_match_by_center[:,0] != 0,:]

                for i, c_gt in enumerate(gt_selected):
                    if pred_gt_match_by_center[i] == 0:
                        continue

                    s = gt_sin_orientation[0:num_orientation_dim,0,int(c_gt[1]), int(c_gt[0])]
                    c = gt_cos_orientation[0:num_orientation_dim,0,int(c_gt[1]), int(c_gt[0])]

                    gt_angle_i = torch.atan2(c, s)
                    gt_angle_i = torch.rad2deg(gt_angle_i)
                    gt_angle_i += 360 * (gt_angle_i < 0).int()

                    pred_angle_i = pred_angles[i]

                    e = np.abs(gt_angle_i.cpu().numpy() - pred_angle_i)
                    
                    is_e_over_180 = (e > 180).astype(np.int32)
                    e = is_e_over_180 * 360 - (is_e_over_180*2-1) * e # the same as: e = 360 - e if e > 180 else e

                    angle_err.append(e)

                if len(angle_err) > 0:
                    angle_err = np.array(angle_err)
                    overall_rot_err = np.mean(angle_err,axis=1)

                    self.metrics['rotation'].extend(overall_rot_err)
                    self.metrics['translation'].extend(trans_err)

                    if self.append_orientation_to_display_name:
                        filename_suffix = f're_{np.mean(overall_rot_err):05.2f}_te_{np.mean(trans_err):05.2f}_{filename_suffix}'

                    if len(angle_err.shape) > 1 and angle_err.shape[1] == 3:
                        axis_rot_err = [angle_err[:,i] for i in range(angle_err.shape[1])]

                        for rot_err,rot_axis in zip(axis_rot_err,['rot_y', 'rot_z', 'rot_x']):
                            self.metrics[rot_axis].extend(rot_err)

                        if self.append_orientation_to_display_name:
                            axis_rot_dict = dict(zip(['ry', 'rz', 'rx'], np.mean(angle_err,axis=0)))
                            filename_suffix = "_".join([f'{a}_{axis_rot_dict[a]:05.2f}' for a in ['rx', 'ry', 'rz']] + [filename_suffix])
            else:
                print(f"No matching predictions found for {im_name}")
        if return_matched_gt_idx:
            return gt_missed, pred_missed, pred_gt_match_by_center, filename_suffix, pred_gt_match_by_center_idx
        else:
            return gt_missed, pred_missed, pred_gt_match_by_center, filename_suffix

    def calc_and_display_final_metrics(self, dataset, print_result=True, plot_result=True, save_dir=None, **kwargs):
        Re = np.array(self.metrics['Re']).mean()
        mae = np.array(self.metrics['mae']).mean()
        rmse = np.array(self.metrics['rmse']).mean()
        ratio = np.array(self.metrics['ratio']).mean()
        AP = np.array(self.metrics['precision']).mean()
        AR = np.array(self.metrics['recall']).mean()
        F1 = np.array(self.metrics['F1']).mean()
        TE = np.array(self.metrics['translation']).mean()
        RE = np.array(self.metrics['rotation']).mean()
        RE_Y = np.array(self.metrics['rot_y']).mean() if len(self.metrics['rot_y']) > 0 else None
        RE_Z = np.array(self.metrics['rot_z']).mean() if len(self.metrics['rot_z']) > 0 else None
        RE_X = np.array(self.metrics['rot_x']).mean() if len(self.metrics['rot_x']) > 0 else None

        if print_result:
            RES = 'Re=%.4f, mae=%.4f, rmse=%.4f, ratio=%.4f, AP=%.4f, AR=%.4f, F1=%.4f, translation=%.4f, rotation=%.4f' % (Re, mae, rmse, ratio, AP, AR, F1, TE, RE)
            if RE_X:
                RES += ", rot_x=%.4f" % RE_X
            if RE_Y:
                RES += ", rot_y=%.4f" % RE_Y
            if RE_Z:
                RES += ", rot_z=%.4f" % RE_Z
            print(RES)

        if self.center_ap_eval is not None:
            metrics_mAP = self.center_ap_eval.calc_and_display_final_metrics(print_result, plot_result)
        else:
            metrics_mAP = None, None

        metrics = dict(AP=AP, AR=AR, F1=F1, ratio=ratio, Re=Re, mae=mae, rmse=rmse, all_images=self.metrics,
                       metrics_mAP=metrics_mAP, translation=TE, rotation=RE, rot_x=RE_X, rot_y=RE_Y, rot_z=RE_Z)

        ########################################################################################################
        # SAVE EVAL RESULTS TO JSON FILE
        if metrics is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, 'results.json'), 'w') as file:
                file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics