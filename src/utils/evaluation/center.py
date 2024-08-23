import json
import os

import numpy as np
import pylab as plt
import torch

from utils.evaluation import get_AP_and_F1, NumpyEncoder

class CenterEvaluation:
    def __init__(self, exp_name='', exp_attributes=None):
        self.Y = []
        self.P = []
        self.is_difficult = []
        self.exp_name = exp_name
        self.exp_attributes = exp_attributes

    def save_str(self):
        return ""
    def save_attributes(self):
        return dict(tau=self.tau_thr, d_alpha=self.merge_threshold_px, score_thr=self.score_thr)

    def add_image_prediction(self, im_name, im_index, im_shape, predictions, predictions_score,
                             gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, **kwargs):
        gt_centers = [gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())]
        gt_centers = np.array(gt_centers)

        gt_missed = False
        pred_missed = False

        pred_gt_match = []

        if len(predictions) > 0:

            pred_gt_center = []
            # TODO: this should be tested for any errors !!!
            # convert predictions from 2d to 1d index and use it to find corresponding GT ids
            for pred_id in np.ravel_multi_index(np.array([[pred[1],pred[0]] for pred in predictions])):
                gt_cent = np.array([0,0]) # by default
                for index,ids in gt_instances_ids.items():
                    if np.any(ids == pred_id):
                        gt_cent = np.mean(np.unravel_index(pred_id,im_shape),dim=0)
                        break
                pred_gt_center.append(gt_cent)
            # Original implementation using gt_instances map
            #pred_gt_center = [(gt_instances == gt_instances[int(np.round(pred[1])),
            #                                                int(np.round(pred[0]))]).nonzero().type(torch.float32).mean(0).cpu().numpy()
            #                  for pred in predictions]

            pred_gt_center = np.array(pred_gt_center)

            pred_list = np.concatenate((pred_gt_center[:, 1:2],
                                        pred_gt_center[:, 0:1],
                                        predictions_score.reshape(-1,1)), axis=1)

            # match predictions with groundtruth
            pred_gt_match = np.zeros(shape=(pred_list.shape[0], 1))
            remaining_pred = np.ones(len(predictions), dtype=np.float32)

            for gt_center in gt_centers:
                center_distance = [np.sqrt(np.sum(np.power(np.array([pred[1], pred[0]]) - gt_center, 2))) for pred in
                                   pred_list]
                # ignore detection that have already been matched to GT
                center_distance = center_distance * remaining_pred
                if (center_distance <= 5).any():
                    best_match_id = np.argmin(center_distance)

                    # mark only the detection with best match with the GT as TRUE POSITIVE
                    pred_gt_match[best_match_id] = 1
                    remaining_pred[best_match_id] = np.infpred_gt_match_by_center
                else:
                    self.Y.append(np.array([1.0]))
                    self.P.append(-np.inf)
                    self.is_difficult.append(gt_difficult[int(gt_center[0]), int(gt_center[1])].item())
                    gt_missed = True
                    print('best distance: %f' % np.min(center_distance))

            self.Y.extend(pred_gt_match[:])
            self.P.extend(pred_list[:, -1])
            self.is_difficult.extend([gt_difficult[int(gt[1]), int(gt[0])].item()
                                            for p, gt in zip(predictions, pred_list)])

            if not pred_gt_match.all():
                pred_missed = True

        elif len(gt_centers) > 0:
            gt_missed = True
            for gt_center in gt_centers:
                self.Y.append(np.array([1.0]))
                self.P.append(-np.inf)
                self.is_difficult.append(gt_difficult[int(gt_center[0]), int(gt_center[1])].item())

        return gt_missed, pred_missed, pred_gt_match, None

    def calc_and_display_final_metrics(self, dataset, print_result=True, plot_result=True, save_dir=None, **kwargs):
        AP, F1, precision, recall, thrs = get_AP_and_F1(self.Y, self.P, self.is_difficult)

        if print_result:
            print('AP=%f' % AP)
            print('best F-measure=%f at thr=%f' % (np.max(F1),thrs[np.argmax(F1)-1]))

        fig = None
        if plot_result:
            fig = plt.figure()
            plt.plot(recall,precision)
            plt.title('Precision-recall (AP=%f, F1=%f)' % (AP, np.max(F1)))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0,1)
            plt.ylim(0, 1)

        metrics = dict(AP=AP, F1=F1, precision=precision, recall=recall, thrs=thrs)

        ########################################################################################################
        # SAVE EVAL RESULTS TO JSON FILE
        if save_dir is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            if fig is not None:
                fig.savefig(os.path.join(out_dir, 'AP.png'))

            if metrics is not None:
                with open(os.path.join(out_dir, 'results.json'), 'w') as file:
                    file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics

    def get_results_timestamp(self, save_dir):
        res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'results.json')

        return os.path.getmtime(res_filename) if os.path.exists(res_filename) else 0