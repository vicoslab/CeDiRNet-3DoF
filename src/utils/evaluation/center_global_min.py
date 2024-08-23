import json
import os
import sys

import numpy as np
import pylab as plt
import scipy
import torch
import tqdm

from utils.evaluation import get_AP_and_F1, NumpyEncoder

class CenterGlobalMinimizationEval:
    def __init__(self, exp_name='', exp_attributes=None, score_thr=0, tau_thr=30, center_ap_eval=None, display_best_threshold=False, append_count_to_display_name=True):
        self.exp_name = exp_name
        self.exp_attributes = exp_attributes
        self.score_thr = score_thr
        self.tau_thr = tau_thr # max distance from grroundtruth that is still allowed to consider as true-positive
        self.metrics = dict(precision=[],recall=[],F1=[],ratio=[],TP=[],FP=[],FN=[],N=[],P=[],Re=[],mae=[],rmse=[])
        self.center_ap_eval = center_ap_eval
        self.display_best_threshold = display_best_threshold

        self.all_detections = []
        self.append_count_to_display_name = append_count_to_display_name

    def save_str(self):
        return "tau=%.1f-score_thr=%.1f-center_ap_eval=%s" % (self.tau_thr, self.score_thr, str(self.center_ap_eval))

    def get_attributes(self):
        attrs = dict(tau=self.tau_thr, score_thr=self.score_thr)
        if self.exp_attributes is not None:
            attrs.update(self.exp_attributes)
        return attrs

    def _assign_detections_to_groundtruth(self, predictions, gt_centers):
        #import time
        #start = time.time()
        if len(predictions) * len(gt_centers) > 1000 * 1000:
            cost_matrix = torch.cdist(torch.from_numpy(predictions),
                                      torch.from_numpy(gt_centers)).cpu().numpy()
        else:
            cost_matrix = scipy.spatial.distance_matrix(predictions, gt_centers)

        # remove predictions where distances are over predefined threshold
        # invalid_predictions = np.where(cost_matrix.min(axis=1) > self.tau_thr)[0]
        # cost_matrix[invalid_predictions,:] = sys.float_info.max

        cost_matrix[cost_matrix > self.tau_thr] = sys.float_info.max

        # minimize global distances which will assign GT to every prediction
        # (implemented as maximization of 1/cost_matrix where invalid values are marked as 0 instead of inf for numerical stability)
        # print('Linear sum assignment:', cost_matrix.shape)

        pred_ind, gt_ind = scipy.optimize.linear_sum_assignment(1.0 / (cost_matrix + 1e-10), maximize=True)

        #end = time.time()
        #print('Linear sum assignment and cdist done in %.2f!' % (end - start), cost_matrix.shape)

        # remove predictions where distances are over predefined threshold
        valid_predictions = np.where(cost_matrix[pred_ind, gt_ind] < self.tau_thr)[0]
        return pred_ind[valid_predictions], gt_ind[valid_predictions]

    def add_image_prediction(self, im_name, im_index, im_shape, predictions, predictions_score,
                             gt_instances_ids, gt_centers_dict, gt_difficult, centerdir_gt, return_matched_gt_idx=False):

        gt_centers = [gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())]
        gt_centers = np.array(gt_centers)

        pred_gt_match = np.array([])
        pred_gt_match_idx = np.array([])

        if len(predictions) > 0:
            pred_gt_match = np.zeros(shape=(predictions.shape[0], 1))
            pred_gt_match_idx = np.ones(shape=(predictions.shape[0], 1))*-1

            if len(gt_centers) > 0:
                # prepare matrix with gt-to-detection distances
                if len(predictions) * len(gt_centers) > 10000*10000:
                    # since there are too many samples to handle at once we can split them into groups
                    # where distance between samples of different groups are not less than self.tau_thr and
                    # then can process each group independently
                    grouped_predictions_and_gt_centers = self.split_samples_into_groups(predictions[:,:2], gt_centers[:, [1, 0]],
                                                                                        distance_thr=self.tau_thr, grid_count=(16,16))

                    for group_pred_idx, group_gt_idx in tqdm.tqdm(grouped_predictions_and_gt_centers):
                        if len(group_pred_idx) > 0 and len(group_gt_idx):
                            pred_ind, gt_ind = self._assign_detections_to_groundtruth(predictions[group_pred_idx,:2],
                                                                                      gt_centers[group_gt_idx][:,[1, 0]])

                            pred_gt_match[group_pred_idx[pred_ind]] = 1
                            pred_gt_match_idx[group_pred_idx[pred_ind], 0] = group_gt_idx[gt_ind] - len(predictions)

                else:
                    pred_ind, gt_ind = self._assign_detections_to_groundtruth(predictions[:,:2], gt_centers[:,[1,0]])

                    pred_gt_match[pred_ind] = 1
                    pred_gt_match_idx[pred_ind,0] = gt_ind

            if predictions_score is not None:
                self.all_detections.append(np.stack([predictions_score, pred_gt_match[:,0]], axis=1))

        # remove difficult samples from final metric (does not matter if we did or did not find them)
        num_difficult = dict(gt=0, pred=0)
        if gt_difficult is not None:
            is_difficult_gt = [gt_difficult[np.clip(int(c[0]),0,gt_difficult.shape[0]-1),
                                            np.clip(int(c[1]),0,gt_difficult.shape[1]-1)].item() != 0 for c in gt_centers]
            is_difficult_pred = [is_difficult_gt[int(gt_id)] if gt_id >= 0 else 0 for gt_id in pred_gt_match_idx]

            num_difficult = dict(gt=sum(is_difficult_gt),
                                 pred=sum(is_difficult_pred))

        NUM_PRED = len(predictions) - num_difficult['pred']
        NUM_GT = len(gt_centers) - num_difficult['gt']
        TP = pred_gt_match.sum() - num_difficult['pred']
        FP = NUM_PRED - TP
        FN = NUM_GT - TP

        precision = float(TP) / (TP+FP) if TP+FP > 0 else 0
        recall = float(TP) / (TP+FN) if TP+FN > 0 else 0

        F1 = 2 * (precision*recall) / (precision+recall) if precision+recall > 0 else 0
        ratio = NUM_PRED / NUM_GT if NUM_GT > 0 else 0
        Re = np.abs(NUM_PRED - NUM_GT) / NUM_GT if NUM_GT > 0 else 0
        mae = np.abs(NUM_PRED - NUM_GT)
        rmse = np.abs(NUM_PRED - NUM_GT)**2

        self.metrics['TP'].append(TP)
        self.metrics['FP'].append(FP)
        self.metrics['FN'].append(FN)
        self.metrics['N'].append(NUM_GT)
        self.metrics['P'].append(NUM_PRED)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['F1'].append(F1)
        self.metrics['ratio'].append(ratio)
        self.metrics['Re'].append(Re)
        self.metrics['mae'].append(mae)
        self.metrics['rmse'].append(rmse)

        gt_missed = FN > 0
        pred_missed = FP > 0

        if self.center_ap_eval is not None:
            self.center_ap_eval.add_image_prediction(predictions, predictions_score, gt_instances_ids, gt_centers_dict, gt_difficult)

        if self.display_best_threshold:
            P = list(predictions_score) + int(FN) * [-np.inf]
            Y = list(pred_gt_match[:, 0]) + int(FN) * [1.0]

            AP, F1, precision, recall, thrs = get_AP_and_F1(Y, P)

            best_thr = thrs[np.argmax(F1) - 1]

            fig = plt.figure()
            plt.plot(recall, precision)
            plt.title('Precision-recall (AP=%f, F1=%f)' % (AP, np.max(F1)))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            print('best_thr=%f' % best_thr)


        if self.append_count_to_display_name:
            filename_suffix = 'mae=%02d_fn=%02d_fp=%02d_' % (self.metrics['mae'][-1], self.metrics['FN'][-1], self.metrics['FP'][-1])
        else:
            filename_suffix = ""

        if return_matched_gt_idx:
            return gt_missed, pred_missed, pred_gt_match, filename_suffix, pred_gt_match_idx
        else:
            return gt_missed, pred_missed, pred_gt_match, filename_suffix

    def calc_and_display_final_metrics(self, dataset, print_result=True, plot_result=True, save_dir=None, **kwargs):
        Re = np.array(self.metrics['Re']).mean()
        mae = np.array(self.metrics['mae']).mean()
        rmse = np.array(self.metrics['rmse']).mean()
        ratio = np.array(self.metrics['ratio']).mean()
        AP = np.array(self.metrics['precision']).mean()
        AR = np.array(self.metrics['recall']).mean()
        F1 = np.array(self.metrics['F1']).mean()

        if print_result:
            print('Re=%.4f, mae=%.4f, rmse=%.4f, ratio=%.4f, AP=%.4f, AR=%.4f, F1=%.4f' % (Re, mae, rmse, ratio, AP, AR, F1))

        if len(self.all_detections) > 0:
            all_detections = np.concatenate(self.all_detections,axis=0)

        if self.center_ap_eval is not None:
            metrics_mAP = self.center_ap_eval.calc_and_display_final_metrics(print_result, plot_result)
        else:
            metrics_mAP = None, None

        metrics = dict(AP=AP, AR=AR, F1=F1, ratio=ratio, Re=Re, mae=mae, rmse=rmse, all_images=self.metrics, metrics_mAP=metrics_mAP)

        ########################################################################################################
        # SAVE EVAL RESULTS TO JSON FILE
        if metrics is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, 'results.json'), 'w') as file:
                file.write(json.dumps(metrics, cls=NumpyEncoder))

        return metrics

    def get_results_timestamp(self, save_dir):
        res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'results.json')

        return os.path.getmtime(res_filename) if os.path.exists(res_filename) else 0

    @staticmethod
    def split_samples_into_groups(predictions, gt_centers, distance_thr, grid_count=(8, 8)):

        def _find_neighboring_samples(existing_loc, neighbor_loc, neighbor_idx):
            dist_matrix = torch.cdist(torch.from_numpy(existing_loc), torch.from_numpy(neighbor_loc)).cpu().numpy()

            all_new_neighbor_idx = []
            added_neighbors = np.any(dist_matrix < distance_thr, axis=0)
            dist_matrix = None
            if any(added_neighbors):

                return np.concatenate((neighbor_idx[added_neighbors],
                                       _find_neighboring_samples(neighbor_loc[added_neighbors],
                                                                 neighbor_loc[~added_neighbors],
                                                                 neighbor_idx[~added_neighbors])), axis=0)
            else:
                return np.array([], dtype=np.int64)

        if len(predictions) > 0 and len(gt_centers) > 0:
            all_samples = np.concatenate((predictions, gt_centers), axis=0)
        elif len(predictions) > 0:
            all_samples = predictions
        elif len(gt_centers) > 0:
            all_samples = gt_centers
        else:
            return []

        # split predictions and gt_centers into grids
        max_loc = np.max(all_samples, axis=0)
        min_loc = np.min(all_samples, axis=0)
        grid_count = np.array(grid_count)
        grid_size = (max_loc - min_loc + 2) / grid_count

        samples_grid_loc = (all_samples[:, :2] - min_loc) // grid_size

        clustered_selections = []

        grid_data = [[dict() for j in range(grid_count[1])] for i in range(grid_count[0])]
        for i in range(grid_count[0]):
            for j in range(grid_count[1]):
                neighbors = [(i + 0, j + 1), (i + 1, j + 0), (i + 1, j + 1)]

                sel_idx = np.where(np.all(samples_grid_loc == np.array([i, j]), axis=1))[0]

                current_pos = all_samples[sel_idx, :]

                border_idx = \
                    np.where(np.any(current_pos + distance_thr > np.array([i + 1, j + 1]) * grid_size, axis=1))[0]

                grid_data[i][j] = dict(idx=sel_idx, neighbors=neighbors, border_loc=current_pos[border_idx, :])

        for i in range(grid_count[0]):
            for j in range(grid_count[1]):

                current_block = grid_data[i][j]
                sel_idx = current_block['idx']

                # extend with any neighboring data samples that are close to existing samples
                neighbors = current_block['neighbors']
                existing_neighbors = []

                while len(neighbors) > 0:
                    new_neighbors = []
                    for n_i, n_j in neighbors:
                        if n_i >= grid_count[0] or n_j >= grid_count[1]:
                            continue
                        neighbors_block = grid_data[n_i][n_j]

                        if len(neighbors_block['idx']) > 0:
                            new_idx = _find_neighboring_samples(current_block['border_loc'],
                                                                all_samples[neighbors_block['idx']],
                                                                neighbors_block['idx'])

                            if len(new_idx) > 0:
                                # add new bordered predictions to current block check
                                sel_idx = np.concatenate((sel_idx, new_idx))
                                # then remove them from neighbors_block since they will be checked now
                                neighbors_block['idx'] = np.array(list(set(neighbors_block['idx']) - set(new_idx)))

                                # add to list of new neighbors that need to be checked in next iteration
                                border_flags = (all_samples[new_idx] + distance_thr > np.array(
                                    [n_i + 1, n_j + 1]) * grid_size).astype(np.int32)
                                for next_i, next_j in np.unique(border_flags, axis=0):
                                    if next_i > 0 or next_j > 0:
                                        new_neighbors.append((n_i + next_i, n_j + next_j))

                    existing_neighbors.extend(neighbors)
                    if len(new_neighbors) > 0:
                        neighbors = [N for N in np.unique(new_neighbors, axis=0) if
                                     not any(np.all(N == existing_neighbors, axis=1))]
                    else:
                        neighbors = []

                current_block['idx'] = sel_idx
                clustered_selections.append(sel_idx.astype(np.int64))

        if False:
            clustered_selections_lens = np.zeros((len(clustered_selections), len(clustered_selections)))
            for i, sel_idx in enumerate(clustered_selections):
                for j in range(i + 1, len(clustered_selections)):
                    dist_matrix = torch.cdist(torch.from_numpy(all_samples[sel_idx]),
                                              torch.from_numpy(all_samples[clustered_selections[j]])).cpu().numpy()
                    clustered_selections_lens[i, j] = np.sum(dist_matrix < distance_thr)

            assert np.sum(clustered_selections_lens) == 0

        # return assigned predictions and groups as ids of original arrays (i.e. relative to predictions and gt_centers)
        return [(sel_idx[sel_idx < len(predictions)],  # prediction ids
                 sel_idx[sel_idx >= len(predictions)] - len(predictions))  # gt_centers ids
                for sel_idx in clustered_selections if
                sum(sel_idx < len(predictions)) > 0 or sum(sel_idx >= len(predictions)) > 0]