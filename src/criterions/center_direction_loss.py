import numpy as np

import torch
import torch.nn as nn

from functools import partial

from models.center_groundtruth import CenterDirGroundtruth

from criterions.weightings.unbalanced_weight import UnbalancedWeighting
from criterions.weightings.instance_weight import InstanceGroupWeighting

from criterions.per_pixel_losses import get_per_pixel_loss_func

from criterions.center_localization_loss import CenterLocalizationLoss

class CenterDirectionLoss(nn.Module):

    def __init__(self, model, num_vector_fields=3,
                 no_instance_loss=True, enable_centerdir_loss=True,
                 regression_loss=None, centerdir_instance_weighted=True, loss_weighted_by_distance_gauss=0,
                 use_log_r=True, use_log_r_base='exp',
                 learnable_center_est=False, learnable_center_loss='l1',
                 learnable_center_ignore_negative_gradient=False, learnable_center_fp_threshold=0.1,
                 learnable_center_with_instance_norm=True, learnable_center_positive_area_radius=1,
                 border_weight=1.0, border_weight_px=0, **kargs):
        super().__init__()

        assert num_vector_fields >= 3

        self.num_vector_fields = num_vector_fields

        ################################
        # enable main settings (cls loss, instance loss, localization loss)
        self.no_instance_loss = no_instance_loss                        # False == using only FG for instances, True == use FG and BG

        self.enable_centerdir_loss = enable_centerdir_loss
        self.localization_criteria = None
        if learnable_center_est:
            self.localization_criteria = CenterLocalizationLoss(loss_type=learnable_center_loss,
                                                                ignore_negative_gradient=learnable_center_ignore_negative_gradient,
                                                                fp_threshold=learnable_center_fp_threshold,
                                                                use_per_instance_normalization=learnable_center_with_instance_norm,
                                                                positive_area_radius=learnable_center_positive_area_radius)

        ################################
        # equalize importance of pixels from instances: pixels in instances should be considered equally
        # important as bg pixels regardless of thier size
        self.centerdir_instance_weighted = centerdir_instance_weighted          # normalize losses to per-instance case for regressopm

        ################################
        # main losses (cls and regression)
        # new way to define loss using dict(name='NAME',args=...)
        if regression_loss is not None:
            self.regression_loss = regression_loss

        self.use_log_r = use_log_r                                      # use logarithm of radii as groundtruth

        ################################################################
        # Prepare log and inverse log functions for R regression

        if use_log_r_base.lower() in ['exp', 'e']:
            self.log_r_fn = lambda x: torch.log(x + 1)
            self.inverse_log_r_fn = lambda x: torch.exp(x) - 1
        elif use_log_r_base.lower() in ['decimal', '10']:
            self.log_r_fn = lambda x: torch.log10(x + 1)
            self.inverse_log_r_fn = lambda x: torch.pow(10, x) - 1
        elif use_log_r_base.lower() in ['pow10']:
            self.log_r_fn =  lambda x: torch.log10(x + 1)
            self.inverse_log_r_fn = lambda x: torch.pow(x, 10) - 1
        else:
            raise Exception('Only "exp" and "10" are allowed logarithms for R')

        ################################################################
        # Prepare all loss functions
        self.regression_loss_fn = get_per_pixel_loss_func(self.regression_loss)

        ################################################################
        # Prepare classes for calculating weight masks
        def get_weighting_op(is_instance_weighted):
            if is_instance_weighted:
                return InstanceGroupWeighting(border_weight=border_weight, border_weight_px=border_weight_px,
                                              add_distance_gauss_weight=loss_weighted_by_distance_gauss)
            else:
                return UnbalancedWeighting(border_weight=border_weight, border_weight_px=border_weight_px,
                                           add_distance_gauss_weight=loss_weighted_by_distance_gauss)

        self.dir_weighting = get_weighting_op(self.centerdir_instance_weighted)


        self.tmp = nn.Conv2d(8,8,3)

    def forward(self, prediction, sample, centerdir_responses=None, centerdir_gt=None, ignore_mask=None, difficult_mask=None,
                w_r=1, w_cos=1, w_sin=1, w_fg=1, w_bg=1, w_cent=1, w_fg_cent=1, w_bg_cent=1,
                reduction_dims=(1,2,3), **kwargs):

        loss_output_shape = [d for i,d in enumerate(prediction.shape) if i not in reduction_dims]
        loss_zero_init = lambda: torch.zeros(size=loss_output_shape,device=prediction.device)

        loss_cls, loss_sin, loss_cos, loss_r = map(torch.clone,[loss_zero_init()]*4)  # centerdir_vectors and cls losses

        loss_centers = loss_zero_init()

        loss_magnitude_reg = loss_zero_init()

        instances = sample["instance"]
        instances = instances.squeeze(1)

        # batch computation ---
        labels = sample["label"]
        bg_mask = labels == 0
        fg_mask = bg_mask == False

        centerdir_vectors = prediction[:, 0:self.num_vector_fields]

        prediction_sin = centerdir_vectors[:, 0].unsqueeze(1)
        prediction_cos = centerdir_vectors[:, 1].unsqueeze(1)
        prediction_R = centerdir_vectors[:, 2].unsqueeze(1)

        if instances.dtype != torch.int16:
            instances = instances.type(torch.int16)

        # mark ignore regions as -9999 in instances so that size can be correctly calculated in InstanceGroupWeighting
        if ignore_mask is not None:
            instances = instances.clone() # do not destroy original
            instances[ignore_mask.squeeze(dim=1) == 1] = InstanceGroupWeighting.IGNORE_FLAG

        # retrieve groundtruth values (either computed or from cache)
        gt_R, _, gt_sin_th, gt_cos_th,\
            gt_centers, gt_center_ignore, gt_center_mask, _ = CenterDirGroundtruth.parse_groundtruth(centerdir_gt)

        if self.use_log_r:
            gt_R = self.log_r_fn(gt_R)

        if centerdir_responses is not None:
            centers_pred, center_heatmap = centerdir_responses
            centers_pred = self._unroll_center_predictions(centers_pred)
        else:
            centers_pred, center_heatmap = None, None
        
        # prepare all arguments that are needed for calculating weighting mask
        weighting_args = dict(gt_instances=instances, gt_centers=gt_centers,
                              predicted_centers=centers_pred,
                              gt_ignore=ignore_mask, gt_difficult=difficult_mask, gt_R=gt_R,
                              w_fg=w_fg, w_bg=w_bg)

        ######################################################
        ### centerdir_vectors losses (cos, sin)
        if self.enable_centerdir_loss:
            with torch.no_grad():
                mask_weights = self.dir_weighting(**weighting_args)

                # we need to ignore center parts since they are often wrong
                mask_weights *= gt_center_ignore.float()

            # add regression loss for sin(x), cos(x) and R
            if self.no_instance_loss:
                if w_sin != 0:
                    loss_sin += torch.sum(mask_weights * self.regression_loss_fn(prediction_sin, gt_sin_th), dim=reduction_dims)
                if w_cos != 0:
                    loss_cos += torch.sum(mask_weights * self.regression_loss_fn(prediction_cos, gt_cos_th), dim=reduction_dims)
                if w_r != 0:
                    loss_r += torch.sum(mask_weights * self.regression_loss_fn(prediction_R, gt_R), dim=reduction_dims)

            else:
                for b in range(mask_weights.shape[0]):
                    fg_mask_weights = mask_weights[b][fg_mask[b]]

                    if w_sin != 0:
                        loss_sin[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_sin[b][fg_mask[b]], gt_sin_th[b][fg_mask[b]]))
                    if w_cos != 0:
                        loss_cos[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_cos[b][fg_mask[b]], gt_cos_th[b][fg_mask[b]]))
                    if w_r != 0:
                        loss_r[b] += torch.sum(fg_mask_weights * self.regression_loss_fn(prediction_R[b][fg_mask[b]], gt_R[b][fg_mask[b]]))

        ######################################################
        ###  localization loss for estimating center from centerdir_vectors outputs
        if self.localization_criteria is not None and w_cent != 0:
            loss_centers = self.localization_criteria(centers_pred, center_heatmap, gt_centers, gt_center_mask, ignore_mask,
                                                      w_fg=w_fg_cent, w_bg=w_bg_cent, reduction_dims=reduction_dims)


        loss_sin = w_sin * loss_sin
        loss_cos = w_cos * loss_cos
        loss_r = w_r * loss_r # THIS SHOULD BE ZERO BUT WAS NOT !!!

        loss_centerdir_total = loss_sin + loss_cos + loss_r + loss_magnitude_reg

        # total/final loss:
        loss = loss_cls + loss_centerdir_total + loss_centers

        # add epsilon as a way to force values to tensor/cuda
        eps = prediction.sum() * 0
        # convert all losses to tensors to ensure proper parallelization with torch.nn.DataParallel
        losses = [t + eps for t in [loss, loss_cls, loss_centerdir_total, loss_centers,
                                    loss_sin, loss_cos, loss_r,
                                    loss_magnitude_reg]]

        return tuple(losses)

    def get_loss_dict(self, loss_tensor):
        # return dict(loss=loss_tensor[0].mean())

        loss, loss_cls, loss_centerdir_total, loss_centers, \
        loss_sin, loss_cos, loss_r, \
        loss_magnitude_reg = [l.sum() for l in loss_tensor]

        return dict(  # main loss for backprop:
            loss=loss,
            # losses for visualization:
            losses_groups=dict(cls=loss_cls, centerdir_total=loss_centerdir_total, centers=loss_centers),
            losses_centerdir_total=dict(sin=loss_sin, cos=loss_cos, r=loss_r, magnitude_reg=loss_magnitude_reg),
            losses_main=dict(cls=loss_cls, sin=loss_sin, cos=loss_cos, r=loss_r, cent=loss_centers))

    def _unroll_center_predictions(self, centers_pred):
        if type(centers_pred) == torch.Tensor:
            centers_pred_res = []
            for b, c in enumerate(centers_pred.cpu().numpy()):
                valid_center_idx = np.where(c[:, 0] != 0)[0].astype(np.int32)
                centers_pred_res.append(np.concatenate((np.ones((len(valid_center_idx), 1)) * b,
                                                        c[valid_center_idx, 1:]),
                                                       axis=1))
            centers_pred = np.concatenate(centers_pred_res, axis=0)
        return centers_pred
