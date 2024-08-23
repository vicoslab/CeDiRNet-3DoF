import numpy as np

import torch
import torch.nn as nn

from criterions.per_pixel_losses import get_per_pixel_loss_func

class CenterLocalizationLoss(nn.Module):

    def __init__(self, loss_type='l1', ignore_negative_gradient=False, fp_threshold=0.1,
                 use_per_instance_normalization=False, positive_area_radius=1, **kargs):
        super().__init__()

        self.ignore_negative_gradient = ignore_negative_gradient
        self.fp_threshold = fp_threshold # 0.1
        self.use_per_instance_normalization = use_per_instance_normalization
        self.positive_area_radius = positive_area_radius

        self.loss_fn = get_per_pixel_loss_func(loss_type)

    def forward(self, prediction_centers, prediction_prob, gt_centers, gt_center_mask, ignore_mask=None,
                w_fg=1, w_bg=1, reduction_dims=(1,2,3), **kwargs):

        batch_size, height, width = prediction_prob.size(0), prediction_prob.size(2), prediction_prob.size(3)

        loss_centers = torch.zeros(size=[d for i,d in enumerate(prediction_prob.shape) if i not in reduction_dims],
                                   device=prediction_prob.device)

        with torch.no_grad():
            if self.use_per_instance_normalization:
                mask_weights = self._calc_per_instance_weight_mask(prediction_centers, prediction_prob, gt_centers,
                                                                   thr=self.fp_threshold, N=100, R=self.positive_area_radius)
            else:
                mask_weights = torch.ones_like(prediction_prob, requires_grad=False, device=prediction_prob.device, dtype=torch.float32)
                mask_weights *= 1.0 / (height * width * batch_size)

            if ignore_mask is not None:
                mask_weights *= 1 - ignore_mask.type(mask_weights.type())

            if w_fg != 1:
                mask_weights[gt_center_mask > 0] *= w_fg
            if w_bg != 1:
                mask_weights[gt_center_mask <= 0] *= w_bg

        if self.ignore_negative_gradient:
            loss_centers += torch.sum(mask_weights * torch.where((prediction_prob <= 0) * (gt_center_mask <= 0),
                                                                 torch.zeros_like(prediction_prob, requires_grad=False),
                                                                 self.loss_fn(prediction_prob, gt_center_mask)),
                                      dim=reduction_dims)
        else:
            loss_centers += torch.sum(mask_weights * self.loss_fn(prediction_prob, gt_center_mask),
                                      dim=reduction_dims)

        return loss_centers

    @staticmethod
    def _calc_per_instance_weight_mask(prediction_centers, prediction_prob, gt_centers, thr, N, R):
        batch_size, height, width = prediction_prob.size(0), prediction_prob.size(2), prediction_prob.size(3)

        mask_weights = torch.zeros_like(prediction_prob, requires_grad=False, device=prediction_prob.device, dtype=torch.float32)

        centers_pred = np.array(prediction_centers)
        for b in range(batch_size):
            # mark positive groundtruth areas
            for x, y in gt_centers[b, 1:]:
                if x == 0 and y == 0: break
                x, y = int(x), int(y)
                mask_weights[b, :, x - R:x + R, y - R:y + R] = 1

            # find hard-negative centers if exist any
            if len(centers_pred) <= 0:
                continue

            batch_centers = centers_pred[centers_pred[:, 0] == b, :]

            if len(batch_centers) <= 0:
                continue

            ids = np.argsort(batch_centers[:, -1])[::-1]

            hard_neg_centers = batch_centers[ids, :]
            hard_neg_centers = hard_neg_centers[hard_neg_centers[:, -1] > thr]
            if len(hard_neg_centers) > N:
                hard_neg_centers = hard_neg_centers[:N, :]

            # mark hard-negative areas
            for x, y in hard_neg_centers[:, 1:3]:
                x, y = int(x), int(y)
                mask_weights[b, :, y - R:y + R, x - R:x + R] = 1

        # original version
        if True:
            hard_neg_pixels = max(mask_weights.sum().item(), 1.0)

            mask_weights *= (1.0 / hard_neg_pixels - 1 / (
                    height * width * batch_size - hard_neg_pixels))
            mask_weights += 1 / (height * width * batch_size - hard_neg_pixels)

        # using only FP pixels
        if False:
            hard_neg_pixels = mask_weights.sum().item()
            if hard_neg_pixels > 0:
                mask_weights *= (1.0 / hard_neg_pixels)

        return mask_weights
