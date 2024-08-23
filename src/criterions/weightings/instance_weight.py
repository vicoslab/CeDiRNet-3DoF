import torch

import numpy as np

from criterions.weightings.unbalanced_weight import UnbalancedWeighting

class InstanceGroupWeighting(UnbalancedWeighting):
    IGNORE_FLAG=-9999

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, gt_instances, gt_ignore=None, gt_R=None, w_fg=1, w_bg=1, *args, **kwargs):

        batch_size, height, width = gt_instances.shape

        bg_mask = (gt_instances == 0).unsqueeze(1)
        fg_mask = bg_mask == False

        mask_weights = torch.ones_like(bg_mask, dtype=torch.float32, requires_grad=False, device=gt_instances.device)

        mask_weights[fg_mask] = w_fg
        mask_weights[bg_mask] = w_bg

        # apply additional weights around borders
        if self.border_weight_px > 0:
            mask_weights = self._apply_border_weights(mask_weights)

        if gt_ignore is not None:
            mask_weights *= 1 - gt_ignore.type(mask_weights.type())

        # ensure each instance (and background) is weighted equally regardless of pixels size
        # count number of pixels per instance
        instance_ids, instance_sizes = gt_instances.reshape(gt_instances.shape[0], -1).unique(return_counts=True, dim=-1)

        # count number of instance for each batch element (without background and ignored regions)
        num_bg_pixels = instance_sizes.repeat(batch_size, 1)[instance_ids == 0].sum().float()

        mask_weights = self._init_grouped_weights(mask_weights, gt_instances, instance_ids, instance_sizes, num_bg_pixels)

        # apply additional weight based on distance to center
        if self.add_distance_gauss_weight > 0:
            mask_weights = self._apply_gauss_distance_weights(mask_weights, gt_R)

        return mask_weights

    def _init_grouped_weights(self, W, group_instance, group_instance_ids, group_instance_sizes, num_bg_pixels,
                              num_hard_negative_pixels=torch.tensor(0.0)):
        # ensure each instance (and background) is weighted equally regardless of pixels size
        num_instances = sum([len(set(ids.unique().cpu().numpy()) - set([0, self.IGNORE_FLAG])) for ids in group_instance])
        for b in range(len(group_instance)):
            for id in group_instance_ids[b].unique():
                mask_id = group_instance[b].eq(id).unsqueeze(0)
                if id == 0:
                    # for BG instance we normalize based on the number of all bg pixels over the whole batch
                    instance_normalization = num_bg_pixels * 1
                    instance_normalization = instance_normalization * (
                        3 / 1.0 if num_hard_negative_pixels > 0 else 2)
                elif id < 0:
                    if num_hard_negative_pixels > 0:
                        # for hard-negative instances we normalized based on number of them (in pixels)
                        instance_normalization = num_hard_negative_pixels * torch.log(num_hard_negative_pixels + 1)
                        instance_normalization = instance_normalization * 3 / 1.0
                    else:
                        instance_normalization = 1.0
                else:
                    # for FG instances we normalized based on the size of instance (in pixel) and the number of
                    # instances over the whole batch
                    instance_pixels = group_instance_sizes[group_instance_ids[b] == id].sum().float()
                    instance_normalization = instance_pixels * num_instances * 1
                    instance_normalization = instance_normalization * ( 3 / 1.0 if num_hard_negative_pixels > 0 else 2)

                # BG and FG are treated as equal so add multiplication by 2 (or 3 if we also have hard-negatives)
                # instance_normalization = instance_normalization * _N
                W[b][mask_id] *= 1.0 / instance_normalization
        return W

