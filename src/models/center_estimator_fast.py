import torch
import torch.nn as nn

import numpy as np

from models.localization.centers import Conv1dMultiscaleLocalization, Conv2dDilatedLocalization

class CenterEstimatorFast(nn.Module):
    def __init__(self, args=dict(), is_learnable=True):
        super().__init__()

        instance_center_estimator_op = Conv1dMultiscaleLocalization
        if args.get('use_dilated_nn'):
            from functools import partial
            instance_center_estimator_op = partial(Conv2dDilatedLocalization,
                                                   **args.get('dilated_nn_args',{}))

        self.instance_center_estimator = instance_center_estimator_op(
            local_max_thr=args.get('local_max_thr', 0.1),
            mask_thr=args.get('mask_thr', 0.01),
            exclude_border_px=args.get('exclude_border_px', 5),
            learnable=is_learnable,
            allow_input_backprop=args.get('allow_input_backprop', True),
            backprop_only_positive=args.get('backprop_only_positive', True),
            apply_input_smoothing_for_local_max=0,
            use_findcontours_for_local_max=args.get('use_findcontours_for_local_max', False),
            local_max_min_dist=1,
            return_time=True
        )

    def set_return_backbone_only(self, val):
        pass

    def is_return_backbone_only(self):
        return False

    def init_output(self, num_vector_fields=1):
        self.num_vector_fields = num_vector_fields

        assert self.num_vector_fields >= 3
        self.instance_center_estimator.init_output()

        return input

    def forward(self, input):

        assert input.shape[1] >= self.num_vector_fields

        predictions = input[:, 0:self.num_vector_fields]

        S = predictions[:, 0].unsqueeze(1)
        C = predictions[:, 1].unsqueeze(1)

        res, _, times = self.instance_center_estimator(C, S, None, None, None, ignore_region=None)
        center_pred = res

        return center_pred, times