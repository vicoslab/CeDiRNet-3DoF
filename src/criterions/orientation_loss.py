import torch
import torch.nn as nn

from models.center_groundtruth import CenterDirGroundtruth

from criterions.weightings.instance_weight import InstanceGroupWeighting
from criterions.center_direction_loss import CenterDirectionLoss

from criterions.per_pixel_losses import get_per_pixel_loss_func

class OrientationLoss(nn.Module):
    def __init__(self, model, orientation_args=dict(), **center_dir_args):
        super().__init__()

        self.enable_orientation_loss = orientation_args.get('enable')
        self.regress_confidence_score = orientation_args.get('regress_confidence_score')
        self.no_instance_loss = orientation_args.get('no_instance_loss')
        self.enable_6dof = orientation_args.get('enable_6dof')
        self.loss_weighting = orientation_args.get('loss_weighting')

        self.centerdir_loss_op = CenterDirectionLoss(model, **center_dir_args)

        self.use_log_r = self.centerdir_loss_op.use_log_r
        self.log_r_fn = self.centerdir_loss_op.log_r_fn
        self.num_vector_fields = self.centerdir_loss_op.num_vector_fields
        
        REQUIRED_VECTOR_FIELDS = 5
        if self.enable_6dof:
            REQUIRED_VECTOR_FIELDS += 4 
        if self.regress_confidence_score:
            REQUIRED_VECTOR_FIELDS += 1
        
        assert self.num_vector_fields >= REQUIRED_VECTOR_FIELDS

        self.regression_loss_fn = get_per_pixel_loss_func(orientation_args.get('regression_loss'))

        self.orientation_weighting = InstanceGroupWeighting(border_weight=center_dir_args.get('border_weight',1.0),
                                                            border_weight_px=center_dir_args.get('border_weight_px',0),
                                                            add_distance_gauss_weight=False)
        self.tmp = nn.Conv2d(8,8,3)

    def forward(self, prediction, sample, centerdir_responses=None, centerdir_gt=None, ignore_mask=None,
                difficult_mask=None, w_orientation=1, w_fg_orientation=1, w_bg_orientation=1, w_confidence_score=1, reduction_dims=(1, 2, 3), **kwargs):

        loss_output_shape = [d for i, d in enumerate(prediction.shape) if i not in reduction_dims]
        loss_zero_init = lambda: torch.zeros(size=loss_output_shape, device=prediction.device)

        loss_sin_orientation, loss_cos_orientation, loss_confidence_score = map(torch.clone, [loss_zero_init()] * 3)
        
        instances = sample["instance"]
        instances = instances.squeeze(1)

        # batch computation ---
        labels = sample["label"]
        bg_mask = labels == 0
        fg_mask = bg_mask == False

        centerdir_vectors = prediction[:, 0:self.num_vector_fields]

        # WARNING: this assumes CenterDirectionLoss is used as parent (not compatible with other)
        if self.enable_6dof:
            prediction_sin_orientation = centerdir_vectors[:, 3:6].unsqueeze(2)
            prediction_cos_orientation = centerdir_vectors[:, 6:9].unsqueeze(2)
        else:
            prediction_sin_orientation = centerdir_vectors[:, 3:4].unsqueeze(2)
            prediction_cos_orientation = centerdir_vectors[:, 4:5].unsqueeze(2)


        if instances.dtype != torch.int16:
            instances = instances.type(torch.int16)

        # mark ignore regions as -9999 in instances so that size can be correctly calculated in InstanceGroupWeighting
        if ignore_mask is not None:
            instances = instances.clone()  # do not destroy original
            instances[ignore_mask.squeeze(dim=1) == 1] = InstanceGroupWeighting.IGNORE_FLAG

        # retrieve groundtruth values
        key = ['gt_orientation_sin', 'gt_orientation_cos']
        gt_sin_orientation, gt_cos_orientation = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=key)

        assert prediction_sin_orientation.shape[1] == prediction_cos_orientation.shape[1]

        ROT_DIM = prediction_sin_orientation.shape[1]

        assert ROT_DIM <= gt_sin_orientation.shape[1]
        assert ROT_DIM <= gt_cos_orientation.shape[1]

        gt_sin_orientation = gt_sin_orientation[:, :ROT_DIM]
        gt_cos_orientation = gt_cos_orientation[:, :ROT_DIM]

        assert prediction_sin_orientation.shape[1] == gt_sin_orientation.shape[1]
        assert prediction_cos_orientation.shape[1] == gt_cos_orientation.shape[1]

        if self.regress_confidence_score:
            gt_confidence_score = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt,keys=['gt_confidence_score'])

            prediction_confidence_score = centerdir_vectors[:, 9:10] if self.enable_6dof else centerdir_vectors[:, 5:6]
            prediction_confidence_score = prediction_confidence_score.unsqueeze(2)


        # prepare all arguments that are needed for calculating weighting mask
        weighting_args = dict(gt_instances=instances,  gt_ignore=ignore_mask, gt_difficult=difficult_mask,
                              w_fg=w_fg_orientation, w_bg=w_bg_orientation)

        ######################################################
        ### centerdir_vectors losses (cos, sin)
        loss_sin_orientation, loss_cos_orientation = zip(*[(loss_sin_orientation.clone(), loss_cos_orientation.clone()) for _ in range(ROT_DIM)])
        
        if self.enable_orientation_loss:

            with torch.no_grad():
                mask_weights = self.orientation_weighting(**weighting_args)

            # add regression loss for sin(orientation), cos(orientation)
            if self.no_instance_loss:

                loss_sin_orientation += torch.sum(
                    mask_weights * self.regression_loss_fn(prediction_sin_orientation, gt_sin_orientation),
                    dim=reduction_dims)
                loss_cos_orientation += torch.sum(
                    mask_weights * self.regression_loss_fn(prediction_cos_orientation, gt_cos_orientation),
                    dim=reduction_dims)
                
                if self.regress_confidence_score:
                    loss_confidence_score += torch.sum(
                        mask_weights * self.regression_loss_fn(prediction_confidence_score, gt_confidence_score),
                        dim=reduction_dims)

            else:
                for b in range(mask_weights.shape[0]):
                    fg_mask_weights = mask_weights[b][fg_mask[b]]
                    for rot_dim in range(ROT_DIM):
                        loss_sin_orientation[rot_dim][b] += torch.sum(
                            fg_mask_weights * self.regression_loss_fn(prediction_sin_orientation[b][rot_dim,fg_mask[b]],
                                                                      gt_sin_orientation[b][rot_dim,fg_mask[b]]))
                        loss_cos_orientation[rot_dim][b] += torch.sum(
                            fg_mask_weights * self.regression_loss_fn(prediction_cos_orientation[b][rot_dim,fg_mask[b]],
                                                                      gt_cos_orientation[b][rot_dim,fg_mask[b]]))
                    if self.regress_confidence_score:
                        loss_confidence_score[b] += torch.sum(
                            fg_mask_weights * self.regression_loss_fn(prediction_confidence_score[b][0,fg_mask[b]],
                                                                        gt_confidence_score[b][0,fg_mask[b]]))

        loss_sin_orientation = [w_orientation*l for l in loss_sin_orientation]
        loss_cos_orientation = [w_orientation*l for l in loss_cos_orientation]

        loss_confidence_score = [w_confidence_score*l for l in loss_confidence_score]

        loss_orientation = sum(loss_sin_orientation) + sum(loss_cos_orientation) + sum(loss_confidence_score)
        loss_orientation += prediction.sum() * 0

        # call base loss function for center direction
        all_centerdir_losses = self.centerdir_loss_op.forward(prediction, sample, centerdir_responses,
                                                              centerdir_gt, ignore_mask, difficult_mask,
                                                              reduction_dims=reduction_dims, **kwargs)

        losses_main = [all_centerdir_losses[0] + loss_orientation]

        return tuple(losses_main + list(all_centerdir_losses[1:]) + [loss_orientation] + list(loss_sin_orientation) + list(loss_cos_orientation))


    def get_loss_dict(self, loss_tensor):

        loss, loss_cls, loss_centerdir_total, loss_centers, loss_sin, \
        loss_cos, loss_r, loss_magnitude_reg, loss_orientation_total = [l.sum() for l in loss_tensor[:9]]

        loss_orientation = [l.sum() for l in loss_tensor[9:]]
        loss_orientation_sin = loss_orientation[:len(loss_orientation) // 2] # first half is sin
        loss_orientation_cos = loss_orientation[len(loss_orientation) // 2:] # second half is cos
        orientation_dims = ['rot_y','rot_z','rot_x']

        return dict(  # main loss for backprop:
            loss=loss,
            # losses for visualization:
            losses_groups=dict(cls=loss_cls, centerdir_total=loss_centerdir_total, centers=loss_centers, orientation_total=loss_orientation_total),
            losses_centerdir_total=dict(sin=loss_sin, cos=loss_cos, r=loss_r, magnitude_reg=loss_magnitude_reg),
            losses_orientation_total={**{f'{orientation_dims[i]}_sin':l for i,l in enumerate(loss_orientation_sin)},
                                      **{f'{orientation_dims[i]}_cos':l for i,l in enumerate(loss_orientation_cos)}},
            losses_main=dict(cls=loss_cls, sin=loss_sin, cos=loss_cos, r=loss_r, cent=loss_centers, ),
            # losses for task weighting:
            losses_tasks=dict(centerdir=loss_centerdir_total,
                              **{f'orinetation_{orientation_dims[i]}':(l_sin+l_cos)
                                    for i,(l_sin,l_cos) in enumerate(zip(loss_orientation_sin,loss_orientation_cos))}),
        )
