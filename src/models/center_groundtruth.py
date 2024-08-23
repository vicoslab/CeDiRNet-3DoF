import os

import numpy as np
import sys
import torch
from torch import nn as nn

from utils.utils import GaussianLayer


class CenterDirGroundtruth(nn.Module):
    def __init__(self, centerdir_gt_cache=None, extend_instance_mask_weights=0, MAX_NUM_CENTERS=100,
                 backbone_output_cache=None, load_cached_backbone_output_probability=0,
                 use_cached_backbone_output=False, save_cached_backbone_output_only=False,
                 add_synthetic_output=False, ignore_instance_mask_and_use_closest_center=False,
                 center_ignore_px=3, center_gt_blur=2, skip_gt_center_mask_generate=False):
        super().__init__()

        self.ignore_instance_mask_and_use_closest_center = ignore_instance_mask_and_use_closest_center
        
        self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

        self.center_ignore_px = center_ignore_px
        self.skip_gt_center_mask_generate = skip_gt_center_mask_generate
        ################################################################
        # Prepare location/coordinate map
        xym = self._create_xym(0)

        self.register_buffer("xym", xym, persistent=False)

        with torch.no_grad():
            self.gaussian_blur = GaussianLayer(num_channels=1, sigma=center_gt_blur)

    def _create_xym(self, size):
        # coordinate map
        # CAUTION: original code may not have correctly aligned offsets
        #          since xm[-1] will not be size-1, but size
        #          -> this has been fixed by adding +1 element to xm and ym array
        align_fix = 1

        xm = torch.linspace(0, 1, size + align_fix).view(1, 1, -1).expand(1, size + align_fix, size + align_fix) * size
        ym = torch.linspace(0, 1, size + align_fix).view(1, -1, 1).expand(1, size + align_fix, size + align_fix) * size
        return torch.cat((xm, ym), 0)

    def _get_xym(self, height, width):
        max_size = max(height, width)
        if max_size > min(self.xym.shape[1], self.xym.shape[2]):
            self.xym = self._create_xym(max_size).to(self.xym.device)

        return self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

    def forward(self, sample, batch_index):
        instances = sample['instance']
        label = sample['label']
        sample_name = sample['im_name']

        centers = sample.get('center')
        centerdir_gt = sample.get('centerdir_groundtruth')
        output = sample.get('output')
        orientation = sample.get('orientation')

        # generate any missing groundtruth and store it
        centerdir_gt = self._generate_and_store_groundtruth(centerdir_gt, instances, centers, sample_name, batch_index, orientation)

        if 'instance_polygon' in sample:
            instance_polygon = sample['instance_polygon']
            centerdir_gt.append(instance_polygon)

        # attach centerdir groundtruh and output values to returned sample
        sample['centerdir_groundtruth'] = centerdir_gt

        return sample

    def _generate_and_store_groundtruth(self, centerdir_gt, instances, centers, sample_name, sample_name_idx, orientation):
        with torch.no_grad():
            batch_size, _, height, width = instances.shape

            # remove single dimensions
            instances = instances[:,0]

            xym_s = self._get_xym(height, width)

            if centerdir_gt is None:
                centerdir_gt = []

            if len(centerdir_gt) < 5:
                centerdir_gt_matrix = torch.zeros(size=[batch_size, 13, 1, height, width], dtype=torch.float, device=xym_s.device, requires_grad=False)
                centerdir_gt_present = [0]*batch_size

                extended_instances_num = [0]*batch_size
                extended_instances = torch.zeros([batch_size, height, width], dtype=torch.int16, device=xym_s.device, requires_grad=False)

                if centers is None:
                    centers_ = torch.zeros(size=[batch_size, self.MAX_NUM_CENTERS, 2], dtype=torch.float, device=xym_s.device, requires_grad=False)
                else:
                    centers_ = centers

                centerdir_gt.extend([centerdir_gt_matrix, centerdir_gt_present, centers_, extended_instances_num, extended_instances])

            if centers is not None:
                assert centerdir_gt[2].shape[1] == centers.shape[1], "ERROR: Different number of MAX_NUM_CENTERS detected!! Use the same value in all settings"

            # centerdir losses
            for b in range(batch_size):
                update_centerdir_vals = not centerdir_gt[1][b]

                if update_centerdir_vals:

                    gt_center_ignore = torch.ones_like(instances[b:b+1], dtype=torch.uint8, device=xym_s.device)
                    # we need separate buffer for gt_center_mask if center_ignore_px is zero since we will not
                    # have any GT values otherwise
                    if self.center_ignore_px <= 0:
                        gt_center_mask = torch.ones_like(instances[b:b + 1], dtype=torch.uint8, device=xym_s.device)
                    else:
                        # just reuse gt_center_ignore
                        gt_center_mask = gt_center_ignore

                    gt_centers = torch.zeros(size=(self.MAX_NUM_CENTERS, 2), dtype=torch.float, device=xym_s.device)

                    instance_ids_b = instances[b].unique()

                    gt_center_x, gt_center_y = None, None

                    # version that assigns closest distance to each pixel and computes results for the whole image at once
                    if self.ignore_instance_mask_and_use_closest_center:
                        # requires list of centers first
                        assert centers is not None

                        # list of all centers valid for this batch
                        valid_centers = (centers[b,:,0] > 0) | (centers[b,:,1] > 0)
                        gt_centers[valid_centers,:] = (centers[b][valid_centers][:,[1,0]]).float()

                        # skip if no centers
                        if valid_centers.sum() > 0:
                            assigned_center_ids = CenterDirGroundtruth.find_closest_center(centers[b], instances[b], xym_s)

                            # per-pixel center locations
                            gt_center_x = centers[b,assigned_center_ids[:], 1].unsqueeze(0)
                            gt_center_y = centers[b,assigned_center_ids[:], 0].unsqueeze(0)

                        # all pixels are considered as instance mask since we may not have valid instance mask at all
                        instance_mask = torch.ones_like(instances[b].unsqueeze(0), dtype=torch.bool)
                    else:
                        # assign centers based on instance map (if there are any centers)

                        if len(instance_ids_b) > 1:
                            gt_center_x = torch.ones_like(instances[b].unsqueeze(0), dtype=torch.float) * -10000
                            gt_center_y = torch.ones_like(instances[b].unsqueeze(0), dtype=torch.float) * -10000

                            for id in instance_ids_b:
                                if id <= 0: continue
                                if id > len(centers[b]):
                                    print("ERROR: GOT OUT OF BOUNDS INDEX:  %d for img: " % id, sample_name[sample_name_idx[b].item()])
                                    sys.stdout.flush()
                                    continue

                                in_mask = instances[b].eq(id).unsqueeze(0)

                                if centers is None:
                                    # calculate center of attraction
                                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                                else:
                                    center = centers[b,id.item()-1,:]

                                gt_centers[id.item()] = center.squeeze()[[1, 0]]

                                gt_center_x[in_mask] = center[1].float()
                                gt_center_y[in_mask] = center[0].float()

                            instance_mask = instances[b].unsqueeze(0) > 0


                    # do nothing if no centers
                    if gt_center_x is not None and gt_center_y is not None:

                        gt_X = gt_center_x - xym_s[1].unsqueeze(0)
                        gt_Y = gt_center_y - xym_s[0].unsqueeze(0)

                        if self.center_ignore_px <= 0:
                            # just set gt_center_mask with default distance vals but not gt_center_ignore
                            if not self.skip_gt_center_mask_generate:
                                gt_center_mask *= ~((gt_X.abs() < 3) * (gt_Y.abs() < 3))
                        else:
                            gt_center_ignore *= ~((gt_X.abs() < self.center_ignore_px) * (gt_Y.abs() < self.center_ignore_px))
                            gt_center_mask = gt_center_ignore

                        gt_R = torch.sqrt(torch.pow(gt_X, 2) + torch.pow(gt_Y, 2)) * instance_mask.float()
                        gt_theta = torch.atan2(gt_Y, gt_X)
                        gt_sin_th = torch.sin(gt_theta) * instance_mask.float()
                        gt_cos_th = torch.cos(gt_theta) * instance_mask.float()

                        # normalize groundtruth vector to 1 for all instance pixels
                        gt_M = torch.sqrt(torch.pow(gt_sin_th, 2) + torch.pow(gt_cos_th, 2))
                        gt_sin_th[instance_mask] = gt_sin_th[instance_mask] / gt_M[instance_mask]
                        gt_cos_th[instance_mask] = gt_cos_th[instance_mask] / gt_M[instance_mask]

                        centerdir_gt[0][b, 0] = gt_R
                        centerdir_gt[0][b, 1] = gt_theta
                        centerdir_gt[0][b, 2] = gt_sin_th
                        centerdir_gt[0][b, 3] = gt_cos_th

                        if orientation is not None:
                            # locations where original orientation is read
                            id_x = gt_center_x[instance_mask].long()
                            id_y = gt_center_y[instance_mask].long()

                            ROT_DIMS = orientation.shape[1]
                            
                            # orientation may hold confidence score at the end as well
                            has_confidence_score = ROT_DIMS == 2 or ROT_DIMS == 4
                            if has_confidence_score:
                                ROT_DIMS -= 1

                            assert ROT_DIMS > 0 and ROT_DIMS < 4

                            angle = orientation[b,0:ROT_DIMS, id_x, id_y]

                            # locations where orientations are stored
                            out_id = torch.nonzero(instance_mask[0])
                            centerdir_gt[0][b, 6:6+ROT_DIMS, 0, out_id[:,0], out_id[:,1]] = torch.sin(angle)
                            centerdir_gt[0][b, 9:9+ROT_DIMS, 0, out_id[:,0], out_id[:,1]] = torch.cos(angle)

                            if has_confidence_score:
                                centerdir_gt[0][b, 12:13, 0, out_id[:,0], out_id[:,1]] = orientation[b, -1, id_x, id_y]


                    if update_centerdir_vals:
                        if gt_center_mask.all() or self.skip_gt_center_mask_generate:
                            gt_center_mask = torch.zeros_like(gt_center_mask)
                        else:
                            gt_center_mask = self.gaussian_blur(1 - gt_center_mask.unsqueeze(0).float())[0]
                            gt_center_mask /= gt_center_mask.max()

                        centerdir_gt[0][b, 4] = gt_center_ignore
                        centerdir_gt[0][b, 5] = gt_center_mask

                        centerdir_gt[2][b] = gt_centers


        return centerdir_gt

    @staticmethod
    def parse_groundtruth(centerdir_gt, ignore_mask=None, return_orientation=False):
        if return_orientation:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_orientation_sin', 'gt_orientation_cos',
                    'gt_centers', 'gt_center_ignore', 'gt_center_mask', 'gt_extended_instances']
        else:
            keys = ['gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th',  'gt_centers', 'gt_center_ignore',
                    'gt_center_mask', 'gt_extended_instances']

        return CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys)

    @staticmethod
    def parse_groundtruth_map(centerdir_gt, keys=None):
        gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0][:, 0], centerdir_gt[0][:, 1], centerdir_gt[0][:, 2], centerdir_gt[0][:,3]
        gt_orientation_sin, gt_orientation_cos = centerdir_gt[0][:, 6:9], centerdir_gt[0][:, 9:12]
        gt_confidence_score = centerdir_gt[0][:, 12:13]

        gt_centers, gt_center_ignore, gt_center_mask = centerdir_gt[2], centerdir_gt[0][:, 4], centerdir_gt[0][:, 5]

        gt_extended_instances = centerdir_gt[4][:]

        vars = gt_R, gt_theta, gt_sin_th, gt_cos_th, gt_orientation_sin, gt_orientation_cos, gt_centers, gt_center_ignore, gt_center_mask, gt_extended_instances, gt_confidence_score
        names = 'gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_orientation_sin', 'gt_orientation_cos', 'gt_centers', 'gt_center_ignore', 'gt_center_mask', 'gt_extended_instances', 'gt_confidence_score'

        # combine into dictionary
        gt_maps = dict(zip(names,vars))

        if keys is None:
            return gt_maps
        else:
            return [gt_maps[k] for k in keys] if len(keys) > 1 else gt_maps[keys[0]]

    @staticmethod
    def parse_single_batch_groundtruth_map(centerdir_gt, keys=None):

        gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0], centerdir_gt[1], centerdir_gt[2], centerdir_gt[3]
        gt_orientation_sin, gt_orientation_cos = centerdir_gt[6:9], centerdir_gt[9:12]
        gt_confidence_score = centerdir_gt[12:13]

        gt_center_ignore, gt_center_mask = centerdir_gt[4], centerdir_gt[5]

        vars = gt_R, gt_theta, gt_sin_th, gt_cos_th, gt_orientation_sin, gt_orientation_cos, gt_center_ignore, gt_center_mask, gt_confidence_score
        names = 'gt_R', 'gt_theta', 'gt_sin_th', 'gt_cos_th', 'gt_orientation_sin', 'gt_orientation_cos', 'gt_center_ignore', 'gt_center_mask', 'gt_confidence_score'

        # combine into dictionary
        gt_maps = dict(zip(names,vars))

        if keys is None:
            return gt_maps
        else:
            return [gt_maps[k] for k in keys] if len(keys) > 1 else gt_maps[keys[0]]


    @staticmethod
    def get_groundtruth_instance_polygon(centerdir_gt):
        return centerdir_gt[5] if len(centerdir_gt) > 5 else None

    @staticmethod
    def convert_gt_centers_to_dictionary(gt_centers, instances, ignore=None):
        gt_centers_dict = []
        for b in range(len(gt_centers)):
            valid_idx = torch.nonzero(torch.logical_and(gt_centers[b, :, 0] > 0, gt_centers[b, :, 1] > 0)).squeeze()
            present_idx = torch.unique(instances[b])

            valid_idx = valid_idx.cpu().numpy().reshape((-1,))
            present_idx = present_idx.cpu().numpy()
            # extract centers from valid idx that are also present in instances
            center_dict = {id: gt_centers[b, id, :2].cpu().numpy()
                                for id in set(valid_idx).intersection(present_idx)}

            # ignore centers that fall within ignore region
            if ignore is not None:
                center_dict = {k: c for k, c in center_dict.items() if ignore[b, 0][instances[b] == k].min() == 0}

            gt_centers_dict.append(center_dict)

        return gt_centers_dict
    
    def _init_datastructure(self, h, w):
        extended_instances_num = 0
        extended_instances = torch.zeros([h, w], dtype=torch.int16, requires_grad=False)

        centers = torch.zeros(size=[self.MAX_NUM_CENTERS, 2], dtype=torch.float, requires_grad=False)

        centerdir_gt_matrix = torch.zeros(size=[13, 1, h, w], dtype=torch.float, requires_grad=False)
        centerdir_gt_present = 0

        output = None

        return (centerdir_gt_matrix, centerdir_gt_present, centers, extended_instances_num, extended_instances), output
    @staticmethod
    def find_closest_center(centers, instances, xym_s):
        assert len(instances.shape) == 2

        height, width = instances.shape

        X, Y = xym_s[1], xym_s[0]

        # function used to calc closest distance
        def _calc_closest_center_patch_i(_center_x,_center_y, _X, _Y):
            # distance in cartesian space to center
            distances_to_center = torch.sqrt((_X[...,None] - _center_x) ** 2 + (_Y[..., None] - _center_y) ** 2)

            closest_center_index = torch.argmin(distances_to_center, dim=-1)

            return closest_center_index.long()

        # select patch size that is dividable but still as large as possible (from ranges of 16 to 128 - from 2**4 to 2**7)
        patch_size_options = [(2**i)*(2**j) for j in range(4,8) for i in range(4,8)]
        patch_size_options = [s for s in patch_size_options if height*width % s == 0]
        patch_size = max(patch_size_options)
        patch_count = (height*width) // patch_size

        # reshape all needed matrices into new shape
        (_X, _Y) = [x.reshape(patch_count,-1)  for x in [X, Y]]

        # main section to calc closest dist by splitting it
        valid_centers = (centers[:, 0] > 0) | (centers[:, 1] > 0) # use only valid centers and then remap indexes

        center_y = centers[valid_centers, 0]
        center_x = centers[valid_centers, 1]

        closest_center_index = torch.zeros((patch_count, patch_size), dtype=torch.long, device=instances.device)
        for i in range(patch_count):
            closest_center_index[i, :] = _calc_closest_center_patch_i(center_x, center_y, _X[i], _Y[i])

        # re-map indexes from list of selected/valid center to list of all centers
        valid_centers_idx = torch.nonzero(valid_centers)
        closest_center_index = valid_centers_idx[closest_center_index]

        return closest_center_index.reshape(instances.shape)

