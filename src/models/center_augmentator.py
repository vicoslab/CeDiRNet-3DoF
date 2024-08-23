import sys
import torch
import torch.nn as nn

import numpy as np
from utils.utils import GaussianLayer
from torchvision.transforms import functional as F
class CenterAugmentator(nn.Module):
    def __init__(self, occlusion_probability=0, occlusion_type='circle', occlusion_distance_type='random',
                 occlusion_center_jitter_probability=0, occlusion_center_jitter_relative_size=0.5, occlusion_center_jitter_px=0,
                 gaussian_noise_probability=0, gaussian_noise_blur_sigma=3, gaussian_noise_std_polar=[0.5,3], gaussian_noise_std_mask=[1,5],
                 gaussian_occlusion_probability=0, gaussian_occlusion_sigma=5.0, gaussian_occlusion_region_prob_thr=0.05):
        super().__init__()

        self.occlusion_probability = occlusion_probability
        self.occlusion_type = occlusion_type
        self.occlusion_distance_type = occlusion_distance_type
        self.occlusion_center_jitter_probability = occlusion_center_jitter_probability
        self.occlusion_center_jitter_relative_size = occlusion_center_jitter_relative_size
        self.occlusion_center_jitter_px = occlusion_center_jitter_px

        self.gaussian_noise_probability = gaussian_noise_probability
        self.gaussian_noise_std_polar = gaussian_noise_std_polar
        self.gaussian_noise_std_mask = gaussian_noise_std_mask
        self.gaussian_noise_blur = [GaussianLayer(num_channels=4, sigma=gaussian_noise_blur_sigma)]

        self.gaussian_occlusion_probability = gaussian_occlusion_probability
        self.gaussian_occlusion_sigma = gaussian_occlusion_sigma
        self.gaussian_occlusion_region_prob_thr = gaussian_occlusion_region_prob_thr

        xym = self._create_xym(0)

        self.register_buffer("xym", xym, persistent=False)

    def _create_xym(self, size):
        Y, X = torch.meshgrid(torch.arange(size), torch.arange(size))

        return torch.stack((X, Y), 0).float()

    def _get_xym(self, height, width):
        max_size = max(height, width)
        if max_size > min(self.xym.shape[1], self.xym.shape[2]):
            self.xym = self._create_xym(max_size).to(self.xym.device)

        return self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

    def _apply_center_augmentation(self, input, gt_instance, gt_centerdir, gt_seed):

        xym_s = self._get_xym(input.shape[-2], input.shape[-1])
        X,Y = xym_s[0], xym_s[1]

        if gt_seed is None:
            gt_seed = [None] * len(gt_instance)

        #gt_instance = gt_instance.to(input.device)
        gt_centers = gt_centerdir[2]

        for b in range(len(gt_instance)):
            # save random seed if not present
            if gt_seed[b] is None:
                gt_seed[b] = np.random.randint(sys.maxsize)

            # set random seed for this sample
            rng = np.random.default_rng(gt_seed[b])

            for i in range(len(gt_instance[b].unique()) - 1):
                if rng.random() < self.occlusion_probability:
                    instance_mask = gt_instance[b, 0] == i + 1
                    instance_nonzero = instance_mask.nonzero().cpu()
                    instance_bbox = np.array([instance_nonzero.min(dim=0)[0].numpy(),
                                              instance_nonzero.max(dim=0)[0].numpy()])
                    instance_wh = [instance_bbox[1][0] - instance_bbox[0][0],
                                   instance_bbox[1][1] - instance_bbox[0][1]]
                    instance_center = [instance_bbox[0][0] + instance_wh[0] / 2,
                                       instance_bbox[0][1] + instance_wh[1] / 2]

                    if rng.random() < self.occlusion_center_jitter_probability:
                        if self.occlusion_center_jitter_px > 0:
                            jitter_px = self.occlusion_center_jitter_px
                        elif self.occlusion_center_jitter_relative_size > 0:
                            jitter_px = np.sqrt(instance_wh[0]**2 + instance_wh[1]**2) * self.occlusion_center_jitter_relative_size
                        else:
                            raise Exception("Cannot add jitter to center occlusion: missing absolute or relative size")

                        RANDOM_CENTER_JITTER = rng.integers(-jitter_px, jitter_px, size=2, endpoint=False)

                        instance_center[0] += RANDOM_CENTER_JITTER[0]
                        instance_center[1] += RANDOM_CENTER_JITTER[1]

                    if self.occlusion_distance_type == 'fixed':
                        RANDOM_DIST = max(instance_wh) // 4
                    elif self.occlusion_distance_type == 'larger':
                        # range numbers to choice from
                        dist_range = list(range(5, max(max(instance_wh) // 4, 6), 2))
                        # add additional larger ranges to the list to increase thier probability
                        dist_range = dist_range + \
                                     list(range(max(instance_wh)//5, max(instance_wh) // 3,2)) + \
                                     list(range(max(instance_wh)//5, max(instance_wh) // 3,2))

                        RANDOM_DIST = rng.choice(dist_range,1)[0]
                    elif self.occlusion_distance_type == 'random':
                        RANDOM_DIST = rng.integers(5, max(6,max(instance_wh) // 4))

                    if self.occlusion_type == 'circle':
                        #gt_center = gt_centers[b][i]

                        dist = torch.sqrt((Y - instance_center[0])**2 + (X - instance_center[1])**2)

                        invalid_box = (dist < RANDOM_DIST) * (instance_mask == 1)

                        input[b][invalid_box.repeat([input.shape[1],1,1])] = 0

                    elif self.occlusion_type == 'box':
                        #occlusion_wh = [instance_wh[0] * (np.random.random() * (0.3) + 0.2),
                        #                instance_wh[1] * (np.random.random() * (0.3) + 0.2)]
                        occlusion_wh = [instance_wh[0] * (1/2),
                                       instance_wh[1] * (1/2)]
                        occlusion_bbox = [int(instance_center[0] - occlusion_wh[0] / 2),
                                          int(instance_center[1] - occlusion_wh[1] / 2),
                                          int(instance_center[0] + occlusion_wh[0] / 2),
                                          int(instance_center[1] + occlusion_wh[1] / 2)]
                        input[b, :, occlusion_bbox[0]:occlusion_bbox[2], occlusion_bbox[1]:occlusion_bbox[3]] = 0
                    else:
                        raise Exception('Unknwon augmentation_occlusion type : "%s"' % self.occlusion_type)
        return input

    def _apply_gaussian_noise(self, input, gt_seed):
        b,_,h,w = input.shape
        if gt_seed is None:
            gt_seed = [None] * input.shape[0]

        noise = torch.ones_like(input)

        std_polar, std_mask = [None]*input.shape[0], [None]*input.shape[0]

        for b in range(input.shape[0]):
            # save random seed if not present
            if gt_seed[b] is None:
                gt_seed[b] = np.random.randint(sys.maxsize)

            # set random seed for this sample
            rng = np.random.default_rng(gt_seed[b])

            if rng.random() < self.gaussian_noise_probability:

                std_polar[b] = rng.uniform(low=self.gaussian_noise_std_polar[0], high=self.gaussian_noise_std_polar[1])
                std_mask[b] = rng.uniform(low=self.gaussian_noise_std_mask[0], high=self.gaussian_noise_std_mask[1])

                #noise[b,:3] = torch.from_numpy(rng.normal(0, scale=std_polar[b], size=(3, h, w)))
                #noise[b,3:] = torch.from_numpy(rng.normal(0, scale=std_mask[b], size=(1, h, w)))
                torch.manual_seed(gt_seed[b])

                torch.normal(0*noise[b][:3], std_polar[b] * noise[b,:3], out=noise[b, :3])
                torch.normal(0*noise[b][3:], std_mask[b] * noise[b,3:], out=noise[b, 3:])
            else:
                noise[b] = 0

        self.gaussian_noise_blur[0].to(noise.device)

        noise = self.gaussian_noise_blur[0](noise)

        return input + noise

    def _apply_gaussian_noise_occlusion_mask(self, input, gt_seed):
        b,_,h,w = input.shape
        if gt_seed is None:
            gt_seed = [None] * input.shape[0]

        occlusion_mask = torch.ones(size=(b,1,h,w),dtype=torch.float32).to(input.device)

        for i in range(input.shape[0]):
            # save random seed if not present
            if gt_seed[i] is None:
                gt_seed[i] = np.random.randint(sys.maxsize)

            # set random seed for this sample
            rng = np.random.default_rng(gt_seed[i])

            if rng.random() < self.gaussian_occlusion_probability:

                torch.manual_seed(gt_seed[i])
                # create noise
                occlusion_mask[i] = torch.rand(size=occlusion_mask[i].shape)
            else:
                occlusion_mask[i] = 1

        mean, std = occlusion_mask.mean(dim=[1,2,3]), occlusion_mask.std(dim=[1,2,3])

        # add gaussian blur to enlarge occlusion pixels
        occlusion_mask = F.gaussian_blur(occlusion_mask, kernel_size=[5, 5], sigma=self.gaussian_occlusion_sigma)

        # normalize to the same mean and std as in non-blurred version
        occlusion_mask = (occlusion_mask - occlusion_mask.mean())/std + mean
        # convert to mask based on user-defined probability threshold
        occlusion_mask = occlusion_mask > self.gaussian_occlusion_region_prob_thr

        return input * occlusion_mask

    def forward(self, input, **gt):
        do_center = self.occlusion_probability > 0 and 'centerdir_groundtruth' in gt
        do_gaussian_noise = self.gaussian_noise_probability > 0
        do_gaussian_occlusion_mask = self.gaussian_occlusion_probability > 0

        with torch.no_grad():
            if do_center or do_gaussian_noise:
                input = input.clone()

            if do_center:
                input = self._apply_center_augmentation(input, gt['instance'], gt['centerdir_groundtruth'], gt.get('seed'))

            if do_gaussian_noise:
                input = self._apply_gaussian_noise(input, gt.get('seed'))

            if do_gaussian_occlusion_mask:
                input = self._apply_gaussian_noise_occlusion_mask(input, gt.get('seed'))

        return input