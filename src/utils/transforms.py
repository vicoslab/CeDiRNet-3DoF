import collections
import random
import os

import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.transforms import InterpolationMode

import torch

def get_image_size(im):
    if isinstance(im,torch.Tensor) or isinstance(im, np.ndarray):
        return im.shape[-2:][::-1]
    elif isinstance(im, Image.Image):
        return im.size
    else:
        raise Exception("Cannot get size of NON-tensor/numpy/PIL image object")

class CropRandomObject:

    def __init__(self, keys=[],keys_bbox=[],object_key="instance", size=100):
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.object_key = object_key
        self.size = size

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        
        if unique_objects.size > 0:
            random_id = rng.choice(unique_objects, 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)
            
            i = int(np.clip(ym-self.size[1]/2, 0, h-self.size[1]))
            j = int(np.clip(xm-self.size[0]/2, 0, w-self.size[0]))

        else:
            i = rng.integers(max(1,h - self.size[1]))
            j = rng.integers(max(1,w - self.size[0]))

        for k in self.keys:
            assert(k in sample)

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.crop(sample[k].unsqueeze(0), i, j, self.size[1], self.size[0])[0]
            else:
                sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

            i,j = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                i, j = i+1,j+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)

            sample[k][idx, i] -= j
            sample[k][idx, j] -= i

        return sample

class RandomCrop(T.RandomCrop):

    def __init__(self, keys=[], keys_bbox=[], keys_camK=[], size=100, pad_if_needed=False, fixed_offset=None):
        self.disabled = False

        try:
            if type(size) in [list, tuple]:
                size = [int(s) if isinstance(s, str) else s for s in size]
            else:
                size = int(size) if isinstance(size, str) else size
        except ValueError:
            # disable it if invalid input size !!
            self.disabled = True
            size = 0

        super().__init__(size, pad_if_needed=pad_if_needed)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK
        self.fixed_offset = fixed_offset        

    def _pad_if_needed(self, img):

        pad_size = [0,0,0,0] # (top,left,right,bottom)
        if self.pad_if_needed:
            img_size = get_image_size(img)
            # pad the width if needed
            if img_size[0] < self.size[1]:
                pad_w = self.size[1] - img_size[0]
                # pad both sides equally
                pad_size[0] = pad_w // 2
                pad_size[2] = pad_w-pad_size[0]

            # pad the height if needed
            if img_size[1] < self.size[0]:
                pad_h = self.size[0] - img_size[1]

                # pad both sides equally
                pad_size[1] = pad_h // 2
                pad_size[3] = pad_h - pad_size[1]

            if type(img) == torch.Tensor and len(img.shape) == 2:
                img = F.pad(img.unsqueeze(0), pad_size, self.fill, self.padding_mode)[0]
            else:
                img = F.pad(img, pad_size, self.fill, self.padding_mode)

        return img, pad_size
    @staticmethod
    def get_params(img, output_size, fixed_offset=None, rng=None):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        if rng is None:
            rng = np.random.default_rng()

        w, h = get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        if fixed_offset is not None:
            i,j = fixed_offset

        else:
            i = rng.integers(max(h - th,1))
            j = rng.integers(max(w - tw,1))


        return i, j, th, tw

    def __call__(self, sample, rng=None, *args):
        if self.disabled:
            return sample

        params = None

        for k in self.keys:
            assert(k in sample)

            if self.pad_if_needed:
                sample[k], pad_size = self._pad_if_needed(sample[k])

            if params is None:
                params = self.get_params(sample[k], self.size, self.fixed_offset, rng)

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.crop(sample[k].unsqueeze(0), *params)[0]
            else:
                sample[k] = F.crop(sample[k], *params)

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert params

            dx,dy, _,_ = params

            if self.pad_if_needed:
                dx, dy = dx - pad_size[1], dy - pad_size[0]

            i,j = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                i, j = i+1,j+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:,i] > 0) | (sample[k][:,j] > 0)
            sample[k][idx, i] -= dy
            sample[k][idx, j] -= dx
        
        for k in self.keys_camK:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert params

            dx,dy, _,_ = params

            sample[k][0, 2] -= dy
            sample[k][1, 2] -= dx

        return sample

class FixedCrop(RandomCrop):
    def __init__(self, loc=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loc = loc
    
    def get_params(self, img, output_size, fixed_offset=None, rng=None):
        i,j = self.loc
        th, tw = output_size

        return i, j, th, tw

class CenterCrop(RandomCrop):
    @staticmethod
    def get_params(img, output_size, fixed_offset=None, rng=None):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        if rng is None:
            rng = np.random.default_rng()

        w, h = get_image_size(img)
        th, tw = output_size
        
        if w == tw and h == th:
            return 0, 0, h, w

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

class RandomCustomRotation(object):

    def __init__(self, angles, rate=0.5, resample=False, expand=False, center=None, keys=[], keys_bbox=[], keys_custom_fn=dict()):

        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn
        self.angles = angles
        self.rate = rate
        self.resample = InterpolationMode.BILINEAR if resample else InterpolationMode.NEAREST
        self.expand = expand
        self.center = center

        if isinstance(self.resample, collections.Iterable):
            assert (len(keys) == len(self.resample))

    @staticmethod
    def get_params(angles, rate, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < rate:
            return rng.choice(angles, 1)[0]
        else:
            return 0

    def __call__(self, sample, rng=None, *args):

        angle = self.get_params(self.angles, self.rate, rng)

        if angle != 0:
            if len(self.keys) > 0:
                org_im_size = get_image_size(sample[self.keys[0]])

            for idx, k in enumerate(self.keys):

                assert(k in sample)

                resample = self.resample
                if isinstance(resample, collections.Iterable):
                    resample = resample[idx]

                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.rotate(sample[k].unsqueeze(0), float(angle), resample,
                                         self.expand, self.center)[0]
                else:
                    sample[k] = F.rotate(sample[k], float(angle), resample,
                                         self.expand, self.center)

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    new_im_size = get_image_size(sample[self.keys[0]])
                assert org_im_size
                assert new_im_size

                old_center = (org_im_size[0]/2,org_im_size[1]/2) if self.center is None else self.center
                new_center = (new_im_size[0]/2,new_im_size[1]/2) if self.expand else old_center

                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k],torch.Tensor) or isinstance(sample[k],np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    org_shape = sample[k][idx, i:].shape
                    points = sample[k][idx, i:].reshape(-1,2)

                    # position relative to the old center
                    points[:, 0] -= old_center[0]
                    points[:, 1] -= old_center[1]

                    # rotate each point
                    angle_rad = angle*np.pi/180
                    rot_mat = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                                        [np.sin(-angle_rad), np.cos(-angle_rad)]], dtype=np.float32)

                    points = np.matmul(rot_mat, points.T).T

                    # position relative to the new center
                    points[:, 0] += new_center[0]
                    points[:, 1] += new_center[1]

                    sample[k][idx, i:] = points.reshape(org_shape)
            
            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k], angle)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '(angles={0}'.format(self.angles)
        format_string += ', rate={0}'.format(self.rate)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string



class RandomRotation(T.RandomRotation):

    def __init__(self, keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys

        if isinstance(self.resample, collections.Iterable):
            assert(len(keys) == len(self.resample))

    @staticmethod
    def get_params(degrees, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        angle = rng.uniform(degrees[0], degrees[1])

        return angle
    def __call__(self, sample, rng=None, *args):

        angle = self.get_params(self.degrees, rng)

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            resample = self.resample
            if isinstance(resample, collections.Iterable):
                resample = resample[idx]

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.rotate(sample[k].unsqueeze(0), angle, resample, self.expand, self.center)[0]
            else:
                sample[k] = F.rotate(sample[k], angle, resample, self.expand, self.center)

        return sample


class RandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, keys=[], keys_bbox=[], keys_custom_fn=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for k in self.keys:
                assert (k in sample)
                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.hflip(sample[k].unsqueeze(0))[0]
                else:
                    sample[k] = F.hflip(sample[k])

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    im_size = get_image_size(sample[self.keys[0]])
                assert im_size
                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    sample[k][idx, i] = im_size[0] - sample[k][idx, i]
                    if sample[k].shape[1] > i+2:
                        sample[k][idx, i] -= sample[k][idx, i+2]
            
            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k])
                
        return sample

class RandomVerticalFlip(T.RandomVerticalFlip):

    def __init__(self, keys=[], keys_bbox=[], keys_custom_fn=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_custom_fn = keys_custom_fn

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for k in self.keys:
                assert (k in sample)
                if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                    sample[k] = F.vflip(sample[k].unsqueeze(0))[0]
                else:
                    sample[k] = F.vflip(sample[k])

            if len(self.keys_bbox) > 0:
                if len(self.keys) > 0:
                    im_size = get_image_size(sample[self.keys[0]])
                assert im_size
                for k in self.keys_bbox:
                    assert (k in sample)
                    assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                    i, j = 0, 1
                    # ignore first element if odd number of elements (we assume it is ID)
                    if sample[k].shape[1] % 2 == 1:
                        i, j = i + 1, j + 1
                        idx = sample[k][:, 0] > 0
                    else:
                        idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                    sample[k][idx, j] = im_size[1] - sample[k][idx, j]
                    if sample[k].shape[1] > j+2:
                        sample[k][idx, j] -= sample[k][idx, j+2]
            
            if self.keys_custom_fn is not None:
                for k,fn in self.keys_custom_fn.items():
                    sample[k] = fn(sample[k])
        return sample

class RandomResize(object):

    def __init__(self, scale_range, keys=[], keys_bbox=[], keys_camK=[], interpolation=Image.BILINEAR, *args, **kwargs):
        self.scale_range = scale_range
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK        
        self.interpolation = interpolation

        if isinstance(self.interpolation, collections.Iterable):
            assert(len(keys) == len(self.interpolation))

    @staticmethod
    def get_params(factor_range, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        resize_factor = rng.uniform(factor_range[0], factor_range[1])

        return resize_factor

    def __call__(self, sample, rng=None, *args):

        resize_factor = self.get_params(self.scale_range, rng)

        if len(self.keys) > 0:
            org_im_size = get_image_size(sample[self.keys[0]])

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            new_size = int(get_image_size(sample[k])[1]*resize_factor), int(get_image_size(sample[k])[0]*resize_factor)
            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.resize(sample[k].unsqueeze(0), new_size, interpolation, antialias=None)[0]
            else:
                sample[k] = F.resize(sample[k], new_size, interpolation, antialias=None)

        if len(self.keys_bbox) > 0:
            if len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])
            assert org_im_size
            assert new_im_size
            real_resize_factor = new_im_size[0] / org_im_size[0], new_im_size[1] / org_im_size[1]

            for k in self.keys_bbox:
                assert (k in sample)
                assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                i, j = 0, 1
                # ignore first element if odd number of elements (we assume it is ID)
                if sample[k].shape[1] % 2 == 1:
                    i, j = i + 1, j + 1
                    idx = sample[k][:, 0] > 0
                else:
                    idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                sample[k][idx, i] *= real_resize_factor[0]
                sample[k][idx, j] *= real_resize_factor[1]
                if sample[k].shape[1] > j+2:
                    sample[k][idx, i+2] *= real_resize_factor[0]
                    sample[k][idx, j+2] *= real_resize_factor[1]
        
        # adjustment of camera calibration matrix 
        if len(self.keys_camK) > 0:
            if len(self.keys) > 0:
                new_im_size = get_image_size(sample[self.keys[0]])
            assert org_im_size
            assert new_im_size
            real_resize_factor = new_im_size[0] / org_im_size[0], new_im_size[1] / org_im_size[1]

            for k in self.keys_camK:
                if k not in sample:
                    continue
                
                sample[k][0,:] *= real_resize_factor[0]
                sample[k][1,:] *= real_resize_factor[1]


        return sample



class Resize(T.Resize):

    def __init__(self, keys=[], keys_bbox=[], keys_camK=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.keys = keys
        self.keys_bbox = keys_bbox
        self.keys_camK = keys_camK

        if isinstance(self.interpolation, collections.Iterable):
            assert(len(keys) == len(self.interpolation))

    def __call__(self, sample, *args):

        if len(self.keys) > 0:
            org_im_size = get_image_size(sample[self.keys[0]])

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            interpolation = self.interpolation
            if isinstance(interpolation, collections.Iterable):
                interpolation = interpolation[idx]

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.resize(sample[k].unsqueeze(0), self.size, interpolation, antialias=None)[0]
            else:
                sample[k] = F.resize(sample[k], self.size, interpolation, antialias=None)

        if len(self.keys_bbox) > 0:
            assert org_im_size
            resize_factor = self.size[1]/org_im_size[0], self.size[0]/org_im_size[1]

            for k in self.keys_bbox:
                assert (k in sample)
                assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)

                i, j = 0, 1
                # ignore first element if odd number of elements (we assume it is ID)
                if sample[k].shape[1] % 2 == 1:
                    i, j = i + 1, j + 1
                    idx = sample[k][:, 0] > 0
                else:
                    idx = (sample[k][:, i] != 0) * (sample[k][:, j] != 0)
                sample[k][idx, i] *= resize_factor[0]
                sample[k][idx, j] *= resize_factor[1]
                if sample[k].shape[1] > j+2:
                    sample[k][idx, i+2] *= resize_factor[0]
                    sample[k][idx, j+2] *= resize_factor[1]

        # adjustment of camera calibration matrix 
        if len(self.keys_camK) > 0:
            
            resize_factor = self.size[1]/org_im_size[0], self.size[0]/org_im_size[1]

            for k in self.keys_camK:
                if k not in sample:
                    continue
                
                sample[k][0,:] *= resize_factor[0]
                sample[k][1,:] *= resize_factor[1]



        return sample


class ColorJitter(T.ColorJitter):

    def __init__(self, keys=[], p=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.p = p

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        fn_idx = rng.permutation(4)

        b = None if brightness is None else float(rng.uniform(brightness[0], brightness[1], size=1))
        c = None if contrast is None else float(rng.uniform(contrast[0], contrast[1], size=1))
        s = None if saturation is None else float(rng.uniform(saturation[0], saturation[1], size=1))
        h = None if hue is None else float(rng.uniform(hue[0], hue[1], size=1))

        return fn_idx, b, c, s, h

    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            for idx, k in enumerate(self.keys):

                assert(k in sample)

                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
                    self.get_params(self.brightness, self.contrast, self.saturation, self.hue, rng)

                for fn_id in fn_idx:
                    if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                        sample[k] = sample[k].unsqueeze(0)

                    if fn_id == 0 and brightness_factor is not None:
                        sample[k] = F.adjust_brightness(sample[k], brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        sample[k] = F.adjust_contrast(sample[k], contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        sample[k] = F.adjust_saturation(sample[k], saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        sample[k] = F.adjust_hue(sample[k], hue_factor)

                    if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                        sample[k] = sample[k][0]
        return sample


class RandomGaussianNoise(object):

    def __init__(self, keys=[], p=0, mean=0., std=1., **kwargs):
        self.keys = keys
        self.p = p
        self.mean = mean
        self.std = std


    def __call__(self, sample, rng=None, *args):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < self.p:
            noise = None
            for idx, k in enumerate(self.keys):
                assert(k in sample)
                
                if noise is  None:
                    mean = rng.uniform(self.mean[0], self.mean[1]) if type(self.mean) in [list,tuple] else self.mean
                    std = rng.uniform(self.std[0], self.std[1]) if type(self.std) in [list,tuple] else self.std
                    
                    noise = np.random.normal(mean, std, sample[k].shape).astype(np.float32)

                sample[k] = sample[k] + noise

        return sample

class Padding(object):

    IGNORE_PADDING_EDGE = 64
    IGNORE_PADDING_SYMMETRIC = 128

    def __init__(self, keys=[], keys_bbox=[], borders=(0,0,0,0), pad_to_size_factor=0, fill=0, padding_mode="constant", mark_key_with_flag=None):

        self.borders = borders # (left,top,right,bottom)
        self.pad_to_size_factor = pad_to_size_factor
        self.fill = fill
        self.padding_mode = padding_mode
        self.keys = keys
        self.keys_bbox = keys_bbox
        self.mark_with_flag_key = mark_key_with_flag
        self.mark_with_flag = False
        if mark_key_with_flag:
            if padding_mode == "edge":
                self.mark_with_flag = self.IGNORE_PADDING_EDGE
            elif padding_mode == "symmetric":
                self.mark_with_flag = self.IGNORE_PADDING_SYMMETRIC

    def get_pad_size(self, img):

        pad_size = list(self.borders)

        if self.pad_to_size_factor > 0:
            img_size = get_image_size(img)
            # pad the width if needed
            pad_w = (img_size[0] + pad_size[0] + pad_size[2]) % self.pad_to_size_factor
            if pad_w > 0:
                pad_w = self.pad_to_size_factor - pad_w
                # pad both sides equally
                pad_size[0] += pad_w // 2
                pad_size[2] += pad_w - (pad_w // 2)

            # pad the height if needed
            pad_h = (img_size[1] + pad_size[1] + pad_size[3]) % self.pad_to_size_factor
            if pad_h > 0:
                pad_h = self.pad_to_size_factor - pad_h
                # pad both sides equally
                pad_size[1] += pad_h // 2
                pad_size[3] += pad_h - pad_h // 2

        return pad_size

    def __call__(self, sample, rng=None, *args):
        pad_size = None

        for k in self.keys:
            assert(k in sample)

            pad_size_ = self.get_pad_size(sample[k])

            if pad_size is None:
                pad_size = pad_size_
                assert pad_size_ == pad_size

            if type(sample[k]) == torch.Tensor and len(sample[k].shape) == 2:
                sample[k] = F.pad(sample[k].unsqueeze(0), pad_size, self.fill, self.padding_mode)[0]
            else:
                sample[k] = F.pad(sample[k], pad_size, self.fill, self.padding_mode)

            if self.mark_with_flag and k == self.mark_with_flag_key:
                l, t, r, b = pad_size
                # ignore = np.array(sample[k])
                # borders = np.zeros_like(ignore)
                # borders[t:borders.shape[0]-b,l:borders.shape[1]-r] = 1
                #
                # ignore |= (1 - borders) * self.mark_with_flag
                # sample[k] = Image.fromarray(ignore)
                # convert to Tensor first
                ignore = torch.tensor(np.array(sample[k])) if not isinstance(sample[k], torch.Tensor) else sample[k]
                borders = torch.zeros_like(sample[k] if isinstance(sample[k], torch.Tensor) else torch.from_numpy(np.array(sample[k])))
                borders[...,t:borders.shape[-2]-b,l:borders.shape[-1]-r] = 1

                ignore |= (1 - borders) * self.mark_with_flag

                # convert back to original type
                if isinstance(sample[k], np.ndarray):
                    ignore = ignore.numpy()
                elif isinstance(sample[k], Image.Image):
                    ignore = Image.fromarray(ignore.numpy())

                sample[k] = ignore

        for k in self.keys_bbox:
            assert (k in sample)
            assert isinstance(sample[k], torch.Tensor) or isinstance(sample[k], np.ndarray)
            assert pad_size

            i,j = 0,1
            # ignore first element if odd number of elements (we assume it is ID)
            if sample[k].shape[1] % 2 == 1:
                i, j = i+1,j+1
                idx = sample[k][:, 0] > 0
            else:
                idx = (sample[k][:,i] > 0) | (sample[k][:,j] > 0)
            sample[k][idx, i] += pad_size[0]
            sample[k][idx, j] += pad_size[1]

        return sample

class ToTensor(object):

    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(type))

        self.keys = keys
        self.type = type

    def __call__(self, sample, *args):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            t = self.type
            if isinstance(t, collections.Iterable):
                t = t[idx]

            if isinstance(sample[k],Image.Image) or isinstance(sample[k],np.ndarray):
                sample[k] = F.to_tensor(sample[k])

                if t == torch.ByteTensor:
                    sample[k] = sample[k]*255

            elif not isinstance(sample[k],torch.Tensor):
                raise Exception("Invalid data format: must be PIL Image, numpy array or pytorch Tensor")

            sample[k] = sample[k].type(t)

        return sample

from scipy.ndimage.filters import gaussian_filter

class RandomGaussianBlur(object):

    def __init__(self, rate, sigma, keys=[]):

        self.rate = rate
        self.sigma_interval = sigma
        self.keys = keys

    @staticmethod
    def get_params(rate, sigma, rng=None):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < rate:
            return rng.random()*(sigma[1]-sigma[0]) + sigma[0]
        else:
            return 0

    def __call__(self, sample, rng=None, *args):

        sigma = self.get_params(self.rate, self.sigma_interval, rng)

        if sigma > 0:
            for idx, k in enumerate(self.keys):
                assert (k in sample)
                #sample[k] = gaussian_filter(sample[k], sigma)
                sample[k] = F.gaussian_blur(sample[k], 2*round(3*sigma) + 1, sigma)

        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ', rate={0}'.format(self.rate)
        format_string += ', sigma_interval=[{0},{1]]'.format(self.sigma_interval[0],self.sigma_interval[1])
        format_string += ')'
        return format_string


class Normalize(object):

    def __init__(self, keys=[], mean=[], std=[], shape=[]):

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(mean))

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(std))

        if isinstance(type, collections.Iterable):
            assert(len(keys) == len(shape))


        self.keys = keys
        self.mean = mean
        self.std = std
        self.shape = shape

    def __call__(self, sample, *args):

        for idx, k in enumerate(self.keys):

            assert(k in sample)

            m = self.mean
            if isinstance(m, collections.Iterable):
                m = m[idx]

            s = self.std
            if isinstance(s, collections.Iterable):
                s = s[idx]

            sh = self.shape
            if isinstance(sh, collections.Iterable):
                sh = sh[idx]

            sample[k] = (sample[k]  - m.reshape(sh) ) / s.reshape(sh)

        return sample

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, *args):
        for t in self.transforms:
            img = t(img, *args)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Background(object):

    def __init__(self, bg_dir, bg_images, keys=('image',), p=0.5, mask_key='label', img_ext='.jpg'):
        self.keys = keys
        self.mask_key = mask_key
        self.images = []
        with open(bg_images) as f:
            lines = f.readlines()
            for l in lines:
                s = l.strip().split(' ')                
                ok = (int(s[-1])==1) if len(s) > 1 else True
                if ok:
                    fn = os.path.join(bg_dir,f"{s[0]}{img_ext}")
                    self.images.append(fn)

        self.p = p

    def __call__(self, sample, rng=None, *args):

        augment = np.random.random()<self.p

        if not augment:
            return sample

        bg_fn = np.random.choice(self.images, 1)[0]

        label = sample[self.mask_key]
        if isinstance(label,torch.Tensor):
            label = label.permute((1, 2, 0)).numpy()

        bg = None

        for key in self.keys:
            to_output_format = lambda x: x
            im = sample[key]
            if isinstance(im,torch.Tensor):
                to_output_format = F.to_tensor
                if im.type() == torch.ByteTensor:
                    im = im/255
                    to_output_format = lambda x: (F.to_tensor(x)*255).type(torch.ByteTensor)

                im = (im*255).permute([1, 2, 0]).numpy()
                
            elif isinstance(im,Image.Image):
                to_output_format = Image.fromarray

            im = np.array(im)

            im_shape = im.shape[:2][::-1]
    
            # get bg
            if bg is None:
                bg = cv2.imread(bg_fn)
                bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                bg = cv2.resize(bg, im_shape)
                label = np.dstack((label,label,label))
                field3d_mask = np.broadcast_to(label > 0, bg.shape)

            final = bg.copy()
            final[field3d_mask]=im[field3d_mask]
            sample[key] = to_output_format(final)

        return sample

def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr['name']
        opts = tr['opts']
        transform_list.append(globals()[name](**opts))

    return Compose(transform_list)
