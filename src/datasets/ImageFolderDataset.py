import glob
import os

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch
from torchvision.transforms import functional as F
from utils.transforms import Padding, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderDataset(Dataset):

    def __init__(self, root_dir='./', pattern='*.jpg', depth_dir=None, resize_factor=None, use_depth=False):

        print('ImageFolderDataset created')

        self.use_depth = use_depth
        self.resize_factor = resize_factor

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, pattern))
        image_list.sort()

        self.image_list = image_list
        self.real_size = len(self.image_list)
        self.depth_dir = depth_dir

        print('found %d images' % self.real_size)

        if self.use_depth:
            self.pad = Padding(keys=['image','depth',] , pad_to_size_factor=32)
            self.to_tensor = ToTensor(keys=['image','depth',] , type=(torch.FloatTensor,torch.FloatTensor,))
        else:
            self.pad = Padding(keys=['image',], pad_to_size_factor=32)
            self.to_tensor = ToTensor(keys=['image',], type=(torch.FloatTensor,))

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        # this will load only image info but not the whole data
        image = Image.open(self.image_list[index])
        im_size = image.size

        if self.resize_factor is not None:
            im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

        if self.resize_factor is not None and self.resize_factor != 1.0:
            image = image.resize(im_size, Image.BILINEAR)

        sample = dict(image=image,
                      im_name=self.image_list[index],
                      im_size=im_size,
                      index=index)

        if self.use_depth:            
            
            if self.depth_dir:                
                depth_fn = os.path.join(self.depth_dir, os.path.splitext(os.path.basename(self.image_list[index]))[0] + ".npy")
            else:
                depth_fn = os.path.splitext(self.image_list[index])[0] + ".npy"

            assert os.path.exists(depth_fn), f"Depth file '{depth_fn}' missing"
            depth = np.load(depth_fn)
            
            if self.resize_factor is not None and self.resize_factor != 1.0:
                import cv2
                depth = cv2.resize(depth,im_size)
                
            invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
            depth[invalid_mask]=depth[~invalid_mask].mean() 

            depth*=1e-3 # mm to m

            # correct depth values so the surface is parallel to the image plane        
            depth/=np.max(depth)

            sample['depth'] = depth
        
        sample = self.to_tensor(sample)
        sample = self.pad(sample)

        sample['image_raw'] = sample['image']

        if self.use_depth:            
            sample['image'] = torch.cat((sample['image'], sample['depth']))

        return sample