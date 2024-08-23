import glob
import os, cv2, sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import json
import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.utils_depth import get_normals


def angle_to_rad(A):
	A = (A + 360) if A < 0 else A
	return np.deg2rad(A - 180)

class MuJoCoDataset(Dataset):

	def __init__(self, root_dir='./', subfolder="", MAX_NUM_CENTERS=1024, transform=None, use_depth=False, segment_cloth=False, use_normals=False, fixed_bbox_size=15, num_cpu_threads=1, normals_mode=1, reference_normal = [0,0,1], **kwargs):
		print('MuJoCoDataset created')

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.root_dir = root_dir
		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS
		self.transform = transform
		self.use_depth = use_depth
		self.use_normals = use_normals
		self.fixed_bbox_size = fixed_bbox_size
		self.normals_mode = normals_mode
		self.reference_normal = reference_normal
		self.segment_cloth = segment_cloth

		if type(subfolder) not in [list, tuple]:
			subfolder = [subfolder]

		image_list = []
		for sub in subfolder:
			image_list += sorted(glob.glob(f"{self.root_dir}/{sub}/rgb/*"))

		#image_list = sorted(glob.glob(f"{self.root_dir}/rgb/*"))

		self.image_list = image_list
		print(f'MuJoCoDataset of size {len(image_list)}')

		self.size = len(self.image_list)

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		im_fn = self.image_list[index]

		fn = os.path.splitext(os.path.split(im_fn)[-1])[0]	

		image = Image.open(im_fn)
		im_size = image.size

		root_dir = os.path.abspath(os.path.join(os.path.dirname(im_fn),'..'))
		depth_fn = os.path.join(root_dir, 'depth', f'{fn}.npy')

		gt_fn = os.path.join(root_dir, 'gt_points_vectors', f'{fn}.npy')
		if os.path.exists(gt_fn):
			gt_data = np.load(gt_fn)
		else:
			print(gt_fn, "not found")
			gt_data = []

		sample = dict(
			image=image,
			im_name=im_fn,
			im_size=im_size,
			index=index,
		)
		
		if self.segment_cloth:
			gt_seg_fn = os.path.join(root_dir, "gt_cloth", f"{fn}.png")

			segmentation_mask = Image.open(gt_seg_fn)

			if self.resize_factor is not None and self.resize_factor != 1.0:
				segmentation_mask = segmentation_mask.resize(im_size, Image.BILINEAR)

			sample["segmentation_mask"] = segmentation_mask


		if self.use_depth:
			depth = np.load(depth_fn)

			if self.use_normals:
				depth = get_normals(depth, normals_mode=self.normals_mode, household=False)
			else:
				depth/=np.max(depth)

			sample['depth']=depth

		# create instances image
		instances = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.int16)
		orientation = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.float32)
		label = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)

		centers = []

		m = self.fixed_bbox_size
		for n, (i,j,s,c) in enumerate(gt_data):
			i = int(i)
			j = int(j)
			centers.append((i,j))

			angle = np.arctan2(s,c)
			angle = np.degrees(angle)

			instances[0, j-m:j+m,i-m:i+m] = n+1
			label[0, j-m:j+m,i-m:i+m] = 1
			orientation[0, j-m:j+m,i-m:i+m]=angle_to_rad(angle)


		sample['orientation'] = orientation
		sample['instance'] = instances
		sample['label'] = label
		sample['mask'] = (label > 0)
		sample['ignore'] = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)

		centers = np.array(centers)
		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		try:
			sample['center'][:centers.shape[0], :] = centers
		except:
			print("no objects in image")

		if self.transform is not None:
			rng = np.random.default_rng(seed=1234)
			sample = self.transform(sample, rng)

		if self.use_depth:
			sample['image'] = torch.cat((sample['image'], sample['depth']))
		

		return sample
	


if __name__ == "__main__":
	import pylab as plt
	import matplotlib

	matplotlib.use('TkAgg')
	from tqdm import tqdm
	import torch

	USE_DEPTH = True
	from utils import transforms as my_transforms

	transform = my_transforms.get_transform([
		{
			'name': 'ToTensor',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
				'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor,
						 torch.ByteTensor)+ ((torch.FloatTensor, ) if USE_DEPTH else ()),
			},			
		}
	])
	subfolders = ['mujoco', 'mujoco_all_combinations_normal_color_temp', 'mujoco_all_combinations_rgb_light', 'mujoco_white_desk_HS_extreme_color_temp', 'mujoco_white_desk_HS_normal_color_temp']

	db = RTFMDataset(root_dir='/storage/datasets/ClothDataset/', resize_factor=1, transform_only_valid_centers=1.0, transform=transform, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolder=subfolders)
	shapes = []
	for item in tqdm(db):
		if item['index'] % 50 == 0:
			print('loaded index %d' % item['index'])
		shapes.append(item['image'].shape)
		# if True or np.array(item['ignore']).sum() > 0:
		# if True:
		if item['index'] % 1 == 0:
			center = item['center']
			gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
			# print(gt_centers)
			plt.clf()

			im = item['image'].permute([1, 2, 0]).numpy()
			# print(im.shape)

			plt.subplot(2, 2, 1)
			plt.imshow(im[...,:3])
			plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

			x = gt_centers[:,0]
			y = gt_centers[:,1]

			r = 100

			for i,j in zip(x,y):
				i = int(i)
				j = int(j)
				if i < 0 or i > item['orientation'].shape[2] or \
					j < 0 or j > item['orientation'].shape[1]:
					continue
				angle = item['orientation'][0][j,i].numpy()
				# print(angle)
				# s = item['orientation'][1][j,i]
				s = -np.sin(angle)
				c = -np.cos(angle)
				# print(i,j,c,s)
				plt.plot([i,i+r*s],[j,j+r*c], 'r-')



			plt.draw(); plt.pause(0.01)
			plt.waitforbuttonpress()
			# plt.show()

	print("end")
