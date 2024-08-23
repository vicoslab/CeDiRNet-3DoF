import glob
import os, cv2

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import json, sys, os

import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.utils_depth import get_normals, eul2rot, rotate_depth, get_normals
from utils import transforms as my_transforms

class ViCoSTowelDataset(Dataset):
	IGNORE_LABEL = "ignore-region"
    
	IGNORE_FLAG = 1
	IGNORE_TRUNCATED_FLAG = 2
	IGNORE_OVERLAP_BORDER_FLAG = 4
	IGNORE_DIFFICULT_FLAG = 8
    
	def __init__(self, root_dir='./', subfolders=None, fixed_bbox_size=15, resize_factor=None, use_depth=False, segment_cloth=False, MAX_NUM_CENTERS=1024, 
			  	transform=None, valid_img_names=None, num_cpu_threads=1, use_mean_for_depth_nan=False, 
				use_normals=False, normals_mode=1, use_only_depth=False, correct_depth_rotation=False, 
				check_consistency=True, transform_only_valid_centers=False, transform_per_sample_rng=False, **kwargs):
		print('ViCoS TowelDataset created')

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.use_depth = use_depth
		self.use_mean_for_depth_nan = use_mean_for_depth_nan
		self.use_normals = use_normals
		self.normals_mode = normals_mode
		self.use_only_depth = use_only_depth
		self.segment_cloth = segment_cloth

		self.correct_depth_rotation = correct_depth_rotation

		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor

		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

		self.transform = my_transforms.get_transform(transform) if type(transform) == list else transform
		self.rng = np.random.default_rng(1337)
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng

		self.return_image = True
		self.check_consistency = check_consistency

		# calibration parameters
		fx = 1081.3720703125
		cx = 959.5
		cy = 539.5

		self.K = np.array([[fx, 0.0, cx], [0, fx, cy], [0,0,1]])

		# find images

		image_list = []
		annot = {}

		if subfolders is None:
			subfolders = ["*/*"]
		if type(subfolders) not in [list, tuple]:
			subfolders = [subfolders]

		# expecting annotations in root if available
		annotations_list = {root_dir: 'annotations.json'}

		for sub in subfolders:
			if type(sub) is dict:
				sub_name = sub['folder']								
				data_paths = sub['data_subfolders']

				for data_path in data_paths:
					image_list += sorted(glob.glob(os.path.join(root_dir, sub_name, data_path, 'rgb', '*')))

				# if subfolder has its own anotations use them
				annotations_list[os.path.join(root_dir,sub_name)] = 'annotations.json'

			else:
				# print(os.path.join(root_dir,sub,'rgb','*'))
				image_list += sorted(glob.glob(os.path.join(root_dir,sub,'rgb','*')))

		# read and merge annotations
		annotations = {}
		for sub,annot_filename in annotations_list.items():
			annot_path = os.path.join(sub,annot_filename)

			if not os.path.exists(annot_path):
				continue

			with open(os.path.join(sub,annot_filename)) as f:
				annot = json.load(f)
			for k,v in annot.items():
				annotations[os.path.abspath(os.path.join(sub,k))] = v

		if valid_img_names is not None:		
			def filter_by_name(x):				
				return any([v in x for v in valid_img_names])
			
			image_list = list(filter(filter_by_name, image_list))
		

		self.image_list = image_list
		self.annotations = annotations

		self.size = len(self.image_list)
		print(f'ViCoS TowelDataset of size {len(image_list)}')        

		self.remove_out_of_bounds_centers = True
		self.BORDER_MARGIN_FOR_CENTER = 0

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		im_fn = self.image_list[index]

		root_dir = os.path.abspath(os.path.join(os.path.dirname(im_fn),'..'))
		fn = os.path.splitext(os.path.split(im_fn)[-1])[0]
		ann_key = os.path.abspath(im_fn)


		image = Image.open(im_fn)
		im_size = image.size
		org_im_size = np.array(image.size)

		if self.resize_factor is not None:
			im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

		# avoid loading full buffer data if image not requested
		if self.return_image:
			if self.resize_factor is not None and self.resize_factor != 1.0:
				image = image.resize(im_size, Image.BILINEAR)
		else:
			image = None

		sample = dict(image=image,
					  im_name=im_fn,
					  org_im_size=org_im_size,
					  im_size=im_size,
					  index=index)
		
		if self.segment_cloth:
			segmentation_mask_file = os.path.join(root_dir, "mask", f"{fn}.png")
			segmentation_mask = Image.open(segmentation_mask_file)
			
			if self.resize_factor is not None and self.resize_factor != 1.0:
				segmentation_mask = segmentation_mask.resize(im_size, Image.BILINEAR)

			sample["segmentation_mask"] = segmentation_mask

		if self.use_depth:
			depth_fn = os.path.join(root_dir, 'depth', f'{fn}.npy')
			# print(depth_fn, os.path.exists(depth_fn))
			depth = np.load(depth_fn)
			
			if self.resize_factor is not None and self.resize_factor != 1.0:
				depth = cv2.resize(depth,im_size)
				
			invalid_mask = np.isinf(depth) | np.isnan(depth) | (depth > 1e4) | (depth<0)
			depth[invalid_mask]=depth[~invalid_mask].mean() if self.use_mean_for_depth_nan else 1e-6

			depth*=1e-3 # mm to m

			# correct depth values so the surface is parallel to the image plane
			if self.correct_depth_rotation:
				surface_pitch = self.annotations.get(ann_key)['surface_pitch']
				#print("surface_pitch", surface_pitch)
				R = eul2rot((np.radians(surface_pitch), 0,0))
				depth = rotate_depth(depth, R, self.K)

			if self.use_normals:
				depth = get_normals(depth, normals_mode=self.normals_mode, household=True)
			else:
				depth/=np.max(depth)

			sample['depth'] = depth

		label = torch.zeros((im_size[1], im_size[0]), dtype=torch.uint8)
		orientation = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.float32)
		instances = torch.zeros((im_size[1], im_size[0]), dtype=torch.int16)
		centers = []

		annot = self.annotations.get(ann_key)['points']

		if annot:
			M = self.fixed_bbox_size
			instance_counter = 1

			for x1,y1,x2,y2 in annot:

				pt1 = np.array([x1, y1])
				pt2 = np.array([x2, y2])

				if self.resize_factor is not None and self.resize_factor != 1.0:
					pt1 = pt1 * self.resize_factor
					pt2 = pt2 * self.resize_factor

				direction = pt1 - pt2
				direction = np.arctan2(direction[0], direction[1])

				orientation[:, int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = direction
				label[int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = 1
				instances[int(pt1[1] - M):int(pt1[1] + M), int(pt1[0] - M):int(pt1[0] + M)] = instance_counter
				centers.append(pt1)

				instance_counter += 1

			centers = np.array(centers)

		sample['center'] = np.zeros((self.MAX_NUM_CENTERS, 2))
		if len(centers) > 0:
			sample['center'][:centers.shape[0], :] = centers
		sample['label'] = label.unsqueeze(0)
		sample['mask'] = (label > 0).unsqueeze(0)
		sample['ignore'] = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)
		sample['orientation'] = orientation
		sample['instance'] = instances.unsqueeze(0)
		sample['name'] = im_fn

		if self.transform is not None:
			import copy
			do_transform = True

			ii = 0
			while do_transform:			
				if ii > 10 and ii % 10 == 0:
					print(f"WARNING: unable to generate valid transform for {ii} iterations")
				new_sample = self.transform(copy.deepcopy(sample), self.rng if not self.transform_per_sample_rng else np.random.default_rng(1337))

				out_of_bounds_ids = [id for id, c in enumerate(new_sample['center']) if c[0] < 0 or c[1] < 0 or c[0] >= new_sample['image'].shape[-1] or c[1] >= new_sample['image'].shape[-2]]

				# stop if sufficent centers still visible
				if not self.transform_only_valid_centers or self.transform_only_valid_centers <= 0:
					do_transform = False
					sample = new_sample
				else:
					if type(self.transform_only_valid_centers) == bool:
						if len(out_of_bounds_ids) < len(centers): # at least one must be present
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == int:
						if len(centers) - len(out_of_bounds_ids) >= self.transform_only_valid_centers:
							do_transform = False
							sample = new_sample
					elif type(self.transform_only_valid_centers) == float:
						min_visible = int(self.transform_only_valid_centers * len(centers))
						if len(centers) - len(out_of_bounds_ids) >= min_visible:
							do_transform = False
							sample = new_sample
					else:
						raise Exception("Invalid type of transform_only_valid_centers, allowed types: bool, int, float")				
				ii += 1
				


		if self.use_depth:
			if self.use_only_depth:
				sample['image'] = torch.cat((sample['image']*0, sample['depth']))
			else:
				sample['image'] = torch.cat((sample['image'], sample['depth']))                     


		if self.remove_out_of_bounds_centers:
			# if instance has out-of-bounds center then ignore it if requested so
			out_of_bounds_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id >= 0 and (c[0] < 0 or c[1] < 0 or
											c[0] >= sample['image'].shape[-1] or
											c[1] >= sample['image'].shape[-2])]
			for id in out_of_bounds_ids:
				sample['instance'][sample['instance'] == id+1] = 0
				sample['orientation'][sample['instance'] == id+1] = 0
				sample['center'][id,:] = -1
   
		if self.transform is not None and 'instance' in sample:
			# recheck for ignore regions due to changes from augmentation:
			#  - mark with ignore any centers/instances that are now outside of image size
			valid_ids = np.unique(sample['instance'])
			truncated_ids = [id for id, c in enumerate(sample['center'])
							 # if center closer to border then this margin than mark it as truncated
							 if id >= 0 and id in valid_ids and (c[0] < self.BORDER_MARGIN_FOR_CENTER or
																c[1] < self.BORDER_MARGIN_FOR_CENTER or
																c[0] >= sample['image'].shape[-1]-self.BORDER_MARGIN_FOR_CENTER or
																c[1] >= sample['image'].shape[-2]-self.BORDER_MARGIN_FOR_CENTER)]
			for id in truncated_ids:
				sample['ignore'][sample['instance'] == id+1] |= self.IGNORE_TRUNCATED_FLAG


		if self.check_consistency:
			if np.any(np.unique(instances) != torch.unique(sample['instance']).numpy()):
				print(f"error in augmentation - missing one or more sample in instance mask after augmentation (before was {np.unique(instances)} now is {torch.unique(sample['instance'])})")

		return sample

if __name__ == "__main__":
	import pylab as plt
	import matplotlib

	matplotlib.use('TkAgg')
	from tqdm import tqdm
	import torch

	from torchvision.transforms import InterpolationMode
	import functools

	def rotate_orientation_values(orientation, angle):
		return (orientation + (angle*np.pi / 180.0) + np.pi)  % (2 * np.pi) - np.pi

	def hflip_orientation_values(orientation):
		return (-1*orientation + np.pi)  % (2 * np.pi) - np.pi

	def vflip_orientation_values(orientation):
		return (np.pi - orientation + np.pi)  % (2 * np.pi) - np.pi

	SIZE_HEIGHT = 544
	SIZE_WIDTH = 960

	USE_DEPTH = True

	transform = my_transforms.get_transform([
		# for training without augmentation (same as testing)
		{
			'name': 'ToTensor',
			'opts': {
				# 'keys': ('image', 'instance', 'label', 'ignore'),
				# 'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
				'type': (
				torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor) + ((torch.FloatTensor, ) if USE_DEPTH else ()),
			}
		},
		{
			'name': 'Resize',
			'opts': {
				# 'keys': ('image', 'instance', 'label', 'difficult'),
				# 'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST),
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
				'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST) + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
				'keys_bbox': ('center',),
				'size': (SIZE_HEIGHT, SIZE_WIDTH),
			}
		},
		{
			'name': 'RandomHorizontalFlip',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
				'keys_custom_fn' : { 'orientation': lambda x: (-1*x + np.pi)  % (2 * np.pi) - np.pi},
				'p': 0.5,
			}
		},
		{
			'name': 'RandomVerticalFlip',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
				'keys_custom_fn' : { 'orientation': lambda x: (np.pi - x + np.pi)  % (2 * np.pi) - np.pi},
				'p': 0.5,
			}
		},
		{
			'name': 'RandomCustomRotation',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()), 'keys_bbox': ('center',),
				'keys_custom_fn' : { 'orientation': lambda x,angle: (x + (angle*np.pi / 180.0) + np.pi)  % (2 * np.pi) - np.pi},
				'resample': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,
									InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST)  + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
				'angles': list(range(0,360,10)),
				'rate':0.5,
			}
		},
		{
			'name': 'ColorJitter',
			'opts': {
				'keys': ('image',), 'p': 0.5,
				'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
			}
		},
	])
	NOT_transform = [
		{
			'name': 'ToTensor',
			'opts': {
				'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
				'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor,
						 torch.ByteTensor)+ ((torch.FloatTensor, ) if USE_DEPTH else ()),
			},			
		}
	]

	subfolders = [dict(folder='bg=white_desk', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=green_checkered', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=poster', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
	   dict(folder='bg=festive_tablecloth', annotation='annotations_propagated.xml', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
	   ]

	normals_mode = 2
	use_normals = False

	db = ViCoSTowelDataset(root_dir='/storage/datasets/ClothDataset/ClothDatasetVICOS/', resize_factor=0.5, transform_only_valid_centers=1.0, transform=transform, use_depth=USE_DEPTH, correct_depth_rotation=False, subfolders=subfolders)
	for item in db:
		if item['index'] % 50 == 0:
			print('loaded index %d' % item['index'])
		
		if item['index'] % 1 == 0:
			center = item['center']
			gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]
			# print(gt_centers)
			plt.clf()

			im = item['image'].permute([1, 2, 0]).numpy()
			print(item['name'])
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

			if use_normals:
				if normals_mode==1 or normals_mode==2:
					d = item['depth'].permute(1,2,0).numpy()

					plt.subplot(2, 2, 2)				
					plt.imshow(d[...,0])
					plt.subplot(2, 2, 3)
					plt.imshow(d[...,1])
					plt.subplot(2, 2, 4)
					plt.imshow(d[...,2])

				if normals_mode==3:
					plt.subplot(2, 2, 2)
					d = item['depth'].permute(1,2,0).numpy()
					plt.imshow(d[...,0])
					plt.subplot(2, 2, 3)
					plt.imshow(d[...,1])
			else:
				d = item['depth'].permute(1,2,0).numpy()
				plt.subplot(2, 2, 2)
				plt.imshow(d)

			# 	# n1 = get_normals(d, normals_mode=1)
			# 	# n2 = get_normals(d, normals_mode=2)
			# 	# n1 = np.arccos(n1)

			# 	# print(np.abs(n1-n2))

			# 	plt.subplot(2,2,3)
			# 	n1 = get_normals(d, normals_mode=1)
			# 	n2 = get_normals(d, normals_mode=2)
			# 	plt.imshow(np.abs(n1-n2))

			# 	plt.subplot(2,2,4)
			# 	n1 = np.arccos(n1)

			# 	plt.imshow(np.abs(n1-n2))


			# for normals==5
			# d = item['depth'].permute(1,2,0).numpy()[...,0]
			# print(d.shape)
			# plt.imshow(d)
			# plt.subplot(2, 2, 3)
			# d2 = item['depth'].permute(1,2,0).numpy()[...,1]
			# print(d2.shape)
			# plt.imshow(d2)

			# except:
			# 	pass

			# plt.subplot(2, 2, 2)
			# plt.imshow(item['instance'][0])
			# plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

			# plt.subplot(2, 2, 3)
			# plt.imshow(item['orientation'][0])
			# plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

			# if sample['depth']:



			# plt.show(block=False)
			plt.draw(); plt.pause(0.01)
			plt.waitforbuttonpress()
			# plt.show()

	print("end")
