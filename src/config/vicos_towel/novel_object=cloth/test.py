import copy
import os

import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
	from torchvision.transforms import InterpolationMode
else:
	from PIL import Image as InterpolationMode

import torch
from utils import transforms as my_transforms

VICOS_TOWEL_DATASET_DIR = os.environ.get('VICOS_TOWEL_DATASET_DIR')

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

NUM_FIELDS = 5
TRAIN_SIZE = os.environ.get('TRAIN_SIZE', default=256*3)
TEST_SIZE_WIDTH = int(os.environ.get('TEST_SIZE', default=256*3))
TEST_SIZE_HEIGHT = int(os.environ.get('TEST_SIZE', default=256*3))
USE_DEPTH = os.environ.get('USE_DEPTH', default='False').lower() == 'true'

IN_CHANNELS = 3

if USE_DEPTH:
	IN_CHANNELS = 4

def img2tags_fn(x):
	# from x=/storage/datasets/ClothDataset/bg=<ozadje>/cloth=<krpa>/rgb/image_0000_viewXX_lsYY_camera0.jpg
	# extract bg, cloth, view (configuration and clutter) and ls (lightning

	# extract bg
	bg = x.split('/')[-4]
	# extract cloth
	cloth = x.split('/')[-3]
	# extract ls
	lightning = int(x.split('/')[-1].split('_')[3].replace('ls', ''))
	# extract view
	view = int(x.split('/')[-1].split('_')[2].replace('view', ''))
	# configuration and clutter are encoded in view, clutter is on if view is odd
	clutter = 'on' if view % 2 == 1 else 'off'
	configuration = view // 2

	return [bg, cloth, f'lightning={lightning}', f'configuration={configuration}', f'clutter={clutter}']



model_dir = os.path.join(OUTPUT_DIR, 'vicos_towel', '{args[ablation_str]}',
						  'backbone={args[model][kwargs][backbone]}' + f'_size={TRAIN_SIZE}x{TRAIN_SIZE}',
						  'num_train_epoch={args[train_settings][n_epochs]}',
						  'depth={args[model][kwargs][use_depth]}',
						  'multitask_weight={args[train_settings][multitask_weighting][name]}')

args = dict(

	cuda=True,
	display=True,
	autoadjust_figure_size=True,

	groundtruth_loading = True,

	save=True,

	save_dir=os.path.join(model_dir,'{args[dataset][kwargs][type]}_results{args[eval_epoch]}',f'test_size={TEST_SIZE_HEIGHT}x{TEST_SIZE_WIDTH}',),
	checkpoint_path=os.path.join(model_dir,'checkpoint{args[eval_epoch]}.pth'),

	eval_epoch='',
	ablation_str='',

	eval=dict(
		# available score types ['mask', 'center', 'hough_energy', 'edge_to_area_ratio_of_mask', 'avg(mask_pix)', 'avg(hough_pix)', 'avg(projected_dist_pix)']
		score_combination_and_thr=[
			{
			'center': [0.1,0.01,0.05,0.15,0.2,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.7,0.75,0.8,0.85,0.9,0.94,0.99],
			},
		],
		score_thr_final=[0.01],
		skip_center_eval=True,
		orientation=dict(
			display_best_threshold=False,
			tau_thr=[20], 
		),
		enable_multivariate_eval=dict(
			image2tags_fn=img2tags_fn,
		)
	),
	visualizer=dict(name='OrientationVisualizeTest',
					opts=dict(show_rot_axis=(True,),
							  impath2name_fn=lambda x: ".".join(x.split('/')[-4:]).replace('.jpg',''))),


	dataset={
		'name': 'vicos_towel',
		'kwargs': {
			'normalize': False,
			'root_dir': os.path.abspath(VICOS_TOWEL_DATASET_DIR),

			'type': 'test_novel_object=cloth',
			'subfolders': [dict(folder='bg=red_tablecloth', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
						   dict(folder='bg=white_desk', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
						   dict(folder='bg=green_checkered', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
						   dict(folder='bg=poster', data_subfolders=['cloth=checkered_rag_small', 'cloth=cotton_napkin']),
						   ],

			'fixed_bbox_size': 5,
			'resize_factor': 1,
			'use_depth': USE_DEPTH,
			'use_mean_for_depth_nan': True,
			'transform': my_transforms.get_transform([
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'type': (
						torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor,
						torch.ByteTensor) + ((torch.FloatTensor, ) if USE_DEPTH else ()),
					}
				},
				{
					'name': 'Resize',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST) + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
						'keys_bbox': ('center',),
						'size': (TEST_SIZE_HEIGHT, TEST_SIZE_WIDTH),
					}
				},

			]),
			'MAX_NUM_CENTERS':16*128,
		},
		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True,
			center_ignore_px=3,

			MAX_NUM_CENTERS=16*128,
		),

		'batch_size': 1,
		'workers': 0,
	},

	model=dict(
		name='fpn',
		kwargs={
			'backbone': 'tu-convnext_base',
			'use_depth': USE_DEPTH,
			'num_classes': [NUM_FIELDS, 1],
			'use_custom_fpn': True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,
			'fpn_args': {
				'decoder_segmentation_head_channels': 64,
				'upsampling':4, # required for ConvNext architectures
				'classes_grouping': [(0, 1, 2, 5), (3, 4)],
				'depth_mean': 0.96, 'depth_std': 0.075,
			},
			'init_decoder_gain': 0.1
		},
	),
	center_model=dict(
		name='CenterOrientationEstimator',
		use_learnable_center_estimation=True,

		kwargs=dict(
			use_centerdir_radii = False,
			
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=True,
			# thresholds for conv2d processing
			local_max_thr=0.01, local_max_thr_use_abs=True,
			
			### dilated neural net as head for center detection
			use_dilated_nn=True,
			dilated_nn_args=dict(
				return_sigmoid=False,
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				use_centerdir_radii=False,
				use_centerdir_magnitude=False,
				use_cls_mask=False
				),
			augmentation=False,
			scale_r=1.0,  # 1024
			scale_r_gt=1,  # 1
			use_log_r=False,
			use_log_r_base='10',
			enable_6dof=False,
		),

	),
	num_vector_fields=NUM_FIELDS,

	# settings from train config needed for automated path construction
	train_settings=dict(
		n_epochs=10,
		multitask_weighting=dict(name='uw'),
	)
)

def get_args():
	return copy.deepcopy(args)
