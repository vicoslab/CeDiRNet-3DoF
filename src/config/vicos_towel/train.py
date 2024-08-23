import copy
import os

import torch
from utils import transforms as my_transforms
from torchvision.transforms import InterpolationMode

VICOS_TOWEL_DATASET_DIR = os.environ.get('VICOS_TOWEL_DATASET_DIR')

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

NUM_FIELDS = 5
SIZE = int(os.environ.get('TRAIN_SIZE', default=256*3))
USE_DEPTH = os.environ.get('USE_DEPTH', default='False').lower() == 'true'

IN_CHANNELS = 3
if USE_DEPTH:
	IN_CHANNELS = 4

args = dict(

	cuda=True,
	display=False,
	display_it=20,

	tf_logging=['loss'],
	tf_logging_iter=2,

	visualizer=dict(name='OrientationVisualizeTrain'),

	save=True,
	save_interval=10,

	# --------
	n_epochs=50,
	ablation_str="",

	save_dir=os.path.join(OUTPUT_DIR, 'vicos_towel', '{args[ablation_str]}',
						  'backbone={args[model][kwargs][backbone]}' + f'_size={SIZE}x{SIZE}',
						  'num_train_epoch={args[n_epochs]}',
						  'depth={args[model][kwargs][use_depth]}',
						  'multitask_weight={args[multitask_weighting][name]}',
						  ),


	pretrained_model_path = None,
	resume_path = None,

	pretrained_center_model_path = None,


	train_dataset = {
		'name': 'vicos_towel',
		'kwargs': {
			'normalize': False,
			'root_dir': os.path.abspath(VICOS_TOWEL_DATASET_DIR),
			'subfolders': [dict(folder='bg=white_desk', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=green_checkered', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=poster', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes']),
						   dict(folder='bg=red_tablecloth', data_subfolders=['cloth=big_towel','cloth=checkered_rag_big','cloth=checkered_rag_medium', 'cloth=linen_rag','cloth=small_towel','cloth=towel_rag','cloth=waffle_rag','cloth=waffle_rag_stripes'])
						   ],
			'fixed_bbox_size': 15,
			'resize_factor': 1,
			'use_depth': USE_DEPTH,
			'correct_depth_rotation': False,
			'use_mean_for_depth_nan': True,
			'use_normals': False,
			'transform_per_sample_rng': True, # TRUE == RA-L version with fixed RNG for each sample (which may not be random as intended but gets good results !!)
			'transform': my_transforms.get_transform([
				# for training without augmentation (same as testing)
				{
					'name': 'ToTensor',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'type': (
						torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor) + ((torch.FloatTensor, ) if USE_DEPTH else ()),
					}
				},
				{
					'name': 'Resize',
					'opts': {
						'keys': ('image', 'instance', 'label', 'ignore', 'orientation', 'mask') + (('depth',) if USE_DEPTH else ()),
						'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.NEAREST) + ((InterpolationMode.BILINEAR, ) if USE_DEPTH else ()),
						'keys_bbox': ('center',),
						'size': (SIZE, SIZE),
					}
				},
				# for training with random augmentation
				{
				    'name': 'RandomGaussianBlur',
				    'opts': {
				        'keys': ('image',),
				        'rate': 0.5, 'sigma': [0.5, 2]
				    }
				},

				{
					'name': 'ColorJitter',
					'opts': {
						'keys': ('image',), 'p': 0.5,
						'saturation': 0.3, 'hue': 0.3, 'brightness': 0.3, 'contrast':0.3
					}
				}

			]),
			'MAX_NUM_CENTERS':16*128,
		},

		'centerdir_gt_opts': dict(
			ignore_instance_mask_and_use_closest_center=True, # by default
			center_ignore_px=3,

			skip_gt_center_mask_generate=True, # gt_center_mask is not needed since we are not training localization network

			MAX_NUM_CENTERS=16*128,
		),

		'batch_size': 4,

		'hard_samples_size': 0,
		'hard_samples_selected_min_percent':0.1,
		'workers': 4,
		'shuffle': True,
	}, 

	model = dict(
		name='fpn',
		kwargs= {
			'backbone': 'tu-convnext_base',
			'use_depth': USE_DEPTH,
			'num_classes': [NUM_FIELDS, 1],
			'use_custom_fpn':True,
			'add_output_exp': False,
			'in_channels': IN_CHANNELS,	

			'fpn_args': {
				'decoder_segmentation_head_channels':64,
				'upsampling':4, # required for ConvNext architectures
				'classes_grouping': [(0, 1, 2, 5), (3, 4)],
				'depth_mean': 0.96, 'depth_std':0.075,
			},
			'init_decoder_gain': 0.1
		},
		optimizer='Adam',
		lr=1e-4,
		weight_decay=0,

	),
	center_model=dict(
		name='PolarVotingCentersMultiscale',
		kwargs=dict(
			# use vector magnitude as mask instead of regressed mask
			use_magnitude_as_mask=False,
			# thresholds for conv2d processing
			local_max_thr=0.1, mask_thr=0.01, exclude_border_px=0,
			use_dilated_nn=True,
			dilated_nn_args=dict(
				# single scale version (nn6)
				inner_ch=16,
				inner_kernel=3,
				dilations=[1, 4, 8, 12],
				freeze_learning=True,
				gradpass_relu=False,
				# version with leaky relu
				leaky_relu=False,
				# input check
				# use_polar_radii=False,
				use_centerdir_radii = False,
				use_centerdir_magnitude = False,
				use_cls_mask = False
			),
			allow_input_backprop=False,
			backprop_only_positive=False,
			augmentation=False,
			scale_r=1.0,  # 1024
			scale_r_gt=1024,  # 1
			use_log_r=True,
			use_log_r_base='10',
			enable_6dof=False,
		),
		# DISABLE TRAINING
		optimizer='Adam',
		lr=0,
		weight_decay=0,
	),
	

	# loss options
	loss_type='OrientationLoss',
	loss_opts={
        'num_vector_fields': NUM_FIELDS,
        'foreground_weight': 1,  

		'enable_centerdir_loss': True,
        'no_instance_loss': True,  # MUST be True to ignore instance mask
        'centerdir_instance_weighted': True,
        'regression_loss': 'l1',

        'use_log_r': True,
        'use_log_r_base': '10',

		'orientation_args': dict(
			enable=True,
			no_instance_loss=False,
			regression_loss='l1',
			enable_6dof=False,
			symmetries=None,
		)
},
	multitask_weighting=dict(
		name='uw',
		kwargs=dict(
			n_tasks=2
		)
	),
	loss_w={
		'w_inst': 1,
		'w_var': 1,
		'w_seed': 1,
		'w_cls': 1,
		'w_r': 1,
		'w_cos': 1,
		'w_sin': 1,
		'w_magnitude': 1,
		'w_cent': 0.1,
		'w_orientation': 1,
	},

)

args['lambda_scheduler_fn']=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9))
#args['lambda_scheduler_fn']=lambda _args: (lambda epoch: 1.0) # disabled

args['model']['lambda_scheduler_fn'] = args['lambda_scheduler_fn']
args['center_model']['lambda_scheduler_fn'] = lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9) if epoch > 1 else 0)


def get_args():
	return copy.deepcopy(args)