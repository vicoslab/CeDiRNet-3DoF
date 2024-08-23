import glob, os, json, cv2
from matplotlib import pyplot as plt
import numpy as np
import pyransac3d as pyrsc
import matplotlib.pyplot as plt

from numpy.linalg import inv

from utils.utils_depth import *

fx = 1081.3720703125
cx = 959.5
cy = 539.5

K = np.array([[fx, 0.0, cx], [0, fx, cy], [0,0,1]])

def calculate_pitch(img_fn, depth_fn, downsample_factor=0.1):
	global K
	pitch = 0

	K_ = K.copy()

	img = cv2.imread(img_fn).astype(float)/255
	depth = np.load(depth_fn)

	# preprocess depth
	depth[np.isnan(depth)]=0
	depth[np.isinf(depth)]=0
	depth[depth>1e6]=0
	depth*=1e-3

	img = cv2.resize(img, None, fx=downsample_factor, fy=downsample_factor)
	depth = cv2.resize(depth, None, fx=downsample_factor, fy=downsample_factor, interpolation=cv2.INTER_NEAREST)

	h, w, _ = img.shape

	# build point cloud
	xx, yy = np.meshgrid(range(w), range(h))
	xx = np.ravel(xx)
	yy = np.ravel(yy)
	dd = np.ravel(depth)
	pc = np.vstack((xx,yy,dd))

	# un-project points
	K_*=downsample_factor
	K_[-1,-1]=1
	pc = inv(K_)@pc

	# estimate plane
	plane1 = pyrsc.Plane()
	best_eq, best_inliers = plane1.fit(pc.T, 0.01)

	# extract pitch
	ca = best_eq[2]/np.sqrt(best_eq[0]**2+best_eq[1]**2+best_eq[2]**2)
	pitch = np.degrees(np.arccos(ca))
	pitch = 180-pitch if pitch>90 else pitch

	display = False
	if display:

		R = eul2rot((np.radians(pitch),0,0))
		pc = K_@R@pc
		d_rotated = pc[-1,:].reshape(h,w)

		plt.clf()
		plt.subplot(2,2,1)
		plt.imshow(img)
		plt.subplot(2,2,2)
		plt.imshow(depth)
		plt.subplot(2,2,3)
		plt.imshow(d_rotated)
		plt.subplot(2,2,4)
		plt.imshow(np.abs(depth-d_rotated))
		plt.draw(); plt.pause(0.01)
		plt.waitforbuttonpress()

	return pitch

def main():

	dataset_path = '/storage/datasets/ClothDataset/ClothDatasetVICOS/'
	dataset_setups = glob.glob(f'{dataset_path}bg=*')

	for setup in dataset_setups:

		cloths = glob.glob(f'{setup}/cloth=*')

		for subset in cloths:
			print(subset)

			images = glob.glob(f'{subset}/rgb/*')
			print(len(images))

			data = {}

			for fn in images:
				name = fn.split('/')[-1]

				depth_fn = f'{subset}/depth/{name[:-4]}.npy'

				if not os.path.exists(depth_fn):
					depth_fn = depth_fn.replace('camera0', 'camera1')

				pitch = calculate_pitch(fn, depth_fn)
				print("pitch", pitch)

				data[name]={'pitch': pitch}

			with open(f'{subset}/plane_angles.json', 'w', encoding='utf-8') as f:
				json.dump(data, f, ensure_ascii=False, indent=4)

def check():

	dataset_path = '/storage/datasets/ClothDataset/ClothDatasetVICOS/'
	dataset_setups = glob.glob(f'{dataset_path}bg=*')

	for setup in dataset_setups:

		cloths = glob.glob(f'{setup}/cloth=*')

		for subset in cloths:
			print(subset)

			images = glob.glob(f'{subset}/rgb/*')

			with open(f'{subset}/plane_angles.json') as f:
				data = json.load(f)

			for fn in images:
				name = fn.split('/')[-1]

				pitch = data[name]['pitch']

				depth_fn = f'{subset}/depth/{name[:-4]}.npy'

				if not os.path.exists(depth_fn):
					depth_fn = depth_fn.replace('camera0', 'camera1')

				# display result
				img = cv2.imread(fn).astype(float)/255
				depth = np.load(depth_fn)

				# preprocess depth				
				depth[np.isnan(depth)]=0
				depth[np.isinf(depth)]=0
				depth[depth>1e6]=0
				depth*=1e-3

				R = eul2rot((np.radians(pitch),0,0))

				depth_rotated = rotate_depth(depth, R, K)

				plt.clf()
				plt.subplot(2,2,1)
				plt.imshow(img)
				plt.subplot(2,2,2)
				plt.imshow(depth)
				plt.subplot(2,2,3)
				plt.imshow(depth_rotated)
				plt.subplot(2,2,4)
				plt.imshow(np.abs(depth-depth_rotated))
				plt.draw(); plt.pause(0.01)
				plt.waitforbuttonpress()

				break
				
				# data.append((name, {'pitch': 20}))

				# TODO calculate pitch
				
			# break

def check_single():

	pth = '/storage/datasets/ClothDataset/ClothDatasetVICOS/bg=festive_tablecloth/cloth=linen_rag/'
	name = 'image_0000_view19_ls2_camera1'

	im_fn = f'{pth}rgb/{name}.jpg'
	im_fn = im_fn.replace('camera1', 'camera0')
	print(im_fn)
	depth_fn = f'{pth}depth/{name}.npy'
	print(depth_fn)

	img = cv2.imread(im_fn).astype(float)/255
	depth = np.load(depth_fn)

	depth[np.isnan(depth)]=0	
	depth[np.isinf(depth)]=0
	depth[depth>1e6]=0
	depth*=1e-3

	pitch = 20

	R = eul2rot((np.radians(pitch),0,0))

	depth_rotated = rotate_depth(depth, R, K)

	plt.clf()
	plt.subplot(2,2,1)
	plt.imshow(img)
	plt.subplot(2,2,2)
	plt.imshow(depth)
	plt.subplot(2,2,3)
	plt.imshow(depth_rotated)
	plt.subplot(2,2,4)
	plt.imshow(np.abs(depth-depth_rotated))
	plt.draw(); plt.pause(0.01)
	plt.waitforbuttonpress()

if __name__=='__main__':
	main()
	# check()
	# check_single()