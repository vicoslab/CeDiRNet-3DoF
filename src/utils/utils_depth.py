import numpy as np, cv2, glob, os
from matplotlib import pyplot as plt
from numpy.linalg import norm, inv

def get_surface_normal_by_depth(depth, K=None):
	"""
	depth: (h, w) of float, the unit of depth is meter
	K: (3, 3) of float, the depth camere's intrinsic
	"""
	K = [[1, 0], [0, 1]] if K is None else K
	fx, fy = K[0][0], K[1][1]

	dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
	du_dx = fx / depth  # x is xyz of camera coordinate
	dv_dy = fy / depth

	dz_dx = dz_du * du_dx
	dz_dy = dz_dv * dv_dy
	normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
	normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
	normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
	return normal_unit

def get_angle_from_depth(depth, household=False, k=15):
	# https://math.stackexchange.com/questions/3433645/how-can-i-find-the-angle-of-the-surface-3d-plane

	depth = cv2.GaussianBlur(depth,(k,k),0)

	# get normals from depth
	if household: # for kinect images
		fx = 1081.3720703125
		cx = 959.5
		cy = 539.5
	else:
		fx = 256
		cx = 256.0
		cy = 256.0

	K = np.array([[fx,0,cx],[0,fx,cy],[0,0,1]])
	normals = get_surface_normal_by_depth(depth, K=K)

	return normals

def eul2rot(theta) :

	R = np.array(
		[
			[np.cos(theta[1])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
			[np.sin(theta[2])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
			[-np.sin(theta[1]),np.sin(theta[0])*np.cos(theta[1]),np.cos(theta[0])*np.cos(theta[1])]
		]
  	)

	return R

def get_normals(depth, normals_mode=1, household=False, k=15):

	normals = get_angle_from_depth(depth, household=household, k=k)

	# 1 is normal vector as 3 channels
	# 2 is angle between normal vector and each of the axes
	# 3 is Sobel operator x and y

	if normals_mode==1:
		normals = get_angle_from_depth(depth, household=household, k=k)
	elif normals_mode==2:
		dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)     
		dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

		dx[np.isnan(dx)]=0
		dy[np.isnan(dy)]=0
		dx[np.isinf(dx)]=0
		dy[np.isinf(dy)]=0

		dx/=np.max(dx)
		dy/=np.max(dy)
		normals = np.dstack((dx,dy))

	return normals

	normals = get_angle_from_depth(depth, household=household, k=k)

	# 1 is normal vector as 3 channels
	# 2 is dot product of normal vector with reference normal vector in 1 channel
	# 3 is angle between normal plane and reference vector in 1 channel
	# 4 is same as 3, except expressed as sin and cos

	n = np.zeros_like(normals)
	n[...,0]=reference_normal[0]
	n[...,1]=reference_normal[1]
	n[...,2]=reference_normal[2]

	if normals_mode==1:
		depth = normals.copy()
	elif normals_mode==2:
		depth = np.sum(normals*n, axis=-1)
	elif normals_mode==3:
		depth = np.sum(normals*n, axis=-1)

		# print("dot 0", np.unique((normals*n)[...,0]))
		# print("dot 1", np.unique((normals*n)[...,1]))
		# print("dot 2", np.unique((normals*n)[...,2]))

		depth = np.arccos(depth)
		# TODO normalize
	elif normals_mode==4:
		# print("dot 0", np.unique((normals*n)[...,0]))
		# print("dot 1", np.unique((normals*n)[...,1]))
		# print("dot 2", np.unique((normals*n)[...,2]))
		depth = np.sum(normals*n, axis=-1)
		angle = np.arccos(depth)
		depth = np.dstack((np.sin(angle), np.cos(angle)))
		# print("depth", depth.shape)
	elif normals_mode==5:
		# depth = normals.copy()

		# x1 = 220
		# x2 = 1051
		# y1 = 400
		# y2 = 1530

		# mask = np.zeros_like(depth)
		# mask[313:1534, 286:1051]=1
		# mask[x1:x2, y1:y2]=1
		# mask = mask.astype(bool)
		# depth[~mask]=0


		dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=k)     
		dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=k)

		dx[np.isnan(dx)]=0
		dy[np.isnan(dy)]=0
		dx[np.isinf(dx)]=0
		dy[np.isinf(dy)]=0

		dx/=np.max(dx)
		dy/=np.max(dy)

		# print("dx", dx.shape, dx.dtype)
		# print("dy", dy.shape, dy.dtype)
		# print("min max dx", np.min(dx), np.max(dx))
		# print("min max dy", np.min(dy), np.max(dy))

		# dx[x1-k:x1+k, :] = 0
		# dx[x2-k:x2+k, :] = 0
		# dx[:, y1-k:y1+k] = 0
		# dx[:, y2-k:y2+k] = 0

		# dy[x1-k:x1+k, :] = 0
		# dy[x2-k:x2+k, :] = 0
		# dy[:, y1-k:y1+k] = 0
		# dy[:, y2-k:y2+k] = 0
		

		depth = np.dstack((dx,dy))

	return depth

def rotate_depth(depth, R, K):

	h, w = depth.shape

	xx, yy = np.meshgrid(range(w), range(h))
	xx = np.ravel(xx)
	yy = np.ravel(yy)
	dd = np.ravel(depth)

	pc = np.vstack((xx,yy,dd))
	pc = inv(K)@pc
	pc = K@R@pc

	d_rotated = pc[-1,:].reshape(h,w)

	return d_rotated