import os
from functools import partial

import numpy as np
import torch

import cv2

from utils.visualize.vis import Visualizer

def get_pose_axis_from_trigonometric_euler_and_depth(p, sin_eulers, cos_eulers, depth, cam_K, length = 0.1):
    cam_K = cam_K.cpu().numpy().squeeze()
        
    # convert sin/cos to eulers    
    eulers = torch.atan2(sin_eulers, cos_eulers)  # convert to angle to fix scaling
    
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz',eulers).as_matrix()
    
    j, i = p    
    
    pt = np.linalg.inv(cam_K) @ np.array([j, i, 1])    
    pt = (pt / pt[-1]) * depth[0,0,int(i),int(j)].item()
    
    points = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, R, pt, cam_K, (0, 0, 0, 0))
    
    plot_axis = []
    for axis in [0,1,2]:
        d = axisPoints[axis].ravel() 
        o = axisPoints[-1].ravel() 
        plot_axis.append([o,d])

    return np.array(plot_axis)

def get_pose_axis_from_trigonometric_param(p, sin_angle, cos_angle, R=30):

    pred_angle = torch.atan2(sin_angle, cos_angle)+np.pi  # convert to angle to fix scaling
    
    sin_angle = torch.sin(pred_angle)
    cos_angle = torch.cos(pred_angle)

    plot_axis = []
    for s,c in zip(sin_angle, cos_angle):
        plot_axis.append((p, (p[0] + c * R, p[1] + s * R)))
                         
    return np.array(plot_axis)
    
class OrientationVisualizeTest(Visualizer):
    KEYS = ['image', 'centers', 'centers-est', 'pred', 'sigma', 'seed', 'gt-diff', 'centerdir_gt']

    def __init__(self, keys=(), show_rot_axis=(True,True,True), axis_from_eulers=False, show_regressed_maps=True, show_gt_maps=True, plot_rank_number=False, bgr_axis=False, show_confidence_score=False, **kwargs):
        super(OrientationVisualizeTest, self).__init__(keys=self.KEYS + list(keys), **kwargs)

        self.show_rot_axis = show_rot_axis
        self.axis_from_eulers = axis_from_eulers
        
        self.show_regressed_maps = show_regressed_maps
        self.show_gt_maps = show_gt_maps 
        self.show_confidence_score = show_confidence_score

        self.plot_rank_number = plot_rank_number
        

        if bgr_axis:
            self.color_rot_axis = [(0, 0, 128), (0, 128, 0),(128, 0, 0)]
            self.color_rot_axis_gt = [(128, 128, 255),(128, 255, 128),(255, 128, 128)]
        else:
            self.color_rot_axis = [(128, 0, 0), (0, 128, 0),(0, 0, 128)]
            self.color_rot_axis_gt=[(255, 128, 128), (128, 255, 128),(128, 128, 255)]

    def parse_results(self, sample, result, difficult):
        im = sample['image'][0]
        output = result['output']
        centerdir_gt = sample.get('centerdir_groundtruth')
        gt_centers_dict = sample['center_dict']

        gt_list = np.array([gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())])

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        return im, output, centerdir_gt, gt_list, is_difficult_gt

    def parse_3d_info_from_sample(self, sample):
        
        assert 'depth' in sample, 'Missing "depth" in sample needed for 3d information'
        assert 'K' in sample, 'Missing camera calibration matrix "K" in sample needed for 3d information'

        gt_depth = sample['depth']
        gt_cam_K = sample['K']

        return gt_depth, gt_cam_K

    @staticmethod
    def plot_orientations(ax_, pred_list, pred_match, sin_angle, cos_angle, r, **plot_args):

        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]

            ax_.plot(pred_list_true[:, 0], pred_list_true[:, 1], 'gx', **plot_args)
            ax_.plot(pred_list_false[:, 0], pred_list_false[:, 1], 'rx', **plot_args)

            for p in pred_list_true:
                sin_6dof = sin_angle[:,int(p[1]), int(p[0])]
                cos_6dof = cos_angle[:,int(p[1]), int(p[0])]
                pred_angle = torch.atan2(sin_6dof,cos_6dof) # convert to angle to fix scaling
                sin_6dof = torch.sin(pred_angle)
                cos_6dof = torch.cos(pred_angle)
                for s,c in zip(sin_6dof, cos_6dof):
                    
                    s = s.cpu().numpy()
                    c = c.cpu().numpy()
                    ax_.plot([p[0], p[0]+c*r], [p[1], p[1]+s*r], 'g')

    @staticmethod
    def plot_orientations_cv(img, pred_list, pred_poly_mask, pred_match, sin_angle, cos_angle, gt_list, gt_sin_angle, gt_cos_angle, is_difficult_gt, 
                             gt_depth=None, gt_cam_K=None, gt=False, bbox=False, predictions_args=dict(), bbox_args=dict(), show_rot_axis=(True,True,True), 
                             color_rot_axis=[(128, 0, 0), (0, 128, 0),(0, 0, 128)], color_rot_axis_gt=[(255, 128, 128), (128, 255, 128),(128, 128, 255)],
                             axis_from_eulers=False, plot_rank_number=False, confidence_heatmap=None):
        predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        predictions_args_.update(predictions_args)
        bbox_args_ = dict(thickness=1)
        bbox_args_.update(bbox_args)

        r = 30

        if show_rot_axis is None:
            show_rot_axis = (True,True,True)

        assert len(show_rot_axis) == len(sin_angle)
        assert len(show_rot_axis) == len(cos_angle)

        assert gt_depth is not None or not axis_from_eulers, "Depth information required if ploting from axis_from_eulers"
        assert gt_cam_K is not None or not axis_from_eulers, "Camera calibration (cam_K) information required if ploting from axis_from_eulers"

        if gt:
            assert len(show_rot_axis) == len(gt_sin_angle)
            assert len(show_rot_axis) == len(gt_sin_angle)

            for i, p in enumerate(gt_list):
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4, color=(0, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                cv2.circle(img, (int(p[1]), int(p[0])), radius=4, color=(0, 0, 0), thickness=1)

                sin_6dof = gt_sin_angle[:,0,int(p[0]), int(p[1])]
                cos_6dof = gt_cos_angle[:,0,int(p[0]), int(p[1])]

                if axis_from_eulers:
                    plot_axis = get_pose_axis_from_trigonometric_euler_and_depth(p[::-1], sin_6dof, cos_6dof, gt_depth, gt_cam_K)
                else:
                    plot_axis = get_pose_axis_from_trigonometric_param(p[::-1], sin_6dof, cos_6dof, r)

                for pt, color, show in zip(plot_axis, color_rot_axis_gt, show_rot_axis):
                    if show:
                        cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), color=color, thickness=4)

        if len(pred_list) > 0:            
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]
            
            pred_rank_true = np.argwhere(pred_match[:, 0] > 0)
            pred_rank_false = np.argwhere(pred_match[:, 0] <= 0)

            for p in pred_list_true: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 255, 0), **predictions_args_)
            for p in pred_list_false: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 255), **predictions_args_)

            for plot_list, pred_rank, color in [(pred_list_true, pred_rank_true, (0, 255, 0)),
                                                (pred_list_false, pred_rank_false, (0, 0, 255))]:
                for p,rank in zip(plot_list,pred_rank):
                    sin_6dof = sin_angle[:,int(p[1]), int(p[0])]
                    cos_6dof = cos_angle[:,int(p[1]), int(p[0])]
    
                    if axis_from_eulers:
                        plot_axis = get_pose_axis_from_trigonometric_euler_and_depth(p, sin_6dof, cos_6dof, gt_depth, gt_cam_K, length=0.1) # length=0.005 + (1-rank[0]/(len(pred_rank_true)+len(pred_rank_false))*0.010))
                    else:
                        plot_axis = get_pose_axis_from_trigonometric_param(p, sin_6dof, cos_6dof, r)

                    for pt, color, show in zip(plot_axis, color_rot_axis, show_rot_axis):
                        if show:
                            cv2.line(img, (int(pt[0][0]), int(pt[0][1])), (int(pt[1][0]), int(pt[1][1])), 
                                     color=color, thickness=4)
                    
                    if plot_rank_number:
                        text = f"{rank[0]}"
                        if confidence_heatmap is not None:
                             text= f"{text}: {confidence_heatmap[0,int(p[1]), int(p[0])]*100:.1f}"
                        cv2.putText(img, text, (int(pt[0][0]), int(pt[0][1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color = (255, 0, 0) , thickness=2)

        if bbox:
            poly_true = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask
                         if is_correct_ap50]
            poly_false = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask
                          if not is_correct_ap50]
            cv2.polylines(img, poly_true, isClosed=True, color=(0, 255, 0), **bbox_args_)
            cv2.polylines(img, poly_false, isClosed=True, color=(0, 0, 255), **bbox_args_)
        return img

    def visualize_pylab(self, sample, result, pred_list, pred_score, pred_poly_mask, pred_gt_match, difficult, base, save_dir=None, plot_bbox_all=False):

        im, output, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        # np.save('im', im.clone().cpu().numpy())
        # np.save('output', output.clone().cpu().numpy())
        # np.save('gt_list', gt_list)

        ENABLE_6DOF = output.shape[1]>=10
        print("ENABLE_6DOF", ENABLE_6DOF)

        if ENABLE_6DOF:

            pred_angle = result['pred_angle']
            orientation_dims = pred_angle.shape[-1]

            assert centerdir_gt is not None

            from models.center_groundtruth import CenterDirGroundtruth

            gt_maps_keys = ['gt_orientation_sin', 'gt_orientation_cos', 'gt_sin_th', 'gt_cos_th']
            gt_maps = CenterDirGroundtruth.parse_single_batch_groundtruth_map(centerdir_gt, keys=gt_maps_keys)
            gt_orientation_sin, gt_orientation_cos, gt_sin_th, gt_cos_th = gt_maps

            orientation_sin_pred = output[0, 3:3+orientation_dims, ...]
            orientation_cos_pred = output[0, 3+orientation_dims:3+orientation_dims+orientation_dims, ...]

            r = 30

            # function pointers - for orientations
            plot_predictions = partial(self.plot_orientations, pred_list=pred_list, pred_match=pred_gt_match, sin_angle=orientation_sin_pred, cos_angle=orientation_cos_pred, r=r)
            plot_bbox_predictions = partial(self.plot_bbox_predictions, pred_poly_mask=pred_list)

            fig_img, ax = self.display(im.cpu()[:3,...], 'image', force_draw=False)
            plot_predictions(ax, markersize=10, markeredgewidth=2)

            if len(gt_list[is_difficult_gt == 0]) > 0:
                ax.plot(gt_list[is_difficult_gt == 0, 1], gt_list[is_difficult_gt==0, 0], 'g.',
                        markersize=5, markeredgewidth=0.2, markerfacecolor=(0,1,0,1), markeredgecolor=(0, 0, 0, 1))
            if len(gt_list[is_difficult_gt != 0]) > 0:
                ax.plot(gt_list[is_difficult_gt!=0, 1], gt_list[is_difficult_gt != 0, 0], 'y.',
                        markersize=5, markeredgewidth=0.2, markerfacecolor=(1,1,0,1), markeredgecolor=(0, 0, 0, 1))

            for i, p in enumerate(gt_list):
                sin_6dof = gt_orientation_sin[:orientation_dims,0,int(p[0]), int(p[1])]
                cos_6dof = gt_orientation_cos[:orientation_dims,0,int(p[0]), int(p[1])]
                for s,c in zip(sin_6dof, cos_6dof):
                    s = s.cpu().numpy()
                    c = c.cpu().numpy()
                    ax.plot([p[1], p[1]+c*r], [p[0], p[0]+s*r], 'b')

            if self.show_regressed_maps:
                fig_centers, ax = self.display([((output[0, 2])).detach().cpu(),
                                            (output[0, 3]).detach().cpu(),
                                            (output[0, 0]).detach().cpu(), (output[0, 1]).detach().cpu()], 'centers', force_draw=False)
                for ax_i in ax:
                    plot_predictions(ax_i, markersize=4, markeredgewidth=1)
                    if plot_bbox_all: plot_bbox_predictions(ax, markersize=10, markeredgewidth=2)

                    for i, p in enumerate(gt_list):
                        sin_6dof = gt_orientation_sin[:orientation_dims,0,int(p[0]), int(p[1])]
                        cos_6dof = gt_orientation_cos[:orientation_dims,0,int(p[0]), int(p[1])]
                        for s, c in zip(sin_6dof, cos_6dof):
                            s = s.cpu().numpy()
                            c = c.cpu().numpy()
                            ax_i.plot([p[1], p[1]+c*r], [p[0], p[0]+s*r], 'b')

        else:
            from matplotlib import pyplot as plt
            r = 30

            plot_batch_i = 0
            orientation_sin_gt = centerdir_gt[6, 0, ...].clone().cpu().numpy()
            orientation_cos_gt = centerdir_gt[9, 0, ...].clone().cpu().numpy()

            im = im.cpu()
            print("im", im.shape, im.dtype)

            if im.shape[0]==4:
                im = im[:3,...]

            _, ax = self.display(im, 'image')
            self.plot_gt(ax, gt_list, is_difficult_gt)

            S = output[plot_batch_i, 3].detach().cpu().numpy()
            C = output[plot_batch_i, 4].detach().cpu().numpy()
            angle = np.arctan2(S,C)+np.pi
            skip = 10
            scale = 0.15
            scale_units = 'x'
            [X,Y] = np.meshgrid(np.arange(0,angle.shape[0], skip),np.arange(0,angle.shape[1], skip))
            S = np.sin(angle)
            C = np.cos(angle)
            s = S[X,Y]
            c = C[X,Y]
            ax.quiver(Y, X, s, c, angles="xy", scale=scale, scale_units=scale_units)

            for c in gt_list:

                i = int(c[0])
                j = int(c[1])
                sx = orientation_sin_gt[i,j]
                cx = orientation_cos_gt[i,j]
                angle = np.arctan2(sx,cx)+np.pi # convert to angle to fix scaling

                sx = np.sin(angle)
                cx = np.cos(angle)
                ax.plot(j,i,'r*')
                ax.plot([j,j+sx*r], [i, i+cx*r], 'r')

            out_viz = [(output[plot_batch_i, 0]).detach().cpu(),
                        (output[plot_batch_i, 1]).detach().cpu(),
                        (output[plot_batch_i, 2]).detach().cpu(),
                        (output[plot_batch_i, 3]).detach().cpu(),
                        (output[plot_batch_i, 4]).detach().cpu(),
                        (output[plot_batch_i, 5]).detach().cpu(),
            ]
            _, ax = self.display(out_viz, 'pred')

            print("centerdir_gt", centerdir_gt.shape, centerdir_gt.dtype)
            
            _, ax = self.display((
                                centerdir_gt[2, 0, ].cpu().numpy(), # sin
                                centerdir_gt[3, 0, ].cpu().numpy(), # cos
                                ), 'centerdir_gt')
            self.plot_gt(ax, gt_list, is_difficult_gt)



    def visualize_opencv(self, sample, result, pred_list, pred_score, pred_poly_mask, pred_gt_match, difficult, base, save_dir, plot_bbox_all=False):

        im, output, centerdir_gt, gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        pred_angle = result['pred_angle']
        orientation_dims = pred_angle.shape[-1]

        assert centerdir_gt is not None

        from models.center_groundtruth import CenterDirGroundtruth

        gt_maps_keys = ['gt_orientation_sin', 'gt_orientation_cos', 'gt_sin_th', 'gt_cos_th']
        gt_maps = CenterDirGroundtruth.parse_single_batch_groundtruth_map(centerdir_gt, keys=gt_maps_keys)
        gt_orientation_sin, gt_orientation_cos, gt_sin_th, gt_cos_th = gt_maps

        gt_orientation_sin = gt_orientation_sin[:orientation_dims]
        gt_orientation_cos = gt_orientation_cos[:orientation_dims]

        sin_pred = output[0, 0, ...]
        cos_pred = output[0, 1, ...]

        OFFSET=3
        orientation_sin_pred = output[0, OFFSET:OFFSET+orientation_dims, ...]
        orientation_cos_pred = output[0, OFFSET+orientation_dims:OFFSET+orientation_dims+orientation_dims, ...]


        # function pointer
        plot_predictions = partial(self.plot_orientations_cv,
                                pred_list=pred_list, pred_poly_mask=pred_poly_mask, pred_match=pred_gt_match,
                                sin_angle=orientation_sin_pred.detach().cpu(), cos_angle=orientation_cos_pred.detach().cpu(),
                                gt_list=gt_list, gt_sin_angle=gt_orientation_sin.detach().cpu(), gt_cos_angle=gt_orientation_cos.detach().cpu(),
                                is_difficult_gt=is_difficult_gt, show_rot_axis=self.show_rot_axis, plot_rank_number=self.plot_rank_number,
                                color_rot_axis=self.color_rot_axis, color_rot_axis_gt=self.color_rot_axis_gt)

        # update params if ploting pose axis from eulers (using depth and camera calibration)
        if self.axis_from_eulers:
            assert orientation_dims == 3, 'Requires 6-DoF regression when ploting pose axis from eulers'

            gt_depth, gt_cam_K = self.parse_3d_info_from_sample(sample)
            
            plot_predictions = partial(plot_predictions, gt_depth=gt_depth, gt_cam_K=gt_cam_K, axis_from_eulers=self.axis_from_eulers)

        if self.show_confidence_score:
            confidence_heatmap = output[0, OFFSET+2*orientation_dims:OFFSET+2*orientation_dims+1, ...]
            plot_predictions = partial(plot_predictions, confidence_heatmap=confidence_heatmap.detach().cpu())


        im = im.cpu()
        if im.shape[0]>3:
            im = im[:3,...]

        fig_img = self.display_opencv(im, 'image', plot_fn=partial(plot_predictions, gt=True, bbox=True))

        if self.show_regressed_maps:
            regressed_maps = [sin_pred.detach().cpu(), cos_pred.detach().cpu(), torch.atan2(sin_pred, cos_pred).detach().cpu()] \
                                + list(orientation_sin_pred.detach().cpu()) \
                                + list(orientation_cos_pred.detach().cpu()) \
                                + list(torch.atan2(orientation_sin_pred, orientation_cos_pred).detach().cpu())
            regressed_maps_colormap = [cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA, cv2.COLORMAP_HSV] \
                                        + [cv2.COLORMAP_PARULA] * len(orientation_sin_pred) \
                                        + [cv2.COLORMAP_PARULA] * len(orientation_cos_pred) \
                                        + [cv2.COLORMAP_HSV] * len(orientation_cos_pred)                                            

            if self.show_confidence_score:
                regressed_maps += list(confidence_heatmap.detach().cpu())
                regressed_maps_colormap += [cv2.COLORMAP_PARULA]

            fig_centers = self.display_opencv(regressed_maps,'centers',plot_fn=partial(plot_predictions, bbox=plot_bbox_all), image_colormap=regressed_maps_colormap )

        if centerdir_gt is not None and self.show_gt_maps:
            fig_centerdir = self.display_opencv([torch.abs(sin_pred.detach().cpu() - gt_sin_th[0].cpu()),
                                                 torch.abs(cos_pred.detach().cpu() - gt_cos_th[0].cpu()),]
                                                + list(torch.abs(orientation_sin_pred.detach().cpu() - gt_orientation_sin[:,0].cpu()))
                                                + list(torch.abs(orientation_cos_pred.detach().cpu() - gt_orientation_cos[:,0].cpu()))
                                                + list(gt_orientation_sin[:, 0].cpu())
                                                + list(gt_orientation_cos[:, 0].cpu()),
                                                'gt-diff',
                                                plot_fn=partial(plot_predictions, bbox=plot_bbox_all))
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '%s_0.img.png' % base), fig_img)
            if 'fig_centers' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.centers.png' % base), fig_centers)
            if 'fig_centerdir' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.gt-diff.png' % base), fig_centerdir)

class OrientationVisualizeTrain(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'pred', 'sigma', 'seed', 'centerdir_gt', 'conv_centers', 'fourier_mask']

    def __init__(self, keys=(), **kwargs):
        super(OrientationVisualizeTrain, self).__init__(keys=self.KEYS + list(keys), **kwargs)

    def visualize_pylab(self, im, output, pred_mask=None, center_conv_resp=None, centerdir_gt=None,
                        gt_centers_dict=None, gt_difficult=None, log_r_fn=None, plot_batch_i=0, device=None, denormalize_args=None):
        with torch.no_grad():
            gt_list = np.array([c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]),0,gt_difficult.shape[0]-1),
                                                         np.clip(int(c[1]),0,gt_difficult.shape[1]-1)].item() != 0 for c in gt_list])

            ENABLE_6DOF = output.shape[1]>=10

            r = 30

            if ENABLE_6DOF:
                orientation_sin_gt = centerdir_gt[0][plot_batch_i, 6:9].clone().cpu().numpy()[:,0,...]
                orientation_cos_gt = centerdir_gt[0][plot_batch_i, 9:12].clone().cpu().numpy()[:,0,...]

                _, ax = self.display(im[plot_batch_i].cpu()[:3,...], 'image', denormalize_args=denormalize_args)
                self.plot_gt(ax, gt_list, is_difficult_gt)
                
                for p in gt_list:
                    s = orientation_sin_gt[2, int(p[0]), int(p[1])]
                    c = orientation_cos_gt[2, int(p[0]), int(p[1])]
                    ax.plot([p[1], p[1]+c*r], [p[0], p[0]+s*r], 'lime')

                    gt_angle = np.degrees(np.arctan2(s,c))

                out_viz = [(output[plot_batch_i, 0]).detach().cpu(),
                            (output[plot_batch_i, 1]).detach().cpu()]
                out_viz += list(output[plot_batch_i, 3:9].detach().cpu())
                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    from models.center_groundtruth import CenterDirGroundtruth

                    gt_maps_keys = ['gt_orientation_sin', 'gt_orientation_cos', 'gt_sin_th', 'gt_cos_th']
                    gt_maps = CenterDirGroundtruth.parse_groundtruth_map(centerdir_gt, keys=gt_maps_keys)
                    gt_sin_orientation, gt_cos_orientation, gt_sin_th, gt_cos_th = gt_maps

                    out_viz += [torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu(),
                                torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),]
                    out_viz += list(torch.abs(output[plot_batch_i, 3:9] - torch.cat((gt_sin_orientation[plot_batch_i, :,0], gt_cos_orientation[plot_batch_i, :,0])).to(device)).detach().cpu())
                else:
                    out_viz += [(output[plot_batch_i, 3 + i]).detach().cpu() for i in
                                range(len(output[plot_batch_i, :]) - 4)]


                _, ax = self.display(out_viz, 'centers')

                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    _, ax = self.display((
                                        centerdir_gt[0][plot_batch_i, 0].cpu(),
                                        centerdir_gt[0][plot_batch_i, 1].cpu(),
                                        centerdir_gt[0][plot_batch_i, 2].cpu(),
                                        centerdir_gt[0][plot_batch_i, 3].cpu(),
                                        centerdir_gt[0][plot_batch_i, 6].cpu(),
                                        centerdir_gt[0][plot_batch_i, 7].cpu(),
                                        centerdir_gt[0][plot_batch_i, 8].cpu(),
                                        centerdir_gt[0][plot_batch_i, 9].cpu(),
                                        centerdir_gt[0][plot_batch_i, 10].cpu(),
                                        centerdir_gt[0][plot_batch_i, 11].cpu(),
                                        ), 'centerdir_gt')
                    self.plot_gt(ax, gt_list, is_difficult_gt)

            else:
                from matplotlib import pyplot as plt

                orientation_sin_gt = centerdir_gt[0][plot_batch_i, 6].clone().cpu().numpy()
                orientation_cos_gt = centerdir_gt[0][plot_batch_i, 9].clone().cpu().numpy()

                S = output[plot_batch_i, 3].detach().cpu().numpy()
                C = output[plot_batch_i, 4].detach().cpu().numpy()
                angle = np.arctan2(S,C)+np.pi
                skip = 10
                scale = 0.15
                scale_units = 'x'
                [X,Y] = np.meshgrid(np.arange(0,angle.shape[0], skip),np.arange(0,angle.shape[1], skip))
                S = np.sin(angle)
                C = np.cos(angle)
                s = S[X,Y]
                c = C[X,Y]

                im = im[plot_batch_i].cpu()[:3,...]

                _, ax = self.display(im, 'image', denormalize_args=denormalize_args)
                self.plot_gt(ax, gt_list, is_difficult_gt)

                ax = [ax]
                for a in ax:
                    a.quiver(Y, X, s, c, angles="xy", scale=scale, scale_units=scale_units)

                    for p in gt_list:
                        sx = orientation_sin_gt[0, int(p[0]), int(p[1])]
                        cx = orientation_cos_gt[0, int(p[0]), int(p[1])]
                        angle = np.arctan2(sx,cx)+np.pi
                        sx = np.sin(angle)
                        cx = np.cos(angle)
                        a.plot([p[1], p[1]+sx*r], [p[0], p[0]+cx*r], 'lime')

                out_viz = [(output[plot_batch_i, 0]).detach().cpu(),
                            (output[plot_batch_i, 1]).detach().cpu(),
                            (output[plot_batch_i, 2]).detach().cpu(),
                            (output[plot_batch_i, 3]).detach().cpu(),
                            (output[plot_batch_i, 4]).detach().cpu(),
                            (output[plot_batch_i, 5]).detach().cpu(),
                ]

                _, ax = self.display(out_viz, 'pred')
                
                
                _, ax = self.display((
                                    centerdir_gt[0][plot_batch_i, 0].cpu(),
                                    centerdir_gt[0][plot_batch_i, 1].cpu(),
                                    centerdir_gt[0][plot_batch_i, 2].cpu(), # sin
                                    centerdir_gt[0][plot_batch_i, 3].cpu(), # cos
                                    centerdir_gt[0][plot_batch_i, 4].cpu(),
                                    centerdir_gt[0][plot_batch_i, 5].cpu(),
                                    centerdir_gt[0][plot_batch_i, 6].cpu(),

                                    centerdir_gt[0][plot_batch_i, 7].cpu(),
                                    centerdir_gt[0][plot_batch_i, 8].cpu(),

                                    centerdir_gt[0][plot_batch_i, 9].cpu(),
                                    centerdir_gt[0][plot_batch_i, 10].cpu(),
                                    centerdir_gt[0][plot_batch_i, 11].cpu(),

                                    ), 'centerdir_gt')

                self.plot_gt(ax, gt_list, is_difficult_gt)