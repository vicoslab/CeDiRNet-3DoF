import os
from functools import partial

import numpy as np
import torch

import cv2

from utils.visualize.vis import Visualizer

class CentersVisualizeTest(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'centers-est', 'pred', 'sigma', 'seed', 'gt-diff']

    def __init__(self, keys=(), **kwargs):
        super(CentersVisualizeTest, self).__init__(keys=self.KEYS + list(keys), **kwargs)

    def parse_results(self, sample, result, difficult):
        im = sample['image'][0]
        output = result['output']
        center_est_imshow = result['pred_heatmap']
        centerdir_gt = sample.get('centerdir_groundtruth')
        gt_centers_dict = sample['center_dict']

        gt_list = np.array([gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())])

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        return im, output, center_est_imshow, centerdir_gt, gt_list, is_difficult_gt

    def visualize_pylab(self, sample, result, pred_list, pred_score, pred_poly_mask, pred_gt_match,
                  difficult, base, save_dir=None, plot_bbox_all=False):

        im, output, center_est_imshow, centerdir_gt, \
            gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        is_difficult_gt = np.array([difficult[np.clip(int(c[0]), 0, difficult.shape[0] - 1),
                                              np.clip(int(c[1]), 0, difficult.shape[1] - 1),].item() != 0 for c in gt_list])

        # function pointers
        plot_predictions = partial(self.plot_predictions, pred_list=pred_list, pred_match=pred_gt_match)
        plot_bbox_predictions = partial(self.plot_bbox_predictions, pred_poly_mask=pred_list)

        fig_img, ax = self.display(im.cpu(), 'image', force_draw=False)
        plot_predictions(ax, markersize=10, markeredgewidth=2)

        if len(gt_list[is_difficult_gt == 0]) > 0:
            ax.plot(gt_list[is_difficult_gt == 0, 1], gt_list[is_difficult_gt == 0, 0], 'g.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(0, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
        if len(gt_list[is_difficult_gt != 0]) > 0:
            ax.plot(gt_list[is_difficult_gt != 0, 1], gt_list[is_difficult_gt != 0, 0], 'y.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(1, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))

        plot_bbox_predictions(ax, markersize=10, markeredgewidth=2)
        
        if output is not None:
            fig_centers, ax = self.display([((output[0, 2])).detach().cpu(),
                                            (output[0, 1]).detach().cpu(),
                                            (output[0, 0]).detach().cpu()], 'centers', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)
                if plot_bbox_all: plot_bbox_predictions(ax, markersize=10, markeredgewidth=2)

        if center_est_imshow is not None:
            fig_centers_conv, ax = self.display(
                [f.detach().cpu() if type(f) == torch.Tensor else f for f in center_est_imshow],
                'centers-est', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)
                if plot_bbox_all: plot_bbox_predictions(ax, markersize=10, markeredgewidth=2)
        
        if centerdir_gt is not None and output is not None:
            fig_centerdir, ax = self.display([torch.abs(output[0, 2].detach().cpu() - centerdir_gt[0].cpu()),
                                              torch.abs(output[0, 1].detach().cpu() - centerdir_gt[3].cpu()),
                                              torch.abs(output[0, 0].detach().cpu() - centerdir_gt[2].cpu())],
                                             'gt-diff', force_draw=False)
            for ax_i in ax:
                plot_predictions(ax_i, markersize=4, markeredgewidth=1)
                if plot_bbox_all: plot_bbox_predictions(ax, markersize=10, markeredgewidth=2)

        if save_dir is not None:
            fig_img.savefig(os.path.join(save_dir, '%s_0.img.png' % base))
            if 'fig_centers' in locals():
                fig_centers.savefig(os.path.join(save_dir, '%s_1.centers.png' % base))
            if 'fig_centers_conv' in locals():
                fig_centers_conv.savefig(os.path.join(save_dir, '%s_2.centers_conv.png' % base))
            if 'fig_centerdir' in locals():
                fig_centerdir.savefig(os.path.join(save_dir, '%s_1.gt-diff.png' % base))

    def visualize_opencv(self, sample, result, pred_list, pred_score, pred_poly_mask, pred_gt_match,
                         difficult, base, save_dir, plot_bbox_all=False):

        im, output, center_est_imshow, centerdir_gt, \
            gt_list, is_difficult_gt = self.parse_results(sample, result, difficult)

        # function pointer
        plot_predictions = partial(self.plot_predictions_cv,
                                   pred_list=pred_list, pred_poly_mask=pred_poly_mask, pred_match=pred_gt_match,
                                   gt_list=gt_list, is_difficult_gt=is_difficult_gt)

        fig_img = self.display_opencv(im.cpu(), 'image', plot_fn=partial(plot_predictions, gt=True, bbox=True))
        
        if output is not None:
            fig_centers = self.display_opencv([((output[0, 2])).detach().cpu(),
                                            (output[0, 1]).detach().cpu(),
                                            (output[0, 0]).detach().cpu(),
                                            torch.atan2(output[0, 0], output[0, 1]).detach().cpu()], 'centers',
                                            plot_fn=partial(plot_predictions, bbox=plot_bbox_all),
                                            image_colormap=[cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA, cv2.COLORMAP_PARULA,
                                                            cv2.COLORMAP_HSV])
        if center_est_imshow is not None:
            fig_centers_conv = self.display_opencv(
                [f.detach().cpu() if type(f) == torch.Tensor else f for f in center_est_imshow],
                'centers-est',
                plot_fn=partial(plot_predictions, bbox=plot_bbox_all, predictions_args=dict(thickness=1)))

        if centerdir_gt is not None and output is not None:
            fig_centerdir = self.display_opencv([torch.abs(output[0, 2].detach().cpu() - centerdir_gt[0].cpu()),
                                                 torch.abs(output[0, 1].detach().cpu() - centerdir_gt[3].cpu()),
                                                 torch.abs(output[0, 0].detach().cpu() - centerdir_gt[2].cpu())],
                                                'gt-diff', plot_fn=partial(plot_predictions, bbox=plot_bbox_all))
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '%s_0.img.png' % base), fig_img)
            if 'fig_centers' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.centers.png' % base), fig_centers)
            if 'fig_centers_conv' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_2.centers_conv.png' % base), fig_centers_conv)
            if 'fig_centerdir' in locals():
                cv2.imwrite(os.path.join(save_dir, '%s_1.gt-diff.png' % base), fig_centerdir)

class CentersVisualizeTrain(Visualizer):
    # default keys for visualization windows
    KEYS = ['image', 'centers', 'pred', 'sigma', 'seed', 'centerdir_gt', 'conv_centers', 'fourier_mask']

    def __init__(self, keys=(), **kwargs):
        super(CentersVisualizeTrain, self).__init__(keys=self.KEYS + list(keys), **kwargs)

    def visualize_pylab(self, im, output, pred_mask=None, center_conv_resp=None, centerdir_gt=None,
                        gt_centers_dict=None, gt_difficult=None, log_r_fn=None, plot_batch_i=0, device=None, denormalize_args=None):

        with torch.no_grad():
            gt_list = np.array(
                [c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]), 0, gt_difficult.shape[0] - 1),
                                                         np.clip(int(c[1]), 0, gt_difficult.shape[1] - 1)].item() != 0
                                            for c in gt_list])


            _, ax = self.display(im[plot_batch_i].cpu(), 'image', denormalize_args=denormalize_args)
            self.plot_gt(ax, gt_list, is_difficult_gt)

            out_viz = [(output[plot_batch_i, 2]).detach().cpu(),
                       (output[plot_batch_i, 1]).detach().cpu(),
                       (output[plot_batch_i, 0]).detach().cpu()]
            if centerdir_gt is not None and len(centerdir_gt) > 0:
                gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0][:, 0], centerdir_gt[0][:, 1], centerdir_gt[0][:,
                                                                                                     2], \
                                                       centerdir_gt[0][:, 3]

                if log_r_fn is not None:
                    gt_R = log_r_fn(gt_R)

                out_viz += [torch.abs(output[plot_batch_i, 2] - gt_R[plot_batch_i, 0].to(device)).detach().cpu(),
                            torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),
                            torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu()]

                if output.shape[1] >= 4 and centerdir_gt[0].shape[1] >= 8:
                    gt_sin_orientation, gt_cos_orientation = centerdir_gt[0][:, 6], centerdir_gt[0][:, 7]
                    out_viz += [torch.abs(
                        output[plot_batch_i, 3] - gt_sin_orientation[plot_batch_i, 0].to(device)).detach().cpu(),
                                torch.abs(output[plot_batch_i, 4] - gt_cos_orientation[plot_batch_i, 0].to(
                                    device)).detach().cpu()]


            else:
                out_viz += [(output[plot_batch_i, 3 + i]).detach().cpu() for i in
                            range(len(output[plot_batch_i, :]) - 4)]

            _, ax = self.display(out_viz, 'centers')
            self.plot_gt(ax, gt_list, is_difficult_gt)

            if center_conv_resp is not None:
                conv_centers = center_conv_resp[plot_batch_i].detach().cpu()
                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    gt_center_mask = centerdir_gt[0][plot_batch_i, 5]
                    conv_centers = [conv_centers,
                                    torch.abs(
                                        center_conv_resp[plot_batch_i] - gt_center_mask.to(device)).detach().cpu()]

                _, ax = self.display(conv_centers, 'conv_centers')
                self.plot_gt(ax, gt_list, is_difficult_gt)

            seed = output[plot_batch_i][-1].cpu()
            self.display(seed, 'seed', vmin=0, vmax=1)

            if centerdir_gt is not None and len(centerdir_gt) > 0:
                _, ax = self.display((centerdir_gt[0][plot_batch_i, 0].cpu(),
                                      centerdir_gt[0][plot_batch_i, 1].cpu(),
                                      centerdir_gt[0][plot_batch_i, 2].cpu(),
                                      centerdir_gt[0][plot_batch_i, 3].cpu(),), 'centerdir_gt')
                self.plot_gt(ax, gt_list, is_difficult_gt)
