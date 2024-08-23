import collections
import os
import shutil

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import cv2

class Visualizer:

    def __init__(self, keys, to_file_only=False, tensorboard_dir=None, tensorboard_continue=False, autoadjust_figure_size=False, impath2name_fn=None):
        self.wins = {k:None for k in keys}
        self.tensorboard_dir = tensorboard_dir

        if self.tensorboard_dir is not None:
            # clear tensorboard folder
            if os.path.exists(self.tensorboard_dir) and not tensorboard_continue:
                shutil.rmtree(self.tensorboard_dir, ignore_errors=True)

            os.makedirs(self.tensorboard_dir, exist_ok=True)

        self.to_file_only = to_file_only
        self.tf_writer = SummaryWriter(log_dir=tensorboard_dir) if tensorboard_dir is not None else None

        self.autoadjust_figure_size = autoadjust_figure_size
        if impath2name_fn is None:
            self.impath2name_fn = lambda name: os.path.splitext(os.path.basename(name))
        else:
            self.impath2name_fn = impath2name_fn

    def display(self, image, key, save=None, title=None, denormalize_args=None, force_draw=True, **kwargs):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            n_cols = int(np.maximum(1,np.ceil(np.sqrt(n_images))))
            n_rows = int(np.maximum(1,np.ceil(n_images/n_cols)))
            self.wins[key] = plt.subplots(ncols=n_cols,nrows=n_rows)

        fig, ax = self.wins[key]

        if isinstance(ax, collections.Iterable):
            ax = ax.reshape(-1)
            n_axes = len(ax)
        else:
            n_axes= 1
        assert n_images <= n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image, denormalize_args), **kwargs)
            max_size = image.shape[-2:]
        else:
            max_size = 0,0
            for i,ax_i in enumerate(ax):
                ax_i.cla()
                ax_i.set_axis_off()
                if i < n_images:
                    ax_i.imshow(self.prepare_img(image[i], denormalize_args), **kwargs)
                    max_size = max(max_size[0], image[i].shape[-2]), max(max_size[1], image[i].shape[-1])

        fig.subplots_adjust(top=1.0, bottom=0.00, left=0.01, right=0.99, wspace=0.01,hspace=0.01)

        if title is not None:
            fig.suptitle(title)

        if self.autoadjust_figure_size:
            if min(max_size) > 2*1024:
                f = (2*1024.0)/min(max_size)
                max_size = max_size[0]*f, max_size[1]*f

            n_cols, n_rows = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
            fig.set_size_inches(n_rows*max_size[1]/100, n_cols*max_size[0]/100)

        if save is not None:
            fig.savefig(save)

        if force_draw or True:
            plt.draw()
            self.mypause(0.001)

        return fig,ax

    def display_opencv(self, image, key, save=None, title=None, denormalize_args=None, plot_fn=None, image_colormap=None, **kwargs):

        def prepare_img_cv(im, colormap=cv2.COLORMAP_PARULA):
            im = self.prepare_img(im, denormalize_args)

            if len(im.shape) == 3 and im.shape[-1] == 3:
                # convert RGB to BGR
                im = im[:,:,[2,1,0]]

            if len(im.shape) == 2 and im.dtype.kind != 'u':
                im = (((im-im.min()) / (im.max() - im.min())) * 255).astype(np.uint8)
                im = cv2.applyColorMap(im, colormap)
            elif im.dtype.kind != 'u':
                im = (im * 255).astype(np.uint8)

            im = np.ascontiguousarray(im, dtype=np.uint8)

            if plot_fn is not None:
                im = plot_fn(im)

            return im

        if not isinstance(image, (list, tuple)):
            image = [image]

        n_images = len(image)

        n_cols = int(np.maximum(1, np.ceil(np.sqrt(n_images))))
        n_rows = int(np.maximum(1, np.ceil(n_images / n_cols)))

        if image_colormap is None:
            image_colormap = [cv2.COLORMAP_PARULA]*n_images
        elif not isinstance(image_colormap, (list, tuple)):
            image_colormap = [image_colormap]

        # prepare images
        I = [prepare_img_cv(I,cmap) for I,cmap in zip(image,image_colormap)]

        # convert to grid with n_cols and n_rows
        if len(I) < n_cols*n_rows:
            I += [np.ones_like(I[0])*255] * (n_cols*n_rows - len(I))

        I = np.concatenate([np.concatenate(I[i*n_cols:(i+1)*n_cols], axis=1) for i in range(n_rows)],axis=0)

        return I

    def log_grouped_scalars_tf(self, grouped_dict, iter, name):
        if self.tf_writer is not None:
            self.tf_writer.add_scalars(name, {k:v for k,v in grouped_dict.items() if type(v) is not dict}, global_step=iter)

            for k, v in grouped_dict.items():
                if type(v) is dict:
                    self.log_grouped_scalars_tf(grouped_dict=v, iter=iter, name=k)

    def log_conv_weights(self, model, iter):
        if self.tf_writer is not None:
            model = model.module
            fpn_model = model.model

            enc_layers = [('ly1',fpn_model.encoder.layer1),
                          ('ly2',fpn_model.encoder.layer2),
                          ('ly3',fpn_model.encoder.layer3),
                          ('ly4',fpn_model.encoder.layer4)]
            dec_layers = [('p4',fpn_model.decoder.p4),
                          ('p3', fpn_model.decoder.p3),
                          ('p2', fpn_model.decoder.p2)]

            for n, ly in enc_layers:
                self.tf_writer.add_histogram(n, ly[-1].conv2.weight, global_step=iter)
                if ly[-1].conv2.weight.grad is not None:
                    self.tf_writer.add_histogram(n+'-grad', ly[-1].conv2.weight.grad, global_step=iter)

            for n, ly in dec_layers:
                self.tf_writer.add_histogram(n, ly.skip_conv.weight, global_step=iter)
                if ly.skip_conv.weight.grad is not None:
                    self.tf_writer.add_histogram(n + '-grad', ly.skip_conv.weight.grad, global_step=iter)

            final_conv_index =  2 if model.use_custom_fpn else 0
            self.tf_writer.add_histogram('seg_sin', fpn_model.segmentation_head[final_conv_index].weight[0], global_step=iter)
            self.tf_writer.add_histogram('seg_cos', fpn_model.segmentation_head[final_conv_index].weight[1], global_step=iter)
            self.tf_writer.add_histogram('seg_r', fpn_model.segmentation_head[final_conv_index].weight[2], global_step=iter)
            self.tf_writer.add_histogram('seg_cls', fpn_model.segmentation_head[final_conv_index].weight[3], global_step=iter)

            self.tf_writer.add_histogram('seg_sin_grad', fpn_model.segmentation_head[final_conv_index].weight.grad[0], global_step=iter)
            self.tf_writer.add_histogram('seg_cos_grad', fpn_model.segmentation_head[final_conv_index].weight.grad[1], global_step=iter)
            self.tf_writer.add_histogram('seg_r_grad', fpn_model.segmentation_head[final_conv_index].weight.grad[2], global_step=iter)
            self.tf_writer.add_histogram('seg_cls_grad', fpn_model.segmentation_head[final_conv_index].weight.grad[3], global_step=iter)

            if model.use_custom_fpn:
                self.tf_writer.add_histogram('seg_beforlast', fpn_model.segmentation_head[0].block[0].weight, global_step=iter)
                self.tf_writer.add_histogram('seg_beforlast_grad', fpn_model.segmentation_head[0].block[0].weight.grad, global_step=iter)

    def __call__(self, *args, **kwargs):
        if self.to_file_only:
            self.visualize_opencv(*args, **kwargs)
        else:
            self.visualize_pylab(*args, **kwargs)

    def visualize_opencv(self, *args, **kwargs):
        raise Exception("Not implemented")

    def visualize_pylab(self, *args, **kwargs):
        raise Exception("Not implemented")

    @staticmethod
    def prepare_img(image, denormalize_args=None):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image = image.squeeze().numpy()
            if denormalize_args is not None:
                denorm_mean,denorm_std = denormalize_args

                image = (image * denorm_std.numpy().reshape(-1,1,1)) +  denorm_mean.numpy().reshape(-1,1,1)


        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


    @staticmethod
    def plot_gt(ax_, gt_list, is_difficult_gt):
        if type(ax_) not in [list, tuple, np.ndarray]:
            ax_ = [ax_]
        gt_list_easy, gt_list_hard = gt_list[is_difficult_gt == 0], gt_list[is_difficult_gt != 0]

        for a in ax_:
            if len(gt_list_easy) > 0:
                a.plot(gt_list_easy[:, 1], gt_list_easy[:, 0], 'g.', markersize=5, markeredgewidth=0.2,
                       markerfacecolor=(0, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
            if len(gt_list_hard) > 0:
                a.plot(gt_list_hard[:, 1], gt_list_hard[:, 0], 'y.', markersize=5, markeredgewidth=0.2,
                       markerfacecolor=(1, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))

    @staticmethod
    def plot_predictions(ax_, pred_list, pred_match, **plot_args):
        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]

            ax_.plot(pred_list_true[:, 0], pred_list_true[:, 1], 'gx', **plot_args)
            ax_.plot(pred_list_false[:, 0], pred_list_false[:, 1], 'rx', **plot_args)

    @staticmethod
    def plot_bbox_predictions(ax_, pred_poly_mask, **plot_args):
        for poly, is_correct_ap50, iou in pred_poly_mask:
            if len(poly) > 0:
                ax_.plot(poly[:, 1], poly[:, 0], 'g-' if is_correct_ap50 else 'r-', **plot_args)

    @staticmethod
    def plot_predictions_cv(img, pred_list, pred_poly_mask, pred_match, gt_list, is_difficult_gt,
                            gt=False, bbox=False, predictions_args=dict(), bbox_args=dict()):
        predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        predictions_args_.update(predictions_args)
        bbox_args_ = dict(thickness=1)
        bbox_args_.update(bbox_args)
        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :]
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :]

            for p in pred_list_true: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 255, 0),
                                                    **predictions_args_)
            for p in pred_list_false: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 255),
                                                     **predictions_args_)
        if gt:
            for i, p in enumerate(gt_list):
                cv2.circle(img, (int(p[1]), int(p[0])), radius=3,
                           color=(0, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                cv2.circle(img, (int(p[1]), int(p[0])), radius=3, color=(0, 0, 0), thickness=1)
        if bbox:
            poly_true = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask
                         if is_correct_ap50]
            poly_false = [np.round(poly[:, [1, 0]]).astype(np.int) for poly, is_correct_ap50, iou in pred_poly_mask
                          if not is_correct_ap50]
            cv2.polylines(img, poly_true, isClosed=True, color=(0, 255, 0), **bbox_args_)
            cv2.polylines(img, poly_false, isClosed=True, color=(0, 0, 255), **bbox_args_)
        return img

    @staticmethod
    def plot_predictions_cam_cv(img, pred_list, pred_match, predictions_args=dict()):
        predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        predictions_args_.update(predictions_args)
        if len(pred_list) > 0:
            pred_list_true = pred_list[pred_match[:, 0] > 0, :] / 16
            pred_list_false = pred_list[pred_match[:, 0] <= 0, :] / 16
            for p in pred_list_true: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 255, 0), **predictions_args_)
            for p in pred_list_false: cv2.drawMarker(img, (int(p[0]), int(p[1])), color=(0, 0, 255), **predictions_args_)
        return img

def show_image(image, is_bgr=True, save_to=None, return_only=False, log=False):
    """
    torch.Tensor -> C x H x W
    cv2 image:   -> H x W x C
    """
    if isinstance(image, torch.Tensor):
        if log:
            print("Converting torch.Tensor to numpy")
        image = image.detach().cpu().numpy()

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    if len(image.shape) == 3:
        if image.shape[0] <= 3:
            if log:
                print("Converting image to channels last")
            image = np.transpose(image, (1, 2, 0))

        if image.shape[2] == 1:
            if log:
                print("Removing dimension from HxWx1 image")
            image = image[:, :, 0]

    if image.dtype == np.uint8:
        if log:
            print("Scaling uint8 image")
        image = (image / np.max(image) * 255).astype(np.uint8)
    elif image.dtype == np.bool:
        if log:
            print("Converting bool image")
        image = image * 255

    if save_to is not None:
        if log:
            print(f"Saving image to ./{save_to}.png")
        cv2.imwrite(f"{save_to}.png", image)

    if is_bgr and len(image.shape) == 3:
        if log:
            print("Converting to RGB for displaying")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype == np.float and np.min(image) < 0 or np.max(image) > 1:
        if log:
            print("Scalling image to [0, 1]")
        image = (image - image.min()) / (image.max() - image.min())

    if return_only:
        return image

    plt.imshow(image)
    #plt.show(bbox_inches="tight")

    if log:
        print("Finished")
    return image