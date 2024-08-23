import torch

class UnbalancedWeighting:
    def __init__(self, border_weight=1.0, border_weight_px=0, add_distance_gauss_weight=0, *args, **kwargs):
        self.border_weight = border_weight
        self.border_weight_px = border_weight_px

        self.add_distance_gauss_weight = add_distance_gauss_weight

    def __call__(self, gt_instances, gt_ignore=None, gt_R=None, w_fg=1, w_bg=1,*args, **kwargs):

        batch_size, height, width = gt_instances.shape

        bg_mask = (gt_instances == 0).unsqueeze(1)
        fg_mask = bg_mask == False

        mask_weights = torch.ones_like(bg_mask, dtype=torch.float32, requires_grad=False, device=gt_instances.device)

        mask_weights[fg_mask] = w_fg
        mask_weights[bg_mask] = w_bg

        # apply additional weights around borders
        if self.border_weight_px > 0:
            mask_weights = self._apply_border_weights(mask_weights)

        # treat each pixel equally but do not count ignored pixels
        if gt_ignore is not None:
            mask_weights *= 1 - gt_ignore.type(mask_weights.type())
            mask_weights /= (~gt_ignore).sum()
        else:
            mask_weights /= (height * width * batch_size)

        # apply additional weight based on distance to center
        if self.add_distance_gauss_weight > 0:
            mask_weights = self._apply_gauss_distance_weights(mask_weights, gt_R)

        return mask_weights

    def _apply_border_weights(self, W):
        B = self.border_weight_px

        mask_border = torch.ones_like(W, dtype=torch.bool, requires_grad=False, device=W.device)

        mask_border[:, :, B:W.shape[2] - B, B:W.shape[3] - B] = 0
        W[mask_border] *= self.border_weight

        return W

    def _apply_gauss_distance_weights(self, W, R):
        return W * torch.exp(-R / (2 * self.add_distance_gauss_weight ** 2))
