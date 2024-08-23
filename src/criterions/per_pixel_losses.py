import torch
import torch.nn.functional as F

from functools import partial

def get_per_pixel_loss_func(loss_type):

    def abs_jit(X, Y):
        return torch.abs(X - Y)
    def mse_jit(X, Y):
        return torch.pow(X - Y, 2)
    def m4e_jit(X, Y):
        return torch.pow(X - Y, 4)

    loss_abs_fn = abs_jit
    loss_mse_fn = mse_jit
    loss_m4e_fn = m4e_jit

    loss_hinge_fn = lambda X, Y, sign_fn, eps=0: (torch.clamp_min(sign_fn(Y - X), eps) - eps)
    loss_smoothL1_fn = lambda X, Y, beta, pow: torch.where((X - Y).abs() < beta,
                                                                torch.pow(X - Y, pow) / (pow * beta),
                                                                (X - Y).abs() - 1 / float(pow) * beta)
    loss_inverted_smoothL1_fn = lambda X, Y, beta, pow: torch.where((X - Y).abs() > beta,
                                                                         torch.pow(X - Y, pow) / (pow * beta),
                                                                         (X - Y).abs() - 1 / float(pow) * beta)
    loss_bce_logits = torch.nn.BCEWithLogitsLoss(reduction='none')

    def binary_hinge_loss(X, Y):
        with torch.no_grad():
            valid_neg = (Y <= 0) * (X > 0)
            valid_pos = (Y >= 1) * (X < 1)
            valid = (valid_neg + valid_pos) > 0

        return torch.abs(X - Y) * valid.float()

    args = {}
    if type(loss_type) is dict:
        args = loss_type['args'] if 'args' in loss_type else {}
        loss_type = loss_type['type']

    if loss_type.upper() in ['L1', 'MAE']:
        return partial(loss_abs_fn, **args)
    elif loss_type.upper() in ['L2', 'MSE']:
        return partial(loss_mse_fn, **args)
    elif loss_type.lower() in ['hinge']:
        return partial(loss_hinge_fn, **args)
    elif loss_type.lower() in ['smoothl1']:
        return partial(loss_smoothL1_fn, **args)
    elif loss_type.lower() in ['inverted-smoothl1']:
        return partial(loss_inverted_smoothL1_fn, **args)
    elif loss_type.lower() in ['cross-entropy', 'bce']:
        return partial(loss_bce_logits, **args)
    elif loss_type.lower() in ['focal']:
        return lambda X, Y: sigmoid_focal_loss(X, Y, reduction="none", **args)
    else:
        raise Exception('Unsuported loss type: \'%s\'' % loss_type)


def sigmoid_focal_loss(
    inputs,
    targets,
    alpha = 0.25,
    delta = 1,
    gamma = 2,
    A = 1,
    reduction = "none"):

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    loss = ce_loss * torch.where(targets == 1,
                                 (1-p)**gamma, # foreground
                                A*(1-targets)**delta * p**gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
