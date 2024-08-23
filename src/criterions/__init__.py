from criterions.center_direction_loss import CenterDirectionLoss
from criterions.orientation_loss import OrientationLoss

def get_criterion(type, loss_opts, model, center_model):

    if type in ['CenterDirectionLoss','PolarCenterLossV2']:
        criterion = CenterDirectionLoss(center_model, **loss_opts)
    elif type in ['CenterDirectionLossOrientation','OrientationLoss']:
        criterion = OrientationLoss(center_model, **loss_opts)
    else:
        raise Exception("Unknown 'loss_type' in config: only allowed 'CenterDirectionLoss'  or 'CenterDirectionLossOrientation'")

    return criterion