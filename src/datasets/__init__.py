from .CenterDirGroundtruthDataset import CenterDirGroundtruthDataset
from .LockableSeedRandomAccess import LockableSeedRandomAccess
from .MuJoCoDataset import MuJoCoDataset
from .ViCoSTowelDataset import ViCoSTowelDataset
from models.center_groundtruth import CenterDirGroundtruth

def get_dataset(name, dataset_opts):
    if name.lower() == "mujoco":
        dataset = MuJoCoDataset(**dataset_opts)
    elif name.lower() == "vicos_towel":
        dataset = ViCoSTowelDataset(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))

    return dataset

def get_centerdir_dataset(name, dataset_opts, centerdir_gt_opts=None, centerdir_groundtruth_op=None, no_groundtruth=False):
    dataset = get_dataset(name, dataset_opts)

    if no_groundtruth:
        dataset.return_gt_heatmaps = False
        dataset.return_gt_box_polygon = False
        dataset.return_gt_polygon = False
        dataset.return_image = True
        return dataset, None

    if centerdir_gt_opts is not None and len(centerdir_gt_opts) > 0:
        if centerdir_groundtruth_op is None:
            centerdir_groundtruth_op = CenterDirGroundtruth(**centerdir_gt_opts)

        dataset = CenterDirGroundtruthDataset(dataset, centerdir_groundtruth_op)
        return dataset, centerdir_groundtruth_op
    else:
        return dataset, None

