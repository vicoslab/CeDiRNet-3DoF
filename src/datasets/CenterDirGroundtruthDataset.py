import os.path
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess

class CenterDirGroundtruthDataset(Dataset, LockableSeedRandomAccess):

    def __init__(self, dataset, centerdir_groundtruth_op):

        self.dataset = dataset
        self.centerdir_groundtruth_op = centerdir_groundtruth_op

    def get_coco_api(self):
        return self.dataset.get_coco_api()

    def lock_samples_seed(self, index_list):
        if isinstance(self.dataset,LockableSeedRandomAccess):
            self.dataset.lock_samples_seed(index_list)
        #else:
        #    print("Warning: underlying dataset not instance of LockableSeedRandomAccess .. skipping")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        instance = sample['instance']

        # init data strucutre for groundtruth
        centerdir_groundtruth, output = self.centerdir_groundtruth_op._init_datastructure(instance.shape[-2], instance.shape[-1])

        # store cached data into sample
        if output is not None:
            sample['output'] = output
        sample['centerdir_groundtruth'] = centerdir_groundtruth

        return sample