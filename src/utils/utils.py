import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

from utils.visualize.vis import Visualizer

import scipy
import torch.nn as nn

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

class GaussianLayer(nn.Module):
    def __init__(self, num_channels=1, sigma=3):
        super(GaussianLayer, self).__init__()

        self.sigma = sigma
        self.kernel_size = int(2 * np.ceil(3*self.sigma - 0.5) + 1)

        self.conv = nn.Conv2d(num_channels, num_channels, self.kernel_size, stride=1,
                              padding=self.kernel_size//2, bias=None, groups=num_channels)


        self.weights_init()
    def forward(self, x):
        return self.conv(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size,self.kernel_size))
        n[self.kernel_size//2,self.kernel_size//2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


def tensor_mask_to_ids(mask):
    ids = {i.item(): (mask == i).nonzero().cpu().numpy()
                for i in mask.unique() if i > 0}
    ids = {i: set(np.ravel_multi_index((np.array(i_mask)[:, 0], np.array(i_mask)[:, 1]), dims=mask.shape[-2:]))
                for i, i_mask in ids.items()}

    return ids

def ids_to_tensor_maks(ids, out_shape):
    out = np.zeros(out_shape)
    for i, indices in ids.items():
        indices = np.unravel_index(indices,out_shape)
        out[(indices[0], indices[1])] = i

    return out

def instance_poly_to_variable_array(polygon_list):
    import numpy as np

    if polygon_list is not None and len(polygon_list) > 0:
        if type(polygon_list) in [list,tuple]:
            # convert from list of [Nx2] to [Nx3] where first value in second axis defines ID of instance
            polygon_list = [np.concatenate(((i + 1) * np.ones((len(p), 1)), p), axis=1) for i, p in enumerate(polygon_list)]
            polygon_list = np.concatenate(polygon_list, axis=0)
        elif len(polygon_list.shape) == 3:
            idx = np.expand_dims(np.repeat(np.expand_dims(np.arange(len(polygon_list)),1),(polygon_list.shape[1]),axis=1),2) + 1
            polygon_list = np.concatenate((idx,polygon_list),axis=2)
        elif len(polygon_list.shape) != 2 or polygon_list.shape[1] != 3:
            raise Exception("Invalid input polygon_list: should be list of Nx2, array of Nx2 or array of Nx3")
    else:
        polygon_list = np.zeros((0,3))

    return polygon_list

import torch
import re
import collections
#from torch._six import string_classes # torch._six does not exist in latest pytorch !!
string_classes = (str,)

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

from torch.nn.utils.rnn import pad_sequence

def variable_len_collate(batch, batch_first=True, padding_value=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        numel = [x.numel() for x in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            storage = elem.storage()._new_shared(sum(numel))
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out) if np.all(numel[0] == numel) else pad_sequence(batch,
                                                                                             batch_first=batch_first,
                                                                                             padding_value=padding_value)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return variable_len_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: variable_len_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(variable_len_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [variable_len_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
