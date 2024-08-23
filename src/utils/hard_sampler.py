import numpy as np
import torch
from torch.utils.data import Sampler
import pickle

from datasets import LockableSeedRandomAccess

class IHardExamplesBatchSampler:
    def has_hard_samples(self):
        raise Exception("Not implemented")
    def update_difficulty_score(self, gt_sample, difficulty_scores, index_key='index', storage_keys=[], selected_samples_only=None):
        raise Exception("Not implemented")
    def retrieve_hard_sample_storage_batch(self, ids, key=None):
        raise Exception("Not implemented")

    def get_difficulty_scores(self):
        raise Exception("Not implemented")
    def get_hard_example_indices(self):
        raise Exception("Not implemented")
    def get_avg_difficulty_score(self):
        raise Exception("Not implemented")
    def get_sample_frequency_use(self):
        raise Exception("Not implemented")


class HardExamplesBatchSampler(Sampler, IHardExamplesBatchSampler):

    def __init__(self, dataset, default_sampler, batch_size, hard_sample_size, drop_last,
                 hard_samples_selected_min_percent=0.0, hard_samples_only_min_selected_when_empty=False,
                 device=None, world_size=None, rank=None, is_distributed=False):
        if not isinstance(default_sampler, Sampler):
            raise ValueError("default_sampler should be an instance of "
                             "torch.utils.data.Sampler, but got default_sampler={}"
                             .format(default_sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not (isinstance(hard_sample_size, int) or hard_sample_size is None) or \
                hard_sample_size < 0 or hard_sample_size >= batch_size :
            raise ValueError("hard_sample_size should be a positive integer value smaller than batch_size, "
                             "but got hard_sample_size={}".format(hard_sample_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.is_distributed = is_distributed and world_size > 1
        self.world_size = world_size if self.is_distributed else 1
        self.rank = rank if self.is_distributed else 0
        self.device = device

        self.dataset = dataset
        self.default_sampler = default_sampler
        if self.is_distributed:
            self.hard_sampler = DistributedSubsetRandomSampler(list(range(len(default_sampler))),device=device)
        else:
            self.hard_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(default_sampler))))
        self.hard_sample_size = hard_sample_size if hard_sample_size is not None else 0
        self.hard_samples_selected_min_percent = hard_samples_selected_min_percent if hard_samples_selected_min_percent is not None else 0
        self.hard_samples_only_min_selected_when_empty = hard_samples_only_min_selected_when_empty
        self.batch_size = batch_size
        self.drop_last = drop_last


        self.sample_difficulty_scores = dict()
        self.sample_storage = dict()
        self.sample_storage_tmp = dict()

    def has_hard_samples(self):
        return self.hard_sample_size > 0

    def update_difficulty_score(self, gt_sample, difficulty_scores, index_key='index', storage_keys=[], selected_samples_only=None):
        assert index_key in gt_sample, "Index key %s is not present in gt_sample" % index_key

        indices = gt_sample[index_key]

        # convert to numpy
        indices = indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor) else indices
        difficulty_scores = difficulty_scores.detach().cpu().numpy() if isinstance(difficulty_scores, torch.Tensor) else difficulty_scores

        for i,l in enumerate(difficulty_scores):
            if selected_samples_only is not None and not selected_samples_only[i]:
                continue
            # get id of the sample (i.e. its index key)
            id = indices[i]

            # store its loss value
            self.sample_difficulty_scores[id] = l
            # store any additional info required to pass along for hard examples
            # (save to temporary array which will be used for next epoch)
            self.sample_storage_tmp[id] = {k:gt_sample[k][i] for k in storage_keys}

    def retrieve_hard_sample_storage_batch(self, ids, key=None):
        # convert to numpy
        ids = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else ids
        # return matching sample_storage value for hard examples (i.e. for first N samples, where N=self.hard_sample_size)
        return [self.sample_storage[id][key] if n < self.hard_sample_size and id in self.sample_storage else None for n,id in enumerate(ids)]

    def _synchronize_dict(self, array):
        return distributed_sync_dict(array, self.world_size, self.rank, self.device)

    def _recompute_hard_samples_list(self):
        if self.is_distributed:
            self.sample_difficulty_scores = self._synchronize_dict(self.sample_difficulty_scores)
        if len(self.sample_difficulty_scores) > 0:
            k = np.array(list(self.sample_difficulty_scores.keys()))
            v = np.array([self.sample_difficulty_scores[i] for i in k])
            v = (v - v.mean()) / v.std()
            # by default set all or only N-percent of most difficult
            if self.hard_samples_only_min_selected_when_empty and self.hard_samples_selected_min_percent > 0:
                # select only N-percent of most difficult samples
                idx = np.argsort(v)
                hard_ids = k[idx[-int(np.ceil(len(v) * self.hard_samples_selected_min_percent)):]]
            else:
                hard_ids = list(k)

            # limit based on standard deviation, but only if enough samples can be collected
            for std_thr in [2, 1, 0.5, 0]:
                new_hard_ids = list(k[v > std_thr])
                if len(new_hard_ids) > len(v)*self.hard_samples_selected_min_percent:
                    hard_ids = new_hard_ids
                    break

            self.hard_sampler.indices = hard_ids

            if self.rank == 0:
                print('Number of hard samples present: %d/%d' % (len(hard_ids), len(self.sample_difficulty_scores)))

        if isinstance(self.dataset,LockableSeedRandomAccess):
            # lock seeds for hard samples BUT not for the whole dataset i.e. 90% of the whole dataset
            # (otherwise this will fully lock seeds for all samples and prevent new random augmentation of samples)
            self.dataset.lock_samples_seed(self.hard_sampler.indices if len(self.hard_sampler.indices) < len(self.sample_difficulty_scores) * 0.9 else [])

        # update storage for next iteration
        self.sample_storage = self._synchronize_dict(self.sample_storage_tmp) if self.is_distributed else self.sample_storage_tmp
        self.sample_storage_tmp = dict()

    def __iter__(self):
        from itertools import islice
        self._recompute_hard_samples_list()
        max_index = len(self.default_sampler)
        if self.drop_last:
            total_batch_size = self.batch_size * self.world_size
            max_index = (max_index // total_batch_size) * total_batch_size

        batch = []
        hard_iter = iter(self.hard_sampler)
        self.usage_freq = {i: 0 for i in range(len(self.default_sampler))}
        for idx in islice(self.default_sampler,self.rank,max_index,self.world_size):
            batch.append(idx)
            # stop when spaces for normal samples filled
            if len(batch) == self.batch_size-self.hard_sample_size:
                # fill remaining places with hard examples
                # (does not need to be sync for distributed since sampling is random with replacement)
                while len(batch) < self.batch_size:
                    try:
                        batch.insert(0,next(hard_iter))
                    except StopIteration: # reset iter if no more samples
                        hard_iter = iter(self.hard_sampler)

                for b in batch: self.usage_freq[b] += 1
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            for b in batch: self.usage_freq[b] += 1
            yield batch

    def get_avg_difficulty_score(self):
        return np.array(list(self.sample_difficulty_scores.values())).mean()

    def get_difficulty_scores(self):
        return self.sample_difficulty_scores.copy()

    def get_sample_frequency_use(self):
        return self.usage_freq.copy()

    def get_hard_example_indices(self):
        return self.hard_sampler.indices

    def __len__(self):
        size_default = len(self.default_sampler)

        if self.is_distributed:
            size_default = size_default // self.world_size

        actual_batch_size = self.batch_size-self.hard_sample_size
        if self.drop_last:
            return size_default // actual_batch_size
        else:
            return (size_default + actual_batch_size - 1) // actual_batch_size


import torch.distributed as dist

class DistributedRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, device=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.device = device

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            iter_order = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).to(self.device)
        else:
            iter_order = torch.randperm(n).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return iter(iter_order.tolist())

    def __len__(self):
        return self.num_samples


class DistributedSubsetRandomSampler(Sampler):
    def __init__(self, indices, device=None):
        self.indices = indices
        self.device = device

    def __iter__(self):
        iter_order = torch.randperm(len(self.indices)).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return (self.indices[i.item()] for i in iter_order)

    def __len__(self):
        return len(self.indices)

def distributed_sync_dict(array, world_size, rank, device, MAX_LENGTH=10*2**20): # default MAX_LENGTH = 10MB
    def _pack_data(_array):
        data = pickle.dumps(_array)
        data_length = int(len(data))
        data = data_length.to_bytes(4, "big") + data
        assert len(data) < MAX_LENGTH
        data += bytes(MAX_LENGTH - len(data))
        data = np.frombuffer(data, dtype=np.uint8)
        assert len(data) == MAX_LENGTH
        return torch.from_numpy(data)
    def _unpack_data(_array):
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        return pickle.loads(data[4:data_length+4])
    def _unpack_size(_array):
        print(_array.shape, _array[:4])
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        print(data_length,data[:4])
        return data_length

    # prepare output buffer
    output_tensors = [torch.zeros(MAX_LENGTH, dtype=torch.uint8, device=device) for _ in range(world_size)]
    # pack data using pickle into input/output
    output_tensors[rank][:] = _pack_data(array)

    # sync data
    dist.all_gather(output_tensors, output_tensors[rank])

    # unpack data and merge into single dict
    return {id:val for array_tensor in output_tensors for id,val in _unpack_data(array_tensor).items()}


if __name__ == "__main__":
    import torchvision
    device = None

    # options for PyTorch DDP
    use_distributed_data_parallel = False
    world_size = 1
    world_rank = 1

    dataset_batch = 16
    dataset_hard_sample_size = 8
    hard_samples_selected_min_percent = 0.1

    dataset_shuffle = True

    train_dataset = torchvision.datasets.ImageFolder(root='/path/to/imgs')

    # prepare hard-examples sampler for dataset
    if dataset_shuffle:
        if use_distributed_data_parallel:
            default_sampler = DistributedRandomSampler(train_dataset, device=device)
        else:
            default_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        default_sampler = torch.utils.data.SequentialSampler(train_dataset)

    batch_sampler = HardExamplesBatchSampler(train_dataset,
                                             default_sampler,
                                             batch_size=dataset_batch,
                                             hard_sample_size=dataset_hard_sample_size,
                                             drop_last=True,
                                             hard_samples_selected_min_percent=hard_samples_selected_min_percent,
                                             device=device, world_size=world_size, rank=world_rank,
                                             is_distributed=use_distributed_data_parallel)

    train_dataset_it = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)

    for sample in train_dataset_it:

        # get loss for samples
        # loss = model(sample)
        loss = [0] * dataset_batch

        # extract difficulty score based on loss or any other metric
        difficulty_score = np.zeros(dataset_batch)
        for n in range(dataset_batch):
            difficulty_score[n] = loss[n]

        # provide difficulty scores for samples:
        #  'index_key' is used as unique identifier of each sample
        #  'storage_keys' are any additional values that should be stored together with difficulty_score for that unique sample
        batch_sampler.update_sample_loss_batch(sample, difficulty_score,
                                               index_key='index')
