from typing import *
import numpy as np
import numba
import torch
from torch.utils.data import Dataset

@numba.jit(nopython=True)
def _get_label_bits(count, bit_count):
    return np.random.choice(2, (count, bit_count), replace=True).astype(np.uint8)

@numba.jit(nopython=True)
def _get_labels(label_bits, bit_count):
    basis = np.array([2**x for x in range(bit_count)], dtype=np.uint32).reshape(1, -1)
    return (label_bits.astype(np.uint32)*basis).sum(axis=-1)

@numba.jit(nopython=True)
def _get_noise_component(count, dim, sigma):
    return sigma*np.random.randn(count, dim).astype(np.float32)

@numba.jit(nopython=True)
def _get_data_dependent_component(labels, bit_count):
    return (12.*(labels.astype(np.float32) - 0.5*(2**bit_count-1))/((2**bit_count-1)**2)).reshape(-1, 1)

@numba.jit(nopython=True)
def _get_nth_order_leaky_point(label_bits, leakage_dim, point_count, bit_count, sigma):
    assert leakage_dim >= 0
    assert point_count > 0
    label_bits = label_bits.copy()
    labels = _get_labels(label_bits, bit_count)
    if leakage_dim >= 1:
        masks = [_get_label_bits(len(label_bits), bit_count) for _ in range(leakage_dim-1)]
        for mask in masks:
            label_bits = label_bits ^ mask
        data = np.full((len(label_bits), leakage_dim*point_count), np.nan, dtype=np.float32)
        data[:, 0] = _get_labels(label_bits, bit_count)
        for idx, mask in enumerate(masks):
            data[:, idx+1] = _get_labels(mask, bit_count)
    else:
        pass
    point = _get_noise_component(len(labels), max(1, leakage_dim)*point_count, sigma)
    for idx in range(leakage_dim):
        point[:, idx*point_count:(idx+1)*point_count] += _get_data_dependent_component(data[:, idx], bit_count)
    return point

#@numba.jit(nopython=True)
def _generate_data(count, point_counts, bit_count, sigma):
    label_bits = _get_label_bits(count, bit_count)
    labels = _get_labels(label_bits, bit_count)
    data = []
    for leakage_dim, point_count in enumerate(point_counts):
        if point_count > 0:
            data.append(_get_nth_order_leaky_point(label_bits, leakage_dim, point_count, bit_count, sigma))
    data = np.concatenate(data, axis=-1)
    return data, labels

class SimpleGaussianDataset(Dataset):
    def __init__(self,
        bit_count: int = 1,
        point_counts: Sequence[int] = None, # Number of leaky points with each 'order' of leakage. Index 0 denotes non-leaky points, index 1 denotes first-order leakage, index 2 denotes 2nd-order, etc.
        dataset_size: int = 10000,
        infinite_dataset: bool = False, # If true, the dataset will 'look' like it has the size given above, but all outputs will be randomly generated.
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True,
        sigma: float = 1.0
    ):
        super().__init__()
        train = None # since every dataset is randomly-generated, there is no need to specify train vs. test
        self.point_counts = [1, 1] if point_counts is None else point_counts
        self.bit_count = bit_count
        self.point_counts = point_counts
        self.dataset_size = dataset_size
        self.infinite_dataset = infinite_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma
        self.timesteps_per_trace = sum(point_count*min(1, leakage_dim) for point_count, leakage_dim in enumerate(point_counts))
        assert self.timesteps_per_trace > 0
        if self.infinite_dataset:
            self.used_points = 0
        self.traces, self.labels = self.generate_data(self.dataset_size)
        self.return_metadata = False
        
    def get_label_bits(self, count):
        return _get_label_bits(count, self.bit_count)
    def get_labels(self, label_bits):
        return _get_labels(label_bits, self.bit_count)
    def get_noise_component(self, count, dim=1):
        return _get_noise_component(count, dim, self.sigma)
    def get_data_dependent_component(self, labels):
        return _get_data_dependent_component(labels, self.bit_count)
    def get_nth_order_leaky_point(self, label_bits, n=1):
        return _get_nth_order_leaky_point(label_bits, n, self.bit_count, self.sigma)
    def generate_data(self, count):
        traces, labels = _generate_data(count, self.point_counts, self.bit_count, self.sigma)
        traces = traces[:, np.newaxis, :]
        return traces, labels
    
    def __getitem__(self, idx):
        if self.infinite_dataset:
            if self.used_points >= self.dataset_size:
                self.traces, self.labels = self.generate_data(self.dataset_size)
                self.used_points = 0
            self.used_points += 1
        trace, label = self.traces[idx], self.labels[idx]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_metadata:
            return trace, label, {'label': label}
        else:
            return trace, label
    
    def __len__(self):
        return self.dataset_size