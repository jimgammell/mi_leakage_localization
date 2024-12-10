from typing import *
import numpy as np
import numba
import torch
from torch.utils.data import Dataset

class SimpleGaussianDataset(Dataset):
    def __init__(self,
        label_bits: int = 1,
        point_counts: Sequence[int] = None, # Number of leaky points with each 'order' of leakage. Index 0 denotes non-leaky points, index 1 denotes first-order leakage, index 2 denotes 2nd-order, etc.
        dataset_size: int = 10000,
        infinite_dataset: bool = False, # If true, the dataset will 'look' like it has the size given above, but all outputs will be randomly generated.
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True
    ):
        super().__init__()
        train = None # since every dataset is randomly-generated, there is no need to specify train vs. test
        self.point_counts = [1, 1] if point_counts is None else point_counts
        self.label_bits = label_bits
        self.point_counts = point_counts
        self.dataset_size = dataset_size
        self.infinite_dataset = infinite_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.timesteps_per_trace = sum(self.point_counts)
        assert self.timesteps_per_trace > 0
        if self.infinite_dataset:
            self.used_points = 0
        self.traces, self.labels = self.generate_data(self.dataset_size)
        self.return_metadata = False
        
    @numba.jit(nopython=True)
    def get_label_bits(self, count):
        return np.random.choice([0, 1], (count, self.label_bits), replace=True, dtype=np.uint8)
    
    @numba.jit(nopython=True)
    def get_labels(self, label_bits):
        basis = np.array([2**x for x in range(self.label_bits)], dtype=np.uint32).reshape(1, -1)
        return (label_bits.astype(np.uint32)*basis).sum(axis=-1)
        
    @numba.jit(nopython=True)
    def get_noise_component(self, labels, dim=1):
        count = len(labels)
        return np.random.randn(count, dim).astype(np.float32)
    
    @numba.jit(nopython=True)
    def get_data_dependent_component(self, labels):
        return (12.*(labels.astype(np.float32) - 0.5*(2**self.label_bits-1))/(2**self.label_bits-1)**2)
    
    @numba.jit(nopython=True)
    def get_nth_order_leaky_point(self, label_bits, n=1):
        assert n >= 0
        label_bits = label_bits.copy()
        labels = self.get_labels(label_bits)
        if n >= 1:
            masks = [self.get_label_bits(len(label_bits)) for _ in range(n-1)]
            for mask in masks:
                label_bits = label_bits ^ mask
            data = np.full((len(label_bits), n), np.nan, dtype=np.float32)
            data[:, 0] = self.get_labels(label_bits)
            for idx, mask in enumerate(masks):
                data[:, idx] = self.get_labels(mask)
        else:
            pass
        point = self.get_noise_component(labels, dim=n)
        for idx in range(n):
            point[:, idx] += self.get_data_dependent_component(data[:, idx])
        return point
    
    @numba.jit(nopython=True)
    def generate_data(self, count):
        label_bits = self.get_label_bits(count)
        labels = self.get_labels(label_bits)
        data = []
        for n, count in enumerate(self.point_counts):
            if count > 0:
                data.append(self.get_nth_order_leaky_point(label_bits, n))
        points = np.concatenate(data, axis=-1)
        return points, labels
    
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