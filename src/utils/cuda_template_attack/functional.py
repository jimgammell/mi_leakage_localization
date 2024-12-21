from typing import *
import os
import numpy as np
import torch
from torch.utils.data import Subset, Dataset

from utils.chunk_iterator import chunk_iterator

@torch.no_grad()
def extract_dataset(
    dataset: Dataset, partitions
):
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    orig_transform = base_dataset.transform
    orig_target_transform = base_dataset.target_transform
    base_dataset.transform = base_dataset.target_transform = None
    datapoint_count = len(dataset)
    partition_count, points_per_partition = partitions.shape
    partitions = partitions.reshape(-1)
    traces = np.full((partition_count, datapoint_count, points_per_partition), np.nan, dtype=np.float32)
    labels = np.zeros((datapoint_count,), dtype=np.uint8)
    for datapoint_idx, (trace, label) in enumerate(chunk_iterator(dataset)):
        trace = trace.squeeze()
        traces[:, datapoint_idx, :] = trace[partitions].reshape(partition_count, points_per_partition)
        labels[datapoint_idx] = label
    base_dataset.transform = orig_transform
    base_dataset.target_transform = orig_target_transform
    assert np.all(np.isfinite(traces))
    traces -= traces.mean(axis=1, keepdims=True)
    traces /= traces.std(axis=1, keepdims=True) + 1e-4
    return traces, labels

@torch.compile()
@torch.no_grad()
def fit_means_and_covs(traces, labels):
    class_count = len(labels.unique())
    partition_count, datapoint_count, points_per_partition = traces.shape
    means = torch.empty((class_count, partition_count, points_per_partition), dtype=traces.dtype, device=traces.device)
    for label in range(class_count):
        means[label, ...] = traces[:, labels==label, :].mean(dim=1)
    covs = torch.empty((class_count, partition_count, points_per_partition, points_per_partition), dtype=traces.dtype, device=traces.device)
    for label in range(class_count):
        mean = means[label, ...].unsqueeze(1)
        trace = traces[:, labels==label, :]
        trace_count = trace.size(1)
        diff = trace - mean
        cov = diff.permute(0, 2, 1) @ diff / (trace_count - 1)
        cov = 0.5*(cov + cov.mT)
        D, U = torch.linalg.eigh(cov)
        D[D < 0] = 0
        cov = U @ torch.diag(D) @ U.mT
        covs[label] = cov
    return means, covs

@torch.compile()
@torch.no_grad()
def choldecomp_covs(covs):
    class_count, partition_count, points_per_partition, points_per_partition = covs.shape
    covs = covs.reshape(-1, points_per_partition, points_per_partition)
    L, error = torch.linalg.cholesky_ex(covs)
    assert torch.all(error == 0)
    L = L.reshape(class_count, partition_count, points_per_partition, points_per_partition)
    return L