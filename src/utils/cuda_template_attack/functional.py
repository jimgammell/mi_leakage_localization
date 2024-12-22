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
def get_class_count(labels):
    return len(labels.unique())

@torch.compile()
@torch.no_grad()
def fit_means_and_covs(traces, labels, class_count):
    partition_count, datapoint_count, points_per_partition = traces.shape
    means = torch.empty((partition_count, class_count, points_per_partition), dtype=traces.dtype, device=traces.device)
    for label in range(class_count):
        means[:, label, :] = traces[:, labels==label, :].mean(dim=1)
    covs = torch.full((partition_count, class_count, points_per_partition, points_per_partition), torch.nan, dtype=traces.dtype, device=traces.device)
    for label in range(class_count):
        mean = means[:, label, :].unsqueeze(1)
        trace = traces[:, labels==label, :]
        trace_count = trace.size(1)
        diff = (trace - mean)
        cov = diff.mT @ diff / (trace_count - 1)
        covs[:, label, ...] = cov
    pooled_cov = covs.mean(dim=1)
    for label in range(class_count):
        cov = covs[:, label, ...]
        cov = 0.5*cov + 0.5*pooled_cov
        cov = 0.5*(cov + cov.mT)
        D, U = torch.linalg.eigh(cov)
        assert torch.all(D > 0)
        cov = U @ torch.diag_embed(D) @ U.mT
        covs[:, label, ...] = cov
    assert torch.all(torch.isfinite(covs))
    return means, covs

@torch.compile()
@torch.no_grad()
def get_choldecomp_covs(covs):
    L, error = torch.linalg.cholesky_ex(covs)
    assert torch.all(error == 0)
    return L

@torch.compile()
@torch.no_grad()
def compute_log_gaussian_density(x, mu, L):
    y = torch.linalg.solve(L, (x-mu).unsqueeze(-1)).squeeze(-1)
    logdets = 2*(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-4).log().sum(dim=-1)
    return -0.5*(y*y).sum(dim=-1) - 0.5*logdets

@torch.compile()
@torch.no_grad()
def get_log_p_y(labels):
    _, counts = torch.unique(labels, sorted=True, return_counts=True)
    return counts.log() - counts.sum().log()

@torch.no_grad()
def get_log_p_x_given_y(traces, means, Ls):
    _, datapoint_count, _ = traces.shape
    _, class_count, _ = means.shape
    traces = traces.unsqueeze(1).repeat(1, class_count, 1, 1)
    means = means.unsqueeze(2).repeat(1, 1, datapoint_count, 1)
    Ls = Ls.unsqueeze(2).repeat(1, 1, datapoint_count, 1, 1)
    log_gaussian_densities = compute_log_gaussian_density(traces, means, Ls)
    return log_gaussian_densities