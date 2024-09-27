from typing import *
import numpy as np
from numba import jit
import torch
from torch.utils.data import Subset, Dataset

from utils.chunk_iterator import chunk_iterator

@torch.no_grad()
def extract_dataset(
    dataset: Dataset, points_of_interest: Sequence[int],
    metadata_keys: Union[str, Sequence[str]] = 'subbytes',
    target_byte: Optional[int] = None
):
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    orig_transform = base_dataset.transform
    orig_ret_mdata = base_dataset.return_metadata
    base_dataset.transform = None
    base_dataset.return_metadata = True
    datapoint_count = len(dataset)
    poi_count = len(points_of_interest)
    traces = np.full((datapoint_count, poi_count), np.nan, dtype=np.float32)
    if isinstance(metadata_keys, str):
        metadata_keys = [metadata_keys]
    metadata = {key: np.zeros((datapoint_count,), dtype=np.uint8) for key in metadata_keys}
    for datapoint_idx, (trace, _, _metadata) in enumerate(chunk_iterator(dataset)):
        for key in metadata_keys:
            mval = _metadata[key]
            if target_byte is not None:
                mval = mval[target_byte]
            metadata[key][datapoint_idx] = mval
        trace = trace.squeeze()
        trace = trace[points_of_interest]
        traces[datapoint_idx, :] = trace
    base_dataset.transform = orig_transform
    base_dataset.return_metadata = orig_ret_mdata
    assert all(np.isfinite(traces))
    return traces, metadata

@jit(nopython=True)
def mean_with_axis(array, axis):
    return np.sum(array, axis=axis) / array.shape[axis]

@jit(nopython=True)
def fit_means(traces, targets):
    (datapoint_count, poi_count) = traces.shape
    means = np.full((256, poi_count), np.nan, dtype=np.float32)
    for byte in range(256):
        means[byte, :] = mean_with_axis(traces[targets == byte, :], 0)
    assert np.all(np.isfinite(means))
    return means

@jit(nopython=True)
def fit_covs(traces, targets, means):
    (datapoint_count, poi_count) = traces.shape
    covs = np.full((256, poi_count, poi_count), np.nan, dtype=np.float32)
    for byte in range(256):
        mean = means[byte]
        traces_byte = traces[targets == byte, :]
        trace_count, _ = traces_byte.shape
        diff = traces_byte - mean
        cov = diff.T @ diff / (trace_count - 1)
        cov = 0.5*(cov + cov.T) # ensure it is symmetric
        D, U = np.linalg.eigh(cov)
        D[D <= 1e-8] = 1e-8 # ensure it is positive semi-definite
        cov = U @ np.diag(D) @ U.T
        covs[byte, ...] = cov
    assert np.all(np.isfinite(covs))
    return covs

@jit(nopython=True)
def choldecomp_covs(covs):
    decomps = np.full_like(covs, np.nan)
    for cov_idx, cov in enumerate(covs):
        L = np.linalg.cholesky(cov)
        decomps[cov_idx, ...] = L
    assert np.all(np.isfinite(decomps))
    return decomps

@jit(nopython=True)
def compute_log_gaussian_density(x, mu, L):
    y = np.linalg.solve(L, x - mu)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * np.dot(y, y) - 0.5*logdet

@jit(nopython=True)
def get_log_p_y(targets):
    probs = np.zeros((256,), dtype=np.float32)
    for target in targets:
        probs[target] += 1
    log_probs = np.log(probs) - np.log(probs.sum())
    return log_probs

@jit(nopython=True)
def get_log_p_x_given_y(traces, means, Ls):
    datapoint_count = traces.shape[0]
    log_probs = np.full((datapoint_count, 256), np.nan, dtype=np.float32)
    for datapoint_idx in range(datapoint_count):
        trace = traces[datapoint_idx, :]
        for byte in range(256):
            log_probs[datapoint_idx, byte] = compute_log_gaussian_density(trace, means[byte], Ls[byte])
    assert np.all(np.isfinite(log_probs))
    return log_probs