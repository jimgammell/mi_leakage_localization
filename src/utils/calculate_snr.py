from typing import *
from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset

from utils.chunk_iterator import chunk_iterator

def calculate_snr(dataset: Dataset, targets: Union[str, Sequence[str]] = 'subbytes', chunk_size: int = 1024):
    base_dataset = dataset
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(targets, str):
        targets = [targets]
    
    orig_transform = copy(base_dataset.transform)
    orig_target_transform = copy(base_dataset.target_transform)
    orig_ret_mdata = base_dataset.return_metadata
    base_dataset.transform = None
    base_dataset.target_transform = None
    base_dataset.return_metadata = True
    
    per_target_means = {key: np.zeros((256, base_dataset.trace_length), dtype=np.float32) for key in targets}
    per_target_counts = {key: np.zeros((256,), dtype=int) for key in targets}
    for trace, _, metadata in chunk_iterator(dataset, chunk_size=chunk_size):
        for target in targets:
            target_val = metadata[target]
            current_mean = per_target_means[target][target_val]
            current_count = per_target_counts[target][target_val]
            per_target_means[target][target_val] = (current_count/(current_count+1))*current_mean + (1/(current_count+1))*trace
            per_target_counts[target][target_val] += 1
    noise_variance = {key: np.zeros((base_dataset.trace_length,), dtype=np.float32) for key in targets}
    for count, (trace, _, metadata) in enumerate(chunk_iterator(dataset, chunk_size=chunk_size)):
        for target in targets:
            target_val = metadata[target]
            mean = per_target_means[target][target_val]
            current_var = noise_variance[target]
            noise_variance[target] = (count/(count+1))*current_var + (1/(count+1))*(trace - mean)**2
    signal_variance = {key: np.var(val, axis=0) for key, val in per_target_means.items()}
    snr_vals = {key: signal_variance[key]/noise_variance[key] for key in signal_variance.keys()}
    
    base_dataset.transform = orig_transform
    base_dataset.target_transform = orig_target_transform
    base_dataset.return_metadata = orig_ret_mdata
    return snr_vals