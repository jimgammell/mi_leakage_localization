from typing import *
from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset

from utils.chunk_iterator import chunk_iterator

def calculate_sosd(dataset: Dataset, targets: Union[str, Sequence[str]] = 'subbytes', bytes=None, chunk_size: int = 1024):
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    if isinstance(targets, str):
        targets = [targets]
    if (bytes is None) or isinstance(bytes, int):
        bytes = [bytes]
    
    orig_transform = copy(base_dataset.transform)
    orig_target_transform = copy(base_dataset.target_transform)
    orig_ret_mdata = base_dataset.return_metadata
    base_dataset.transform = None
    base_dataset.target_transform = None
    base_dataset.return_metadata = True
    
    per_target_means = {(key, byte): np.zeros((256, base_dataset.timesteps_per_trace), dtype=np.float32) for key in targets for byte in bytes}
    per_target_counts = {(key, byte): np.zeros((256,), dtype=int) for key in targets for byte in bytes}
    for trace, _, metadata in chunk_iterator(dataset, chunk_size=chunk_size):
        for target in targets:
            for byte in bytes:
                target_val = metadata[target]
                if (byte is not None) and (target_val.size > 1):
                    target_val = target_val[byte]
                current_mean = per_target_means[(target, byte)][target_val]
                current_count = per_target_counts[(target, byte)][target_val]
                per_target_means[(target, byte)][target_val] = (current_count/(current_count+1))*current_mean + (1/(current_count+1))*trace
                per_target_counts[(target, byte)][target_val] += 1
    sosd = {(key, byte): np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32) for key in targets for byte in bytes}
    for target in targets:
        for byte in bytes:
            for i in range(256):
                for j in range(i+1, 256):
                    sosd[(target, byte)] += (per_target_means[(target, byte)][i, :] - per_target_means[(target, byte)][j, :])**2
    
    base_dataset.transform = orig_transform
    base_dataset.target_transform = orig_target_transform
    base_dataset.return_metadata = orig_ret_mdata
    return sosd