from typing import *
from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset

from utils.chunk_iterator import chunk_iterator

def calculate_snr(dataset: Dataset, base_dataset: Dataset, targets: Union[str, Sequence[str]] = 'subbytes', bytes=None, chunk_size: int = 1024):
    if isinstance(targets, str):
        targets = [targets]
    if (bytes is None) or isinstance(bytes, int):
        bytes = [bytes]
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
    noise_variance = {(key, byte): np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32) for key in targets for byte in bytes}
    for count, (trace, _, metadata) in enumerate(chunk_iterator(dataset, chunk_size=chunk_size)):
        for target in targets:
            for byte in bytes:
                target_val = metadata[target]
                if (byte is not None) and (target_val.size > 1):
                    target_val = target_val[byte]
                mean = per_target_means[(target, byte)][target_val]
                current_var = noise_variance[(target, byte)]
                noise_variance[(target, byte)] = (count/(count+1))*current_var + (1/(count+1))*(trace - mean)**2
    signal_variance = {key: np.var(val, axis=0) for key, val in per_target_means.items()}
    snr_vals = {key: signal_variance[key]/noise_variance[key] for key in signal_variance.keys()}
    base_dataset.return_metadata = False
    return snr_vals