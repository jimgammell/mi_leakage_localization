from typing import *
from copy import copy
import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils.chunk_iterator import chunk_iterator

def hamming_weight(x):
    return np.unpackbits(x).astype(np.float32).sum()

def calculate_cpa(dataset: Dataset, base_dataset: Dataset, targets: Union[str, Sequence[str]] = 'subbytes', bytes=None, chunk_size: int = 1024):
    if isinstance(targets, str):
        targets = [targets]
    if (bytes is None) or isinstance(bytes, int):
        bytes = [bytes]
    mean_trace = np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32)
    mean_hamming_weight = {(key, byte): 0. for key in targets for byte in bytes}
    current_count = 0
    for trace, _, metadata in chunk_iterator(dataset, chunk_size=chunk_size):
        for target in targets:
            for byte in bytes:
                target_val = metadata[target]
                if (byte is not None) and (target_val.size > 1):
                    target_val = target_val[byte]
                mean_trace = (current_count/(current_count+1))*mean_trace + (1/(current_count+1))*trace
                mean_hamming_weight[(target, byte)] = (
                    (current_count/(current_count+1))*mean_hamming_weight[(target, byte)]
                    + (1/(current_count+1))*hamming_weight(target_val)
                )
        current_count += 1
    unnormalized_correlation = {(key, byte): np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32) for key in targets for byte in bytes}
    trace_variance = np.zeros((base_dataset.timesteps_per_trace,), dtype=np.float32)
    hw_variance = {(key, byte): 0. for key in targets for byte in bytes}
    current_count = 0
    for trace, _, metadata in chunk_iterator(dataset, chunk_size=chunk_size):
        for target in targets:
            for byte in bytes:
                target_val = metadata[target]
                if (byte is not None) and (target_val.size > 1):
                    target_val = target_val[byte]
                unnormalized_correlation[(target, byte)] = (
                    (current_count/(current_count+1))*unnormalized_correlation[(target, byte)]
                    + (1/(current_count+1))*(trace - mean_trace)*(hamming_weight(target_val)-mean_hamming_weight[(target, byte)])
                )
                trace_variance = (current_count/(current_count+1))*trace_variance + (1/(current_count+1))*(trace - mean_trace)**2
                hw_variance[(target, byte)] = (
                    (current_count/(current_count+1))*hw_variance[(target, byte)]
                    + (1/(current_count+1))*(hamming_weight(target_val)-mean_hamming_weight[(target, byte)])**2
                )
        current_count += 1
    cpa = {(key, byte): unnormalized_correlation[(key, byte)] / np.sqrt(trace_variance * hw_variance[(key, byte)]) for key in targets for byte in bytes}
    return cpa