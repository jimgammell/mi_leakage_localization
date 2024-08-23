import numpy as np
from numba import jit
from torch.utils.data import Dataset

@jit(nopython=False)
def chunk_iterator(dataset: Dataset, chunk_size: int = 1024):
    chunk_count = int(np.ceil(len(dataset)/chunk_size))
    for chunk_idx in range(chunk_count):
        min_idx = chunk_idx*chunk_size
        max_idx = min(len(dataset), (chunk_idx+1)*chunk_size)
        return_vals = dataset[np.arange(min_idx, max_idx)]
        for datapoint_idx in range(len(return_vals[0])):
            return tuple([
                {key: val[datapoint_idx, ...] for key, val in x.items()} if isinstance(x, dict) else x[datapoint_idx, ...]
                for x in return_vals
            ])