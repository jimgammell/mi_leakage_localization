import numpy as np
import numba
from numba import jit
import torch
from torch.utils.data import Dataset

from ..common import *

@jit(nopython=True)
def apply_ema(trace, ema_coeff):
    for time_idx in range(1, trace.shape[1]):
        trace[:, time_idx] = ema_coeff*trace[:, time_idx-1] + (1-ema_coeff)*trace[:, time_idx]
    return trace

@jit(nopython=True)
def get_hamming_weight(number):
    hamming_weight = 0
    while number != 0:
        hamming_weight += number & 1
        number >>= 1
    return hamming_weight

@jit(nopython=True, parallel=True)
def generate_trace(data_vals, data_locs, cycles_per_trace, timesteps_per_cycle, fixed_noise, random_noise, random_integers, data_var, leakage_model):
    trace = np.empty((cycles_per_trace*timesteps_per_cycle,), dtype=numba.float32)
    prev_data_val = random_integers[0]
    for cycle in numba.prange(cycles_per_trace):
        data_val = None
        for i in range(len(data_vals)):
            if cycle == data_locs[i]:
                data_val = data_vals[i]
        if data_val is None:
            data_val = random_integers[cycle]
        if leakage_model == 'hamming_weight':
            data_power = (get_hamming_weight(data_val).astype(np.float32) - 4) / np.sqrt(2)
        elif leakage_model == 'hamming_distance':
            data_power = (get_hamming_weight(data_val ^ prev_data_val).astype(np.float32) - 4) / np.sqrt(2)
        elif leakage_model == 'identity':
            data_power = (data_val.astype(np.float32) - 127.5) / np.sqrt((256**2 - 1)/12)
        else:
            assert False
        data_power = np.float32(data_power)
        data_power = np.sqrt(data_var) * data_power / np.sqrt(2)
        for timestep in range(timesteps_per_cycle):
            trace[cycle*timesteps_per_cycle+timestep] = (
                (data_power if timestep == 0 else 0)
                + fixed_noise[cycle*timesteps_per_cycle + timestep]
                + random_noise[cycle*timesteps_per_cycle + timestep]
            )
        prev_data_val = data_val
    return trace