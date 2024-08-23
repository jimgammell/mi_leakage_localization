from typing import *
import numpy as np
from torch.utils.data import Dataset

from .data_generation import *
from ..common import *

# To do:
#  - Set up the fixed noise profile to reflect no-ops

LPF_BURN_IN_CYCLES = 10

class SyntheticAES(Dataset):
    def __init__(self,
        length: int = 10000, # number of datapoints in the dataset.
        infinite_dataset: bool = False, # Whether or not to generate a new random datapoint for every __getitem__ call. If True, length will determine epoch length.
        cycles_per_trace: int = 250, # Number of clock cycles per power trace.
        timesteps_per_cycle: int = 4, # Number of timesteps per clock cycle.
        fixed_noise_var: float = 1.0, # Variance of the normal distribution from which the fixed noise profile will be drawn.
        random_noise_var: float = 0.5, # Variance of the normal distribution from which the datapoint-dependent component of noise will be drawn.
        data_var: float = 1.0, # Variance due to the hamming weight of data at each clock cycle.
        num_leaking_subbytes_cycles: int = 1, # Number of cycles at which the subbytes variable leaks.
        num_leaking_mask_cycles: int = 0, # Number of cycles at which the Boolean mask leaks.
        num_leaking_masked_subbytes_cycles: int = 0, # Number of cycles at which the masked subbytes variable leaks.
        shuffle_locations: int = 1, # The cycle at which leakage happens will be randomly chosen from this number of possibilities.
        max_no_ops: int = 0, # Leakage will be randomly delayed by between 0 and this number of cycles.
        lpf_beta: float = 0.0, # Traces will be discrete low-pass filtered according to x_lpf[t+1] = lpf_beta*x_lpf[t] + (1-lpf_beta)x[t+1].
        leakage_model: str = 'hamming_weight', # This function of the sensitive variable will be leaked. Options: ['identity', 'hamming_weight', 'hamming_distance']
        target_values: Union[str, Sequence[str]] = 'subbytes', # The sensitive variable to be targeted. Options: ['subbytes', 'mask', 'masked_subbytes']
        fixed_key: Optional[np.uint8] = None, # Instead of randomly sampling keys, fix the key to this value.
        fixed_mask: Optional[np.uint8] = None, # Instead of randomly sampling mask values, fix the mask to this value. Only relevant if leaking_cycles_2o > 0.
        fixed_plaintext: Optional[np.uint8] = None, # Instead of randomly sampling plaintexts, fix the plaintext to this value.
        transform: Optional[Callable] = None, # Traces will be transformed by this function before being returned.
        target_transform: Optional[Callable] = None, # Target values will be transformed by this function before being returned.
        should_generate_data: bool = True, # we might not want to generate datapoints immediately -- e.g. if we are copying properties from another dataset
        return_metadata: bool = False # whether to return metadata -- i.e. key, plaintext, subbytes, regardless of whether they are used as a target
    ):
        for key, val in locals().items():
            if key != 'self':
                setattr(self, key, val)
        if isinstance(self.target_values, int):
            self.target_values = [self.target_values]
        if self.should_generate_data:
            self.generate_data()
    
    def generate_data(self):
        # randomly select cycles for leakage
        available_cycles = list(range(self.cycles_per_trace-self.max_no_ops))
        leaking_cycles = []
        for _ in range(self.shuffle_locations*(self.num_leaking_subbytes_cycles+self.num_leaking_mask_cycles+self.num_leaking_masked_subbytes_cycles)):
            cycle = NUMPY_RNG.choice(available_cycles)
            for x in range(cycle, cycle+self.max_no_ops):
                if x in available_cycles:
                    available_cycles.remove(x)
            leaking_cycles.append(cycle)
        leaking_cycles = np.array(leaking_cycles)
        self.leaking_subbytes_cycles = leaking_cycles[:self.shuffle_locations*self.num_leaking_subbytes_cycles]
        self.leaking_mask_cycles = leaking_cycles[self.shuffle_locations*self.num_leaking_subbytes_cycles:-self.shuffle_locations*self.num_leaking_masked_subbytes_cycles]
        self.leaking_masked_subbytes_cycles = leaking_cycles[-self.shuffle_locations*self.num_leaking_masked_subbytes_cycles:]
        
        self.trace_length = self.cycles_per_trace*self.timesteps_per_cycle
        self.fixed_profile = np.sqrt(self.fixed_noise_var)*NUMPY_RNG.standard_normal((self.trace_length + LPF_BURN_IN_CYCLES*self.timesteps_per_cycle,), dtype=np.float32)
        if not self.infinite_dataset:
            self.traces, self.metadata = self.generate_datapoints(self.length)
    
    def generate_datapoints(self, num_datapoints):
        if self.fixed_key is not None:
            keys = np.full((num_datapoints,), self.fixed_key, dtype=np.uint8)
        else:
            keys = NUMPY_RNG.integers(256, size=(num_datapoints,), dtype=np.uint8)
        if self.fixed_plaintext is not None:
            plaintexts = np.full((num_datapoints,), self.fixed_plaintext, dtype=np.uint8)
        else:
            plaintexts = NUMPY_RNG.integers(256, size=(num_datapoints,), dtype=np.uint8)
        subbytes = AES_SBOX[keys ^ plaintexts]
        if (self.num_leaking_mask_cycles > 0) or (self.num_leaking_masked_subbytes_cycles):
            assert (self.num_leaking_mask_cycles > 0) and (self.num_leaking_masked_subbytes_cycles > 0)
            if self.fixed_mask is not None:
                masks = np.full((num_datapoints,), self.fixed_mask, dtype=np.uint8)
            else:
                masks = NUMPY_RNG.integers(256, size=(num_datapoints,), dtype=np.uint8)
            masked_subbytes = masks ^ subbytes
        traces = np.empty((num_datapoints, self.trace_length), dtype=np.float32)
        for idx in range(num_datapoints):
            locs = np.array([], dtype=np.uint8)
            vals = np.array([], dtype=np.uint8)
            if self.num_leaking_subbytes_cycles > 0:
                locs = np.concatenate([locs, NUMPY_RNG.choice(self.leaking_subbytes_cycles, self.num_leaking_subbytes_cycles, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.num_leaking_subbytes_cycles, subbytes[idx], dtype=np.uint8)], axis=0)
            if self.num_leaking_mask_cycles > 0:
                locs = np.concatenate([locs, NUMPY_RNG.choice(self.leaking_mask_cycles, self.num_leaking_mask_cycles, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.num_leaking_mask_cycles, masks[idx], dtype=np.uint8)], axis=0)
            if self.num_leaking_masked_subbytes_cycles > 0:
                locs = np.concatenate([locs, NUMPY_RNG.choice(self.leaking_masked_subbytes_cycles, self.num_leaking_masked_subbytes_cycles, replace=False)], axis=0)
                vals = np.concatenate([vals, np.full(self.num_leaking_masked_subbytes_cycles, masked_subbytes[idx], dtype=np.uint8)], axis=0)
            locs += LPF_BURN_IN_CYCLES
            if self.max_no_ops > 0:
                no_ops = NUMPY_RNG.integers(self.max_no_ops+1, size=(len(locs),))
                locs += no_ops
            random_noise = np.sqrt(self.random_noise_var)*NUMPY_RNG.standard_normal(self.fixed_profile.shape)
            random_integers = NUMPY_RNG.integers(256, size=(self.cycles_per_trace+LPF_BURN_IN_CYCLES,), dtype=np.uint8)
            trace = generate_trace(
                vals, locs, self.cycles_per_trace+LPF_BURN_IN_CYCLES, self.timesteps_per_cycle, self.fixed_profile,
                random_noise, random_integers, self.data_var, self.leakage_model
            )
            if self.lpf_beta > 0:
                trace = apply_ema(trace, self.lpf_beta)
            trace = trace[:, :LPF_BURN_IN_CYCLES*self.timesteps_per_cycle]
            traces[idx, ...] = trace
        metadata = {'key': keys, 'plaintext': plaintexts, 'subbytes': subbytes}
        if self.leaking_mask_cycles > 0:
            metadata.update({'mask': masks})
        if self.leaking_masked_subbytes_cycles > 0:
            metadata.update({'masked_subbytes': masked_subbytes})
        return traces, metadata
    
    def __getitem__(self, idx):
        if self.infinite_dataset:
            trace, metadata = self.generate_datapoints(1 if isinstance(idx, int) else len(idx))
            target = np.stack([metadata[key] for key in self.target_values])
        else:
            trace = self.traces[idx, ...]
            target = np.stack([self.metadata[key][idx] for key in self.target_values])
            metadata = {key: val[idx] for key, val in self.metadata.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target
    
    def __len__(self):
        return self.length

class SyntheticAESLike(SyntheticAES):
    def __init__(self,
        base_dataset: SyntheticAES,
        length: int = 10000,
        infinite_dataset: bool = False,
        target_values: Union[str, Sequence[str]] = 'subbytes',
        fixed_key: Optional[np.uint8] = None,
        fixed_plaintext: Optional[np.uint8] = None,
        fixed_mask: Optional[np.uint8] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        super().__init__(
            length=length,
            infinite_dataset=infinite_dataset,
            cycles_per_trace=base_dataset.cycles_per_trace,
            timesteps_per_cycle=base_dataset.timesteps_per_cycle,
            fixed_noise_var=base_dataset.fixed_noise_var,
            random_noise_var=base_dataset.random_noise_var,
            data_var=base_dataset.data_var,
            num_leaking_subbytes_cycles=base_dataset.num_leaking_subbytes_cycles,
            num_leaking_mask_cycles=base_dataset.num_leaking_mask_cycles,
            num_leaking_masked_subbytes_cycles=base_dataset.num_leaking_masked_subbytes_cycles,
            shuffle_locations=base_dataset.shuffle_locations,
            max_no_ops=base_dataset.max_no_ops,
            lpf_beta=base_dataset.lpf_beta,
            leakage_model=base_dataset.leakage_model,
            target_values=base_dataset.target_values,
            fixed_key=fixed_key,
            fixed_plaintext=fixed_plaintext,
            fixed_mask=fixed_mask,
            transform=transform,
            target_transform=target_transform,
            should_generate_data=False,
            return_metadata=return_metadata
        )
        self.leaking_subbytes_cycles = base_dataset.leaking_subbytes_cycles
        self.leaking_mask_cycles = base_dataset.leaking_mask_cycles
        self.leaking_masked_subbytes_cycles = base_dataset.leaking_masked_subbytes_cycles
        self.trace_length = base_dataset.trace_length
        self.fixed_profile = base_dataset.fixed_profile
        if not self.infinite_dataset:
            self.traces, self.metadata = self.generate_datapoints(self.length)