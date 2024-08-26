import numpy as np
from matplotlib import pyplot as plt

from _common import *
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from utils.calculate_snr import calculate_snr
import models

dataset = SyntheticAES(length=10000, lpf_beta=0.9, num_leaking_subbytes_cycles=10)
print(dataset)

eg_trace, eg_target = dataset[0]
print(f'Trace shape: {eg_trace.shape}')
print(f'Target shape: {eg_target.shape}')

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for trace, _ in dataset:
    ax.plot(trace, color='blue', linestyle='none', marker='.', markersize=0.01)
fig.savefig(os.path.join(get_trial_dir(), 'traces.png'))

snr = calculate_snr(dataset)['subbytes']
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for cycle in dataset.leaking_subbytes_cycles:
    ax.axvline(cycle*dataset.timesteps_per_cycle, color='red')
ax.plot(snr, color='blue', linestyle='none', marker='.', markersize=2)
fig.savefig(os.path.join(get_trial_dir(), 'snr.png'))

model = models.load('sca-cnn')
print(model)
eg_input = torch.as_tensor(eg_trace[np.newaxis, np.newaxis, :])
eg_output = model(eg_input)
print(f'Model: {eg_input.shape} -> {eg_output.shape}')