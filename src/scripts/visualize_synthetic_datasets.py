import numpy as np
from matplotlib import pyplot as plt

from _common import *
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from utils.calculate_snr import calculate_snr
import models

dataset = SyntheticAES(epoch_length=10000, lpf_beta=0.9, leaking_timestep_count_1o=1)
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
    ax.axvline(cycle, color='red')
ax.plot(snr, color='blue', linestyle='none', marker='.', markersize=2)
fig.savefig(os.path.join(get_trial_dir(), 'snr.png'))

dataset = SyntheticAES(epoch_length=10000, lpf_beta=0.9, leaking_timestep_count_1o=0, leaking_timestep_count_2o=1)
snr = calculate_snr(dataset, targets=['subbytes', 'mask', 'masked_subbytes'])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for cycle in dataset.leaking_subbytes_cycles:
    axes[0].axvline(cycle, color='red')
for cycle in dataset.leaking_mask_cycles:
    axes[1].axvline(cycle, color='red')
for cycle in dataset.leaking_masked_subbytes_cycles:
    axes[2].axvline(cycle, color='red')
axes[0].plot(snr['subbytes'], color='blue', linestyle='none', marker='.', markersize=2)
axes[1].plot(snr['mask'], color='blue', linestyle='none', marker='.', markersize=2)
axes[2].plot(snr['masked_subbytes'], color='blue', linestyle='none', marker='.', markersize=2)
fig.savefig(os.path.join(get_trial_dir(), '2o_snr.png'))