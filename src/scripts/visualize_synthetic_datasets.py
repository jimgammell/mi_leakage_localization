import numpy as np
from matplotlib import pyplot as plt

from _common import *
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike

dataset = SyntheticAES(length=1000)
print(dataset)

eg_trace, eg_target = dataset[0]
print(f'Trace shape: {eg_trace.shape}')
print(f'Target shape: {eg_target.shape}')

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for trace, _ in dataset:
    ax.plot(trace, color='blue', linestyle='none', marker='.', markersize=0.01)
fig.savefig(os.path.join(get_trial_dir(), 'traces.png'))