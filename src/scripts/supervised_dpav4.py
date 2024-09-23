import os
import numpy as np
from matplotlib import pyplot as plt

from _common import *
from datasets.dpav4 import DPAv4
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4')
profiling_dataset = DPAv4(
    root=ROOT, train=True
)
attack_dataset = DPAv4(
    root=ROOT, train=False
)

cpa = calculate_cpa(profiling_dataset, targets=['subbytes'])
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(cpa[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
ax.set_xlabel('Timestep')
ax.set_ylabel('CPA')
ax.set_title('$Y$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'cpa.pdf'))

snr = calculate_snr(profiling_dataset, targets=['subbytes'])
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(snr[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
ax.set_xlabel('Timestep')
ax.set_ylabel('SNR')
ax.set_title('$Y$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'))

sosd = calculate_sosd(profiling_dataset, targets=['subbytes'])
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(sosd[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
ax.set_xlabel('Timestep')
ax.set_ylabel('SOSD')
ax.set_title('$Y$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'sosd.pdf'))