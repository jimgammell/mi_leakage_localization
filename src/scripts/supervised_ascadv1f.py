import os
import numpy as np
from matplotlib import pyplot as plt

from _common import *
from datasets.ascadv1 import ASCADv1
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization','downloads', 'ascadv1')

profiling_dataset = ASCADv1(
    root=ROOT,
    train=True
)
attack_dataset = ASCADv1(
    root=ROOT,
    train=False
)

cpa = calculate_cpa(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
axes[0].plot(cpa[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[1].plot(cpa[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[2].plot(cpa[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[3].plot(cpa[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[4].plot(cpa[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
for ax in axes:
    ax.set_xlabel('Timestep')
    ax.set_ylabel('CPA')
axes[0].set_title('$Y$')
axes[1].set_title('$Y \oplus R$')
axes[2].set_title('$R$')
axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
axes[4].set_title('$R_{\mathrm{out}}$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'cpa.pdf'))

snr = calculate_snr(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
axes[0].plot(snr[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[1].plot(snr[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[2].plot(snr[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[3].plot(snr[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[4].plot(snr[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
for ax in axes:
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SNR')
axes[0].set_title('$Y$')
axes[1].set_title('$Y \oplus R$')
axes[2].set_title('$R$')
axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
axes[4].set_title('$R_{\mathrm{out}}$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'))

sosd = calculate_sosd(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
axes[0].plot(sosd[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[1].plot(sosd[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[2].plot(sosd[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[3].plot(sosd[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
axes[4].plot(sosd[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
for ax in axes:
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SOSD')
axes[0].set_title('$Y$')
axes[1].set_title('$Y \oplus R$')
axes[2].set_title('$R$')
axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
axes[4].set_title('$R_{\mathrm{out}}$')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'sosd.pdf'))