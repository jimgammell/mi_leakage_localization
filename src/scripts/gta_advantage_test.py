from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from _common import *
from utils.template_attack import TemplateAttack
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from utils.calculate_snr import calculate_snr

profiling_dataset = SyntheticAES(
    epoch_length=int(1e5),
    timesteps_per_trace=1000,
    max_no_ops=2,
    shuffle_locs=2,
    lpf_beta=0.9
)
attack_dataset = SyntheticAESLike(
    profiling_dataset,
    epoch_length=int(1e3),
    fixed_key=np.uint8(0)
)

snr = calculate_snr(profiling_dataset)['subbytes']
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(snr, color='blue', linestyle='none', marker='.')
for cycle_idx, cycle in enumerate(profiling_dataset.leaking_subbytes_cycles):
    ax.axvspan(cycle, cycle+profiling_dataset.max_no_ops, color='blue', alpha=0.5, label='ground truth' if cycle_idx == 0 else None)
ax.set_xlabel('Timestep')
ax.set_ylabel('SNR')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'))

poi_counts = np.arange(1, 21, 2)
fig, axes = plt.subplots(1, len(poi_counts), figsize=(2*len(poi_counts), 2))
for poi_count, ax in tqdm(zip(poi_counts, axes.flatten())):
    points_of_interest = snr.argsort()[-poi_count:]
    if isinstance(points_of_interest, int):
        points_of_interest = np.array([points_of_interest])
    template_attack = TemplateAttack(points_of_interest)
    template_attack.profile(profiling_dataset)
    rank_over_time = template_attack.attack(attack_dataset, n_repetitions=100, n_traces=1000)
    mean = rank_over_time.mean(axis=0)
    stdev = rank_over_time.std(axis=0)
    ax.plot(mean, color='blue')
    ax.fill_between(range(len(mean)), mean-stdev, mean+stdev, color='blue', alpha=0.5)
    ax.set_ylim(0, 255)
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('Correct key rank')
    ax.set_title(f'POI count: {poi_count}')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'rank_over_time.pdf'))