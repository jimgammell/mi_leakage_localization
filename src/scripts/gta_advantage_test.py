from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from _common import *
from utils.template_attack import TemplateAttack
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from utils.calculate_snr import calculate_snr
from utils.performance_correlation import MeasurePerformanceCorrelation
from utils.performance_correlation.functional import soft_kendall_tau, partition_timesteps
from utils.advantage_correlation import AdvantageCorrelation

profiling_dataset = SyntheticAES(
    epoch_length=int(1e5),
    timesteps_per_trace=100,
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
poi_partition = partition_timesteps(snr, 5)
fig, ax = plt.subplots(figsize=(4, 4))
for idx, _poi_partition in enumerate(poi_partition):
    ax.plot(_poi_partition, snr[_poi_partition], linestyle='none', marker='.')
for cycle_idx, cycle in enumerate(profiling_dataset.leaking_subbytes_cycles):
    ax.axvspan(cycle, cycle+profiling_dataset.max_no_ops, color='blue', alpha=0.5, label='ground truth' if cycle_idx == 0 else None)
ax.set_xlabel('Timestep')
ax.set_ylabel('SNR')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
performance_correlation = MeasurePerformanceCorrelation(snr, profiling_dataset, attack_dataset)
correlation, metrics = performance_correlation.measure_performance()
mean, std = metrics.mean(axis=0), metrics.std(axis=0)
axes[0].plot(mean, color='blue')
axes[0].fill_between(np.arange(len(mean)), mean-std, mean+std, color='blue', alpha=0.5)
print(f'Performance correlation (SNR): {correlation}')
performance_correlation = MeasurePerformanceCorrelation(np.random.randn(len(snr)), profiling_dataset, attack_dataset)
correlation, metrics = performance_correlation.measure_performance()
mean, std = metrics.mean(axis=0), metrics.std(axis=0)
axes[1].plot(mean, color='blue')
axes[1].fill_between(np.arange(len(mean)), mean-std, mean+std, color='blue', alpha=0.5)
print(f'Performance correlation (random): {correlation}')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'performance_correlation.pdf'))


r"""advantage_correlation = AdvantageCorrelation(snr, profiling_dataset, attack_dataset)
print(f'Advantage correlation (SNR): {advantage_correlation.measure_performance()}')
advantage_correlation = AdvantageCorrelation(np.random.randint(100), profiling_dataset, attack_dataset)
print(f'Advantage correlation (random): {advantage_correlation.measure_performance()}')"""

r"""fig, axes = plt.subplots(1, len(poi_partition), figsize=(2*len(poi_partition), 2))
for pois, ax in zip(poi_partition, axes.flatten()):
    template_attack = TemplateAttack(pois, target_key='subbytes')
    template_attack.profile(profiling_dataset)
    rank_over_time = template_attack.attack(attack_dataset, n_repetitions=1000, n_traces=1000)
    mean = rank_over_time.mean(axis=0)
    stdev = rank_over_time.std(axis=0)
    ax.plot(mean, color='blue')
    ax.fill_between(range(len(mean)), mean-stdev, mean+stdev, color='blue', alpha=0.5)
    ax.set_ylim(0, 255)
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('Correct key rank')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'perf_vs_poi.pdf'))"""


r"""good_ranking = np.arange(100, dtype=np.float32) + np.random.randn(100)
bad_ranking = np.random.randn(100)
print(f'Good ranking Kendall tau correlation: {soft_kendall_tau(good_ranking, np.arange(100))}')
print(f'Bad ranking Kendall tau correlation: {soft_kendall_tau(bad_ranking, np.arange(100))}')"""

r"""performance_metric = MeasurePerformanceCorrelation(snr, profiling_dataset, attack_dataset)
correlation, performance_metrics = performance_metric.measure_performance(poi_count=10, seed_count=1, attack_seed_count=1000)
print(f'Correlation after attacking device with SNR: {correlation}')
fig, ax = plt.subplots(figsize=(4, 4))
mean, std = performance_metrics.mean(axis=0), performance_metrics.std(axis=0)
ax.fill_between(np.arange(mean.shape[0]), mean-std, mean+std, color='blue', alpha=0.5)
ax.plot(mean, color='blue', linestyle='-')
ax.set_xlabel('Position in ranking')
ax.set_ylabel('Performance metric')
fig.savefig(os.path.join(get_trial_dir(), 'correlation.pdf'))"""

r"""poi_counts = np.arange(1, 21, 2)
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
fig.savefig(os.path.join(get_trial_dir(), 'rank_over_time.pdf'))"""