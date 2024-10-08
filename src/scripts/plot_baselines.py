import os
from copy import copy
import pickle
import numpy as np
from matplotlib import pyplot as plt

from _common import *

BASE_DIR = r'/home/jgammell/Desktop/mi_leakage_localization/outputs/performance_correlation_computations'
DATASET_DIRS = [os.path.join(BASE_DIR, x) for x in ['ascadv1f', 'dpav4']]
MAIN_VERSION = True

NAMES = {
    'random': 'Random',
    'snr': 'Signal-noise ratio',
    'sosd': 'Sum of squared differences',
    'cpa': 'Correlation power analysis',
    'gradvis': 'Gradient visualization',
    'inputxgrad': 'Input$\\times$Grad',
    'ablation': 'Ablation',
    'all_results': 'Adversarial leakage localization (ours)'
}

for dataset_dir in DATASET_DIRS:
    if not MAIN_VERSION:
        to_run = ['random', 'snr', 'sosd', 'cpa', 'gradvis', 'inputxgrad', 'ablation', 'all_results']
    elif 'ascad' in dataset_dir:
        to_run = ['random', 'snr', 'gradvis', 'all_results']
    elif 'dpa' in dataset_dir:
        to_run = ['random', 'sosd', 'gradvis', 'all_results']
    prob_fig, prob_axes = plt.subplots(1, len(to_run), figsize=(18, 15/len(to_run)))
    corr_fig, corr_axes = plt.subplots(1, len(to_run), figsize=(12, 10/len(to_run)))
    for idx, technique in enumerate(to_run):
        loc_ax = prob_axes[idx]
        perf_ax = corr_axes[1, idx]
        if technique in ['random', 'snr', 'sosd', 'cpa', 'gradvis', 'inputxgrad', 'ablation', 'all_results']:
            if technique in ['random']:
                with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}__perfcorr__seed=0.pickle'), 'rb') as f:
                    perfcorr = pickle.load(f)
                corr, perf_mean, perf_std = perfcorr['corr'], perfcorr['means'].squeeze(), perfcorr['stds'].squeeze()
                loc = np.random.randn(700 if 'ascadv1f' in dataset_dir else 4000)
            elif technique in ['snr', 'sosd', 'cpa']:
                with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}.pickle'), 'rb') as f:
                    results = pickle.load(f)
                    loc = results[('subbytes', None)]
                    if 'ascadv1f' in dataset_dir:
                        gt_loc_mask = results[('r', None)]
                        gt_loc_masked_subbytes = results[('subbytes__r', None)]
                with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}__perfcorr.pickle'), 'rb') as f:
                    perfcorr = pickle.load(f)
                if 'ascadv1f' in dataset_dir:
                    with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}_gt__perfcorr.pickle'), 'rb') as f:
                        gt_perfcorr = pickle.load(f)
                corr, perf_mean, perf_std = perfcorr['corr'], perfcorr['means'].squeeze(), perfcorr['stds'].squeeze()
                if 'ascadv1f' in dataset_dir:
                    gt_corr, gt_perf_mean, gt_perf_std = gt_perfcorr['corr'], gt_perfcorr['means'].squeeze(), gt_perfcorr['stds'].squeeze()
            elif technique in ['gradvis', 'inputxgrad', 'ablation']:
                with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}__seed=0.pickle'), 'rb') as f:
                    loc = pickle.load(f)
                with open(os.path.join(BASE_DIR, dataset_dir, f'{technique}__perfcorr__seed=0.pickle'), 'rb') as f:
                    perfcorr = pickle.load(f)
                corr, perf_mean, perf_std = perfcorr['corr'], perfcorr['means'].squeeze(), perfcorr['stds'].squeeze()
            elif technique in ['all_results']:
                with open(os.path.join(BASE_DIR, dataset_dir, 'all_results__seed=0.pickle'), 'rb') as f:
                    results = pickle.load(f)
                    loc = results['erasure_probs'].squeeze()
                with open(os.path.join(BASE_DIR, dataset_dir, 'all__perfcorr__seed=0.pickle'), 'rb') as f:
                    perfcorr = pickle.load(f)
                corr, perf_mean, perf_std = perfcorr['corr'], perfcorr['means'].squeeze(), perfcorr['stds'].squeeze()
            if (technique in ['snr', 'sosd', 'cpa']) and ('ascadv1f' in dataset_dir) and not(MAIN_VERSION):
                loc_ax.plot(gt_loc_mask.squeeze(), color='red', marker='.', linestyle='none', markersize=3, rasterized=True)
                loc_ax.plot(gt_loc_masked_subbytes.squeeze(), color='yellow', marker='.', linestyle='none', markersize=3, rasterized=True)
                perf_ax.fill_between(np.arange(gt_perf_mean.shape[0]), gt_perf_mean-gt_perf_std, gt_perf_mean+gt_perf_std, color='orange', alpha=0.25, rasterized=True)
                perf_ax.plot(gt_perf_mean, color='orange')
            loc_ax.plot(loc.squeeze(), color='blue', marker='.', linestyle='none', markersize=3, rasterized=True)
            loc_ax.set_xlabel('Timestep $t$')
            loc_ax.set_title(f'{NAMES[technique]}')
            perf_ax.fill_between(np.arange(perf_mean.shape[0]), perf_mean-perf_std, perf_mean+perf_std, color='blue', alpha=0.25, rasterized=True)
            perf_ax.plot(perf_mean, color='blue')
            perf_ax.set_xlabel('Partition index')
    axes[0, 0].set_ylabel('Estimated leakage of $X_t$')
    axes[1, 0].set_ylabel('Rank of correct key')
    fig.tight_layout()
    filename = f'{dataset_dir.split(os.sep)[-1]}__baselines'
    if MAIN_VERSION:
        filename += '__main'
    filename += '.pdf'
    fig.savefig(os.path.join(get_trial_dir(), filename))