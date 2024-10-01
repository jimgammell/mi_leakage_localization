import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from _common import *

BASE_DIR = r'/home/jgammell/Desktop/mi_leakage_localization/outputs/performance_correlation_computations'
DATASET_DIRS = [os.path.join(BASE_DIR, x) for x in ['ascadv1f', 'dpav4']]

NAMES = {
    'random': 'Random',
    'snr': 'SNR',
    'sosd': 'SOSD',
    'cpa': 'CPA',
    'gradvis': 'GradVis',
    'inputxgrad': 'Input$\\times$Grad',
    'ablation': 'Ablation',
    'all_results': 'ALL (Ours)'
}

for dataset_dir in DATASET_DIRS:
    fig, axes = plt.subplots(2, 8, figsize=(18, 4))
    for idx, technique in enumerate(['random', 'snr', 'sosd', 'cpa', 'gradvis', 'inputxgrad', 'ablation', 'all_results']):
        loc_ax = axes[0, idx]
        perf_ax = axes[1, idx]
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
            if (technique in ['snr', 'sosd', 'cpa']) and ('ascadv1f' in dataset_dir):
                loc_ax.plot(gt_loc_mask.squeeze(), color='red', marker='.', linestyle='none', markersize=3, rasterized=True)
                loc_ax.plot(gt_loc_masked_subbytes.squeeze(), color='yellow', marker='.', linestyle='none', markersize=3, rasterized=True)
                perf_ax.fill_between(np.arange(gt_perf_mean.shape[0]), gt_perf_mean-gt_perf_std, gt_perf_mean+gt_perf_std, color='orange', alpha=0.25, rasterized=True)
                perf_ax.plot(gt_perf_mean, color='orange')
            loc_ax.plot(loc.squeeze(), color='blue', marker='.', linestyle='none', markersize=3, rasterized=True)
            loc_ax.set_xlabel('Timestep')
            loc_ax.set_ylabel('Leakage assessment')
            loc_ax.set_title(f'{NAMES[technique]}')
            perf_ax.fill_between(np.arange(perf_mean.shape[0]), perf_mean-perf_std, perf_mean+perf_std, color='blue', alpha=0.25, rasterized=True)
            perf_ax.plot(perf_mean, color='blue')
            perf_ax.set_xlabel('Partition index')
            perf_ax.set_ylabel('Correct-key rank')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), f'{dataset_dir.split(os.sep)[-1]}__baselines.pdf'))