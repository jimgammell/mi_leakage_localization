import os
from copy import copy
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
    'all_results': '\\textbf{ALL (ours)}'
}

for dataset_dir in DATASET_DIRS:
    to_run = ['random', 'snr', 'sosd', 'cpa', 'gradvis', 'inputxgrad', 'ablation', 'all_results']
    prob_fig, prob_axes = plt.subplots(1, len(to_run), figsize=(18, 18/len(to_run)))
    for idx, technique in enumerate(to_run):
        loc_ax = prob_axes[idx]
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
            loc_ax.plot(loc.squeeze(), color='blue', marker='.', linestyle='none', markersize=3, rasterized=True)
            loc_ax.set_xlabel('Timestep $t$')
            loc_ax.set_title(f'{NAMES[technique]}')
    prob_axes[0].set_ylabel('Est. leakage of $X_t$')
    prob_fig.tight_layout()
    filename = f'{dataset_dir.split(os.sep)[-1]}__baselines__main.pdf'
    prob_fig.savefig(os.path.join(get_trial_dir(), filename))