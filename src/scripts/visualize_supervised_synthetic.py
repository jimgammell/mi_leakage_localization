import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict

from _common import *

Results = OrderedDict([
    ('synthetic_boolean_masking_supervised_htune', None),
    ('synthetic_no_ops_supervised_htune', None),
    ('synthetic_shuffling_supervised_htune', None),
    ('synthetic_unprotected_supervised_htune', None)
])
CURVES_PLOT_COUNT = 5

for sweep_dir in Results.keys():
    trial_dirs = os.listdir(os.path.join(OUTPUT_DIR, sweep_dir))
    results = {
        'learning_rate': [],
        'train_loss': [],
        'val_loss': [],
        'train_rank': [],
        'val_rank': [],
        'rank_over_time': []
    }
    for trial_dir in trial_dirs:
        if not 'learning_rate=' in trial_dir:
            continue
        learning_rate = float(trial_dir.split('learning_rate=')[-1])
        try:
            with open(os.path.join(OUTPUT_DIR, sweep_dir, trial_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
            with open(os.path.join(OUTPUT_DIR, sweep_dir, trial_dir, 'rank_over_time.pickle'), 'rb') as f:
                rank_over_time = pickle.load(f)
        except:
            continue
        results['learning_rate'].append(learning_rate)
        results['train_loss'].append(training_curves['train-loss'])
        results['val_loss'].append(training_curves['val-loss'])
        results['train_rank'].append(training_curves['train-rank'])
        results['val_rank'].append(training_curves['val-rank'])
        results['rank_over_time'].append(rank_over_time)
    for key, val in results.items():
        results[key] = np.array(val)
    sorted_indices = results['learning_rate'].argsort()
    for key, val in results.items():
        results[key] = val[sorted_indices]
    Results[sweep_dir] = results

for dir, results in Results.items():
    trial_count = len(results['learning_rate'])
    fig, axes = plt.subplots(3, CURVES_PLOT_COUNT, figsize=(12, 12*3/CURVES_PLOT_COUNT))
    trial_indices = np.linspace(0, trial_count-1, CURVES_PLOT_COUNT).astype(int)
    for ax_idx, trial_idx in enumerate(trial_indices):
        axes[0, ax_idx].plot(results['train_loss'][trial_idx, 0, :], results['train_loss'][trial_idx, 1, :], color='red', linestyle='--', rasterized=True)
        axes[0, ax_idx].plot(results['val_loss'][trial_idx, 0, :], results['val_loss'][trial_idx, 1, :], color='red', linestyle='-', rasterized=True)
        axes[1, ax_idx].plot(results['train_rank'][trial_idx, 0, :], results['train_rank'][trial_idx, 1, :], color='red', linestyle='--', rasterized=True)
        axes[1, ax_idx].plot(results['val_rank'][trial_idx, 0, :], results['val_rank'][trial_idx, 1, :], color='red', linestyle='-', rasterized=True)
        rank_over_time = results['rank_over_time'][trial_idx]
        axes[2, ax_idx].fill_between(np.arange(1, rank_over_time.shape[1]+1), np.percentile(rank_over_time, 25, axis=0), np.percentile(rank_over_time, 75, axis=0), color='blue', alpha=0.25, rasterized=True)
        axes[2, ax_idx].plot(np.arange(1, rank_over_time.shape[1]+1), np.median(rank_over_time, axis=0), color='blue', linestyle='-', rasterized=True)
        axes[0, ax_idx].set_title(f'lr={results["learning_rate"][trial_idx]:.1e}')
        axes[2, ax_idx].set_xlabel('Traces seen')
        axes[2, ax_idx].set_ylabel('Correct-key rank')
        axes[1, ax_idx].set_xlabel('Training steps')
        axes[0, ax_idx].set_xlabel('Training steps')
        axes[0, ax_idx].set_ylabel('Loss')
        axes[1, ax_idx].set_ylabel('Mean rank')
        axes[2, ax_idx].set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), f'{dir}__raw_output.pdf'))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(results['learning_rate'], results['val_loss'][:, 1, -1], color='blue', marker='.', linestyle='-', rasterized=True)
    axes[1].plot(results['learning_rate'], results['val_rank'][:, 1, -1], color='blue', marker='.', linestyle='-', rasterized=True)
    axes[2].plot(results['learning_rate'], np.mean(results['rank_over_time'], axis=(1, 2)), color='blue', marker='.', linestyle='-', rasterized=True)
    for ax in axes:
        ax.set_xlabel('Learning rate')
        ax.set_xscale('log')
    axes[0].set_ylabel('Final loss')
    axes[1].set_ylabel('Final mean rank')
    axes[2].set_ylabel('Final mean accumulated rank')
    axes[0].set_ylim(5, 10)
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), f'{dir}__sweep.pdf'))