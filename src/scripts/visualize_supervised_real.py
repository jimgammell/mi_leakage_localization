import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict

from _common import *

CURVES_PLOT_COUNT = 5
Results = OrderedDict([
    ('aes_hd_supervised_htune', None),
    ('aes_rd_supervised_htune', None),
    ('ascadv1f_supervised_htune', None),
    ('dpav4_hd_supervised_htune', None)
])
for sweep_dir in Results.keys():
    trial_dirs = os.listdir(os.path.join(OUTPUT_DIR, sweep_dir))
    results = {
        'learning_rate': [],
        'weight_decay': [],
        'additive_noise': [],
        'train_loss': [],
        'val_loss': [],
        'train_rank': [],
        'val_rank': [],
        'rank_over_time': []
    }
    for trial_dir in trial_dirs:
        if not 'learning_rate=' in trial_dir:
            continue
        learning_rate = float(trial_dir.split('learning_rate=')[-1].split('__')[0])
        weight_decay = float(trial_dir.split('weight_decay=')[-1].split('__')[0])
        additive_noise = float(trial_dir.split('additive_noise=')[-1])
        try:
            with open(os.path.join(OUTPUT_DIR, sweep_dir, trial_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
            with open(os.path.join(OUTPUT_DIR, sweep_dir, trial_dir, 'rank_over_time.pickle'), 'rb') as f:
                rank_over_time = pickle.load(f)
        except:
            continue
        results['learning_rate'].append(learning_rate)
        results['weight_decay'].append(weight_decay)
        results['additive_noise'].append(additive_noise)
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
    learning_rates = np.unique(results['learning_rate'])
    trial_count = len(results['learning_rate'])
    fig, axes = plt.subplots(3, CURVES_PLOT_COUNT, figsize=(12, 12*3/CURVES_PLOT_COUNT))
    lr_indices = np.linspace(0, len(learning_rates)-1, CURVES_PLOT_COUNT).astype(int)
    for ax_idx in range(CURVES_PLOT_COUNT):
        learning_rate = learning_rates[lr_indices[ax_idx]]
        for trial_idx in range(trial_count):
            if not results['learning_rate'][trial_idx] == learning_rate:
                continue
            if results['weight_decay'][trial_idx] == results['additive_noise'][trial_idx] == 0:
                color = 'red'
            elif (results['weight_decay'][trial_idx] > 0) and (results['additive_noise'][trial_idx] == 0):
                color = 'green'
            elif (results['weight_decay'][trial_idx] == 0) and (results['additive_noise'][trial_idx] > 0):
                color = 'blue'
            elif (results['weight_decay'][trial_idx] > 0) and (results['additive_noise'][trial_idx] > 0):
                color = 'orange'
            else:
                assert False
            axes[0, ax_idx].plot(results['train_loss'][trial_idx, 0, :], results['train_loss'][trial_idx, 1, :], color=color, linestyle='--', rasterized=True)
            axes[0, ax_idx].plot(results['val_loss'][trial_idx, 0, :], results['val_loss'][trial_idx, 1, :], color=color, linestyle='-', rasterized=True)
            axes[1, ax_idx].plot(results['train_rank'][trial_idx, 0, :], results['train_rank'][trial_idx, 1, :], color=color, linestyle='--', rasterized=True)
            axes[1, ax_idx].plot(results['val_rank'][trial_idx, 0, :], results['val_rank'][trial_idx, 1, :], color=color, linestyle='-', rasterized=True)
            rank_over_time = results['rank_over_time'][trial_idx]
            axes[2, ax_idx].fill_between(np.arange(1, rank_over_time.shape[1]+1), np.percentile(rank_over_time, 25, axis=0), np.percentile(rank_over_time, 75, axis=0), color=color, alpha=0.25, rasterized=True)
            axes[2, ax_idx].plot(np.arange(1, rank_over_time.shape[1]+1), np.median(rank_over_time, axis=0), color=color, linestyle='-', rasterized=True)
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
    for additive_noise in [0.0, 0.25]:
        for weight_decay in [0.0, 1e-2]:
            if weight_decay == additive_noise == 0:
                color = 'red'
            elif (weight_decay > 0) and (additive_noise == 0):
                color = 'green'
            elif (weight_decay == 0) and (additive_noise > 0):
                color = 'blue'
            elif (weight_decay > 0) and (additive_noise > 0):
                color = 'orange'
            else:
                assert False
            indices = np.logical_and(results['additive_noise'] == additive_noise, results['weight_decay'] == weight_decay)
            axes[0].plot(results['learning_rate'][indices], results['val_loss'][:, 1, -1][indices], color=color, marker='.', linestyle='-', rasterized=True, label=f'Weight decay: {weight_decay}, input noise: {additive_noise}')
            axes[1].plot(results['learning_rate'][indices], results['val_rank'][:, 1, -1][indices], color=color, marker='.', linestyle='-', rasterized=True)
            axes[2].plot(results['learning_rate'][indices], np.mean(results['rank_over_time'][indices], axis=(1, 2)), color=color, marker='.', linestyle='-', rasterized=True)
    for ax in axes:
        ax.set_xlabel('Learning rate')
        ax.set_xscale('log')
    axes[0].set_ylabel('Final loss')
    axes[1].set_ylabel('Final mean rank')
    axes[2].set_ylabel('Final mean accumulated rank')
    axes[2].set_yscale('log')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), f'{dir}__sweep.pdf'))