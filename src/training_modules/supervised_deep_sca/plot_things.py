import numpy as np
from matplotlib import pyplot as plt

from common import *
from trials.utils import *

def plot_hparam_sweep(logging_dir):
    with open(os.path.join(logging_dir, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)
    result_names = ['min_rank', 'final_rank', 'min_loss', 'final_loss']
    assert all(name in results.keys() for name in result_names)
    hparam_names = [key for key in results.keys() if key not in result_names]
    fig, axes = plt.subplots(len(hparam_names), len(result_names), figsize=(PLOT_WIDTH*len(result_names), PLOT_WIDTH*len(hparam_names)))
    optimal_settings = {}
    for row_idx, (hparam_name, axes_row) in enumerate(zip(hparam_names, axes)):
        for col_idx, (result_name, ax) in enumerate(zip(result_names, axes_row)):
            hparam_vals = results[hparam_name]
            distinct_hparam_vals = list(set(hparam_vals))
            if all(isinstance(x, int) or isinstance(x, float) for x in hparam_vals):
                distinct_hparam_vals.sort()
            result_vals = results[result_name]
            label_to_num = {hparam_name: idx for idx, hparam_name in enumerate(distinct_hparam_vals)}
            mean_vals = [np.exp(np.mean(np.log(np.array([result_vals[idx] for idx in range(len(result_vals)) if hparam_vals[idx] == val])))) for val in distinct_hparam_vals]
            optimal_idx = np.argmin(mean_vals)
            optimal_settings[hparam_name] = distinct_hparam_vals[optimal_idx]
            xx = [label_to_num[x] for x in hparam_vals]
            ax.plot(xx, result_vals, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            ax.plot([label_to_num[x] for x in distinct_hparam_vals], mean_vals, color='red', marker='.', linestyle='none', markersize=5, **PLOT_KWARGS)
            ax.set_xticks(list(label_to_num.values()))
            ax.set_xticklabels([str(x) for x in label_to_num.keys()])
            ax.set_xlabel(hparam_name.replace('_', '\_'))
            ax.set_ylabel(result_name.replace('_', '\_'))
            if 'loss' in result_name:
                ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'hparam_sweep.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)
    return optimal_settings

def plot_training_curves(logging_dir):
    training_curves = get_training_curves(logging_dir)
    fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes = axes.flatten()
    if all(x in training_curves for x in ['train_loss', 'val_loss']):
        axes[0].plot(*training_curves['train_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
        axes[0].plot(*training_curves['val_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    if all(x in training_curves for x in ['train_rank', 'val_rank']):
        axes[1].plot(*training_curves['train_rank'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
        axes[1].plot(*training_curves['val_rank'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    if 'train_rms_grad' in training_curves:
        axes[2].plot(*training_curves['train_rms_grad'], color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel('Training step')
    axes[1].set_xlabel('Training step')
    axes[2].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Rank')
    axes[2].set_ylabel('RMS gradient')
    axes[0].set_yscale('log')
    axes[2].set_yscale('log')
    fig.suptitle('Supervised deep SCA training curves')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'), **SAVEFIG_KWARGS)
    plt.close(fig)