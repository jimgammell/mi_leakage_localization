import numpy as np
from matplotlib import pyplot as plt

from common import *
from trials.utils import *

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