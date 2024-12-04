from matplotlib import pyplot as plt

from common import *
from trials.utils import *

def plot_theta_pretraining_curves(logging_dir):
    training_curves = get_training_curves(logging_dir)
    assert all(key in training_curves for key in ['train_theta_loss', 'val_theta_loss', 'train_theta_rank', 'val_theta_rank'])
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes[0].plot(*training_curves['train_theta_loss'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_theta_loss'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_theta_rank'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_theta_rank'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[1].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Correct key rank')
    axes[0].set_yscale('log')
    fig.suptitle('Classifiers pretraining stage')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'theta_pretraining_curves.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)

def plot_gammap_training_curves(logging_dir):
    training_curves = get_training_curves(logging_dir)
    assert all(key in training_curves for key in [
        'train_gammap_mutinf_loss', 'train_gammap_identity_loss', 'train_gammap_total_loss', 'train_gammap_rank', 'train_gammap_mutinf_rms_grad', 'train_gammap_identity_rms_grad',
        'val_gammap_mutinf_loss', 'val_gammap_identity_loss', 'val_gammap_total_loss', 'val_gammap_rank', 'gamma'
    ])
    fig, axes = plt.subplots(2, 2, figsize=(2*PLOT_WIDTH, 2*PLOT_WIDTH))
    axes = axes.flatten()
    axes[0].plot(*training_curves['train_gammap_mutinf_loss'], color='red', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['train_gammap_identity_loss'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['train_gammap_total_loss'], color='purple', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_mutinf_loss'], color='red', linestyle='-', label='mutinf', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_identity_loss'], color='blue', linestyle='-', label='id', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_total_loss'], color='purple', linestyle='-', label='total', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_gammap_mutinf_rms_grad'], color='red', linestyle='none', marker='.', markersize=1, label='mutinf', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_gammap_identity_rms_grad'], color='blue', linestyle='none', marker='.', markersize=1, label='id', **PLOT_KWARGS)
    axes[2].plot(*training_curves['train_gammap_rank'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[2].plot(*training_curves['val_gammap_rank'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[3].plot(training_curves['gamma'][0], training_curves['gamma'][1].T, color='blue', linestyle='-', linewidth=0.1, **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[1].set_xlabel('Training step')
    axes[2].set_xlabel('Training step')
    axes[3].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('RMS gradient')
    axes[2].set_ylabel('Correct key rank')
    axes[3].set_ylabel('$\gamma_t$')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[3].set_yscale('log')
    fig.suptitle('Erasure probabilities training stage')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'gamma_training_curves.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)