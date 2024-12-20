import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import imageio

from common import *
from trials.utils import *

def plot_inclusion_probs(inclusion_probs, ax):
    line, *_ = ax.plot(inclusion_probs.squeeze(), color='blue', marker='.', linestyle='-', markersize=1, linewidth=0.1, **PLOT_KWARGS)
    return line

def anim_inclusion_probs_traj(logging_dir):
    log_gammas = extract_log_gamma(logging_dir)
    output_path = os.path.join(logging_dir, 'inclusion_prob_traj.gif')
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].set_xlabel('Timestep $t$')
    axes[1].set_xlabel('Timestep $t$')
    axes[0].set_ylabel('Estimated leakage of $X_t$')
    axes[1].set_ylabel('Estimated leakage of $X_t$')
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale('log')
    indices = np.linspace(0, len(log_gammas[1])-1, 100).astype(int)
    with imageio.get_writer(output_path, mode='I', fps=20) as writer:
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx in indices:
                step = log_gammas[0][idx]
                log_gamma = log_gammas[1][idx]
                filename = os.path.join(temp_dir, f'step={step}.png')
                x0 = plot_inclusion_probs(np.exp(log_gamma), axes[0])
                x1 = plot_inclusion_probs(np.exp(log_gamma), axes[1])
                fig.suptitle(f'Training step: {step}')
                fig.tight_layout()
                fig.savefig(filename, **SAVEFIG_KWARGS)
                x0.remove()
                x1.remove()
                image = imageio.imread(filename)
                writer.append_data(image)
    plt.close(fig)

def plot_training_curves(logging_dir, anim_gammas=True):
    training_curves = get_training_curves(logging_dir)
    fig, axes = plt.subplots(2, 3, figsize=(3*PLOT_WIDTH, 2*PLOT_WIDTH))
    axes = axes.flatten()
    axes[0].plot(*training_curves['train_etat_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_etat_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_theta_loss'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_theta_loss'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
    axes[2].plot(*training_curves['train_theta_rank'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
    axes[2].plot(*training_curves['val_theta_rank'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
    axes[3].plot(*training_curves['train_etat_rms_grad'], color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[4].plot(*training_curves['train_theta_rms_grad'], color='red', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    lines = [np.column_stack([training_curves['log_gamma'][0], y]) for y in training_curves['gamma'][1].T]
    linekwargs = {'linewidth': 0.1, 'alpha': 0.5} if anim_gammas else {'linewidth': 1, 'alpha': 1.0}
    lc = LineCollection(lines, color='blue', linestyle='-', **linekwargs, **PLOT_KWARGS)
    axes[5].add_collection(lc)
    axes[5].autoscale()
    for ax in axes:
        ax.set_xlabel('Training step')
    axes[0].set_ylabel('Loss ($\tilde{\eta}$)')
    axes[1].set_ylabel('Loss ($\theta$)')
    axes[2].set_ylabel('Rank ($\theta$)')
    axes[3].set_ylabel('RMS gradient ($\tilde{\eta}$)')
    axes[4].set_ylabel('RMS gradient ($\theta$)')
    axes[5].set_ylabel('Inclusion probability $\gamma_t$')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_ylim(0, 256)
    axes[3].set_yscale('log')
    axes[4].set_yscale('log')
    axes[5].set_yscale('log')
    fig.suptitle('Cooperative fixed-budget leakage localization training curves')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'), **SAVEFIG_KWARGS)
    plt.close(fig)
    if anim_gammas:
        anim_inclusion_probs_traj(logging_dir)