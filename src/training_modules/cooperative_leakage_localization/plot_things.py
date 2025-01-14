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
    axes[0].set_xlabel(r'Timestep $t$')
    axes[1].set_xlabel(r'Timestep $t$')
    axes[0].set_ylabel(r'Estimated leakage of $X_t$')
    axes[1].set_ylabel(r'Estimated leakage of $X_t$')
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
    
def plot_vs_reference(logging_dir, inclusion_probs, reference):
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
    ax.set_xlabel('Reference')
    ax.set_ylabel('Gammas')
    ax.plot(reference.squeeze(), inclusion_probs.squeeze(), marker='.', markersize=1, linestyle='none', color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'comparison_to_reference.png'))
    plt.close(fig)

def plot_training_curves(logging_dir, anim_gammas=True, reference=None):
    training_curves = get_training_curves(logging_dir)
    fig, axes = plt.subplots(4, 3, figsize=(3*PLOT_WIDTH, 2*PLOT_WIDTH))
    axes = axes.flatten()
    if all(x in training_curves for x in ['train_etat_loss', 'val_etat_loss']):
        axes[0].plot(*training_curves['train_etat_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
        axes[0].plot(*training_curves['val_etat_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
        axes[0].plot(*training_curves['train_hard_eta_loss'], color='red', linestyle='--', **PLOT_KWARGS)
        axes[0].plot(*training_curves['val_hard_eta_loss'], color='red', linestyle='-', **PLOT_KWARGS)
    if all(x in training_curves for x in ['train_theta_loss', 'val_theta_loss']):
        axes[1].plot(*training_curves['train_theta_loss'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
        axes[1].plot(*training_curves['val_theta_loss'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
        if ('train_theta_loss_calibrated' in training_curves) and ('val_theta_loss_calibrated' in training_curves):
            axes[1].plot(*training_curves['train_theta_loss_calibrated'], color='blue', linestyle='--', **PLOT_KWARGS)
            axes[1].plot(*training_curves['val_theta_loss_calibrated'], color='blue', linestyle='-', **PLOT_KWARGS)
    if all(x in training_curves for x in ['train_theta_rank', 'val_theta_rank']):
        axes[2].plot(*training_curves['train_theta_rank'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
        axes[2].plot(*training_curves['val_theta_rank'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
    if 'train_etat_rms_grad' in training_curves:
        axes[3].plot(*training_curves['train_etat_rms_grad'], color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    if 'train_theta_rms_grad' in training_curves:
        axes[4].plot(*training_curves['train_theta_rms_grad'], color='red', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    lines = [np.column_stack([training_curves['log_gamma'][0], np.exp(y)]) for y in training_curves['log_gamma'][1].T]
    if not anim_gammas: # this is a simple Gaussian dataset trial where the first line is the nonleaky point
        nonleaky_line = lines[0]
        lines = lines[1:]
    linekwargs = {'linewidth': 0.1, 'alpha': 0.5} if anim_gammas else {'linewidth': 1, 'alpha': 1.0}
    lc = LineCollection(lines, color='blue', linestyle='-', **linekwargs, **PLOT_KWARGS)
    axes[5].add_collection(lc)
    if not anim_gammas:
        axes[5].plot(nonleaky_line[:, 0], nonleaky_line[:, 1], color='red')
    axes[5].autoscale()
    ktcc_curves = {key: val for key, val in training_curves.items() if key.endswith('_ktcc')}
    corr_curves = {key: val for key, val in training_curves.items() if key.endswith('_corr')}
    if len(ktcc_curves) > 0:
        for key, val in ktcc_curves.items():
            axes[6].plot(*val, label=key.replace('_', r'\_'), **PLOT_KWARGS)
        axes[6].legend()
    if len(corr_curves) > 0:
        for key, val in corr_curves.items():
            axes[7].plot(*val, label=key.replace('_', r'\_'), **PLOT_KWARGS)
        axes[7].legend()
    if 'gmmperfcorr' in training_curves:
        axes[8].plot(*training_curves['gmmperfcorr'], color='blue', **PLOT_KWARGS)
    if 'train_rebar_eta' in training_curves:
        axes[9].plot(*training_curves['train_rebar_eta'], color='blue', **PLOT_KWARGS)
    if 'train_rebar_tau' in training_curves:
        axes[10].plot(*training_curves['train_rebar_tau'], color='blue', **PLOT_KWARGS)
    if 'train_temperature' in training_curves:
        axes[11].plot(*training_curves['train_temperature'], color='blue', **PLOT_KWARGS)
    for ax in axes:
        ax.set_xlabel('Training step')
    axes[0].set_ylabel(r'Loss ($\tilde{\eta}$)')
    axes[1].set_ylabel(r'Loss ($\theta$)')
    axes[2].set_ylabel(r'Rank ($\theta$)')
    axes[3].set_ylabel(r'RMS gradient ($\tilde{\eta}$)')
    axes[4].set_ylabel(r'RMS gradient ($\theta$)')
    axes[5].set_ylabel(r'Inclusion probability $\gamma_t$')
    axes[6].set_ylabel('KTCC with reference leakage assessment')
    axes[7].set_ylabel('Correlation with reference leakage assessment')
    axes[8].set_ylabel('GMM performance correlation')
    axes[9].set_ylabel(r'REBAR $\eta$')
    axes[10].set_ylabel(r'REBAR $\tau$')
    axes[11].set_ylabel('Calibration temperature')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_yscale('symlog')
    axes[1].set_yscale('log')
    axes[3].set_yscale('log')
    axes[4].set_yscale('log')
    axes[5].set_yscale('log')
    fig.suptitle('Cooperative fixed-budget leakage localization training curves')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'), **SAVEFIG_KWARGS)
    plt.close(fig)
    if reference is not None:
        plot_vs_reference(logging_dir, np.exp(training_curves['log_gamma'][1][-1, ...].squeeze()), reference.squeeze())
    if anim_gammas:
        anim_inclusion_probs_traj(logging_dir)