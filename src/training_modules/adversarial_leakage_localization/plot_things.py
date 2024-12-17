from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import imageio
import tempfile

from common import *
from trials.utils import *

def plot_erasure_probs(erasure_probs, ax):
    line, *_ = ax.plot(erasure_probs.squeeze(), color='blue', marker='.', linestyle='-', markersize=1, linewidth=0.1, **PLOT_KWARGS)
    return line

def animate_erasure_probs_traj(logging_dir):
    gammas = extract_gamma(logging_dir)
    output_path = os.path.join(logging_dir, 'erasure_prob_traj.gif')
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].set_xlabel('Timestep $t$')
    axes[0].set_ylabel('Estimated leakage of $X_t$')
    axes[1].set_xlabel('Timestep $t$')
    axes[1].set_ylabel('Estimated leakage of $X_t$')
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale('log')
    indices = np.linspace(0, len(gammas[1]), 100).astype(int)
    with imageio.get_writer(output_path, mode='I', fps=20) as writer:
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx in indices:
                step = gammas[0][idx]
                erasure_probs = gammas[1][idx]
                filename = os.path.join(temp_dir, f'ep_step={step}.png')
                x0 = plot_erasure_probs(erasure_probs, axes[0])
                x1 = plot_erasure_probs(erasure_probs, axes[1])
                fig.suptitle(f'Training step: {step}')
                fig.tight_layout()
                fig.savefig(filename, **SAVEFIG_KWARGS)
                x0.remove()
                x1.remove()
                image = imageio.imread(filename)
                writer.append_data(image)
    plt.close(fig)

def plot_theta_pretraining_curves(logging_dir):
    training_curves = get_training_curves(logging_dir)
    assert all(key in training_curves for key in ['train_theta_loss', 'val_theta_loss', 'train_theta_rank', 'val_theta_rank', 'train_theta_grad'])
    fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes[0].plot(*training_curves['train_theta_loss'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_theta_loss'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_theta_rank'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[1].plot(*training_curves['val_theta_rank'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[2].plot(*training_curves['train_theta_grad'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[1].set_xlabel('Training step')
    axes[2].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Correct key rank')
    axes[2].set_ylabel('Max grad element')
    axes[2].set_yscale('log')
    fig.suptitle('Classifiers pretraining stage')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'theta_pretraining_curves.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)

def plot_gammap_training_curves(logging_dir, anim_gammas=True):
    training_curves = get_training_curves(logging_dir)
    assert all(key in training_curves for key in [
        'train_gammap_mutinf_loss', 'train_gammap_identity_loss', 'train_gammap_loss', 'train_gammap_mutinf_rms_grad', 'train_gammap_identity_rms_grad',
        'val_gammap_mutinf_loss', 'val_gammap_identity_loss', 'val_gammap_loss', 'gamma', 'train_theta_rank', 'val_theta_rank', 'tau', 'eta'
    ])
    fig, axes = plt.subplots(2, 3, figsize=(3*PLOT_WIDTH, 2*PLOT_WIDTH))
    axes = axes.flatten()
    axes[0].plot(*training_curves['train_gammap_mutinf_loss'], color='red', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['train_gammap_identity_loss'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['train_gammap_loss'], color='purple', linestyle='--', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_mutinf_loss'], color='red', linestyle='-', label='mutinf', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_identity_loss'], color='blue', linestyle='-', label='id', **PLOT_KWARGS)
    axes[0].plot(*training_curves['val_gammap_loss'], color='purple', linestyle='-', label='total', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_gammap_mutinf_rms_grad'], color='red', linestyle='none', marker='.', markersize=1, label='mutinf', **PLOT_KWARGS)
    axes[1].plot(*training_curves['train_gammap_identity_rms_grad'], color='blue', linestyle='none', marker='.', markersize=1, label='id', **PLOT_KWARGS)
    axes[2].plot(*training_curves['train_theta_rank'], color='blue', linestyle='--', **PLOT_KWARGS)
    axes[2].plot(*training_curves['val_theta_rank'], color='blue', linestyle='-', **PLOT_KWARGS)
    lines = [np.column_stack([training_curves['gamma'][0], y]) for y in training_curves['gamma'][1].T]
    lc = LineCollection(lines, color='blue', linestyle='-', linewidth=0.1, alpha=0.5, **PLOT_KWARGS)
    axes[3].add_collection(lc)
    axes[3].autoscale()
    axes[4].plot(*training_curves['tau'], color='red', linestyle='-', **PLOT_KWARGS)
    axes[4].plot(*training_curves['eta'], color='blue', linestyle='-', **PLOT_KWARGS)
    axes[0].set_xlabel('Training step')
    axes[1].set_xlabel('Training step')
    axes[2].set_xlabel('Training step')
    axes[3].set_xlabel('Training step')
    axes[4].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('RMS gradient')
    axes[2].set_ylabel('Correct key rank')
    axes[3].set_ylabel(r'$\gamma_t$')
    axes[4].set_ylabel(r'$\tau$ and $\eta$')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_yscale('symlog')
    axes[1].set_yscale('log')
    axes[3].set_yscale('log')
    axes[4].set_yscale('log')
    fig.suptitle('Erasure probabilities training stage')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'gamma_training_curves.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)
    if anim_gammas:
        animate_erasure_probs_traj(logging_dir)