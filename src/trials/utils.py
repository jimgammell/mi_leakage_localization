import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_trace(trace):
    x = np.array([u.step for u in trace])
    y = np.array([u.value for u in trace])
    return (x, y)

def extract_gamma(logging_dir):
    gamma_dir = os.path.join(logging_dir, 'gamma_log')
    if not os.path.exists(gamma_dir):
        return None
    steps, gammas = [], []
    for file in os.listdir(gamma_dir):
        if not file.endswith('.npy'):
            continue
        step = int(file.split('=')[-1].split('.')[0])
        gamma = np.load(os.path.join(gamma_dir, file))
        steps.append(step)
        gammas.append(gamma)
    indices = np.argsort(steps)
    steps = np.array(steps)[indices]
    gammas = np.stack(gammas)[indices, ...]
    return (steps, gammas)

def get_training_curves(logging_dir):
    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
    ea.Reload()
    training_curves = {key: extract_trace(ea.Scalars(key)) for key in ea.Tags()['scalars']}
    gamma_curves = extract_gamma(logging_dir)
    if gamma_curves is not None:
        training_curves.update({'gamma': gamma_curves})
    return training_curves

def save_training_curves(training_curves, logging_dir):
    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
        pickle.dump(training_curves, f)

def load_training_curves(logging_dir):
    if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
        return None
    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
        training_curves = pickle.load(f)
    return training_curves

def plot_training_curves(training_curves, logging_dir, keys=[]):
    plot_count = len(keys)
    fig, axes = plt.subplots(1, plot_count, figsize=(4*plot_count, 4), sharex=True)
    for _keys, ax in zip(keys, axes):
        for key in _keys:
            ax.plot(*training_curves[key], label=key.replace('_', '\_'))
        ax.set_xlabel('Training step')
        ax.set_ylabel('Metric value')
        ax.legend()
        if any('loss' in key for key in keys) and all(training_curves[key][1].min() > 0 for key in keys):
            ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'))
    plt.close(fig)