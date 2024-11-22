import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_trace(trace):
    x = np.array([u.step for u in trace])
    y = np.array([u.value for u in trace])
    return (x, y)

def get_training_curves(self, logging_dir):
    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
    ea.Reload()
    training_curves = {key: extract_trace(ea.Scalars(key)) for key in ea.Tags()['scalars']}
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
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'))
    plt.close(fig)