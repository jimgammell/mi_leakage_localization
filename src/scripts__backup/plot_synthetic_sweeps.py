import os
import pickle
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule

UNPROTECTED_EG_BASE_DIR = os.path.join(OUTPUT_DIR, 'synthetic_noise_level_sweep__backup', 'data_var=0.5__lambda=0.1__seed=0')
NOOPS_EG_BASE_DIR = os.path.join(OUTPUT_DIR, 'synthetic_no_op_sweep__backup', 'no_ops=16__lambda=0.01__seed=0')
SHUFFLING_EG_BASE_DIR = os.path.join(OUTPUT_DIR, 'synthetic_shuffling_sweep__backup', 'shuffle_locs=8__lambda=0.01__seed=0')

def get_leakage_assessment(base_dir):
    lambda_val = float(base_dir.split('lambda=')[-1].split('__')[0])
    checkpoints_dir = os.path.join(base_dir, 'lightning_output', 'lightning_logs', 'version_0', 'checkpoints')
    checkpoint_names = os.listdir(checkpoints_dir)
    assert len(checkpoint_names) == 1
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_names[0])
    training_module = AdversarialLocalizationModule.load_from_checkpoint(
        checkpoint_path,
        classifier_name='multilayer-perceptron',
        classifier_optimizer_name='AdamW',
        obfuscator_optimizer_name='AdamW',
        obfuscator_l2_norm_penalty=lambda_val,
        classifier_kwargs={'input_shape': (2, 100)},
        classifier_optimizer_kwargs={'lr': 1e-3},
        obfuscator_optimizer_kwargs={'lr': 1e-2},
        obfuscator_batch_size_multiplier=8
    )
    erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
    leakage_assessment = erasure_probs * lambda_val
    with open(os.path.join(base_dir, 'dataset_properties.pickle'), 'rb') as f:
        dataset_properties = pickle.load(f)
    return leakage_assessment, lambda_val, dataset_properties

unprotected_leakage_assessment, unprotected_lambda, unprotected_dataset_properties = get_leakage_assessment(UNPROTECTED_EG_BASE_DIR)
noops_leakage_assessment, noops_lambda, noops_dataset_properties = get_leakage_assessment(NOOPS_EG_BASE_DIR)
shuffling_leakage_assessment, shuffling_lambda, shuffling_dataset_properties = get_leakage_assessment(SHUFFLING_EG_BASE_DIR)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].axvline(unprotected_dataset_properties['leaking_subbytes_cycles'][0], color='orange', label='Ground truth')
axes[0].plot(unprotected_leakage_assessment, color='blue', marker='.', linestyle='none', markersize=3, label='$\lambda \gamma_t^*$')
axes[0].set_ylim(0, unprotected_lambda)
max_no_ops = int(NOOPS_EG_BASE_DIR.split('no_ops=')[-1].split('__')[0])
axes[1].axvspan(noops_dataset_properties['leaking_subbytes_cycles'][0], noops_dataset_properties['leaking_subbytes_cycles'][0]+max_no_ops+1, color='orange', alpha=0.25, label='Ground truth')
axes[1].plot(noops_leakage_assessment, color='blue', marker='.', linestyle='none', markersize=3, label='$\lambda \gamma_t^*$')
axes[1].set_ylim(0, noops_lambda)
for idx, cycle in enumerate(shuffling_dataset_properties['leaking_subbytes_cycles']):
    axes[2].axvline(cycle, color='orange', label='Ground truth' if idx == 0 else None)
axes[2].plot(shuffling_leakage_assessment, color='blue', marker='.', linestyle='none', markersize=3, label='$\lambda \gamma_t^*$')
axes[2].set_ylim(0, shuffling_lambda)
for ax in axes:
    ax.legend()
    ax.set_xlabel('Timestep: $t$')
    ax.set_ylabel('Estimated leakage: $\lambda \gamma_t^*$')
axes[0].set_title('Simulated unprotected implementation')
axes[1].set_title('Simulated random no-op insertion')
axes[2].set_title('Simulated random shuffling')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'eg_leakage_assessments.pdf'))