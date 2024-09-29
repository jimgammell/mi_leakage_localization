import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
import datasets
from utils.metrics.rank import accumulate_ranks
from utils.aes import subbytes_to_keys
from training_modules import AdversarialLocalizationModule

TIMESTEPS_PER_TRACE = 500
STEP_COUNT = 10000

data_module = datasets.load(
    'synthetic-aes',
    train_batch_size=1024,
    eval_batch_size=10000,
    aug=False,
    dataset_kwargs=dict(
        timesteps_per_trace=TIMESTEPS_PER_TRACE,
        leaking_timestep_count_1o=1,
        leaking_timestep_count_2o=0,
        shuffle_locs=9,
        max_no_ops=0,
        infinite_dataset=True,
        data_var=1.0,
        residual_var=1.0,
        lpf_beta=0.9
    )
)
norm_penalties = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
for norm_penalty in norm_penalties:
    logging_dir = os.path.join(get_trial_dir(), f'lambda={norm_penalty}')
    training_module = AdversarialLocalizationModule(
        classifier_name='multilayer-perceptron',
        classifier_kwargs={'input_shape': (2, TIMESTEPS_PER_TRACE)},
        classifier_optimizer_name='AdamW',
        obfuscator_optimizer_name='AdamW',
        classifier_optimizer_kwargs={'lr': 1e-4},
        obfuscator_optimizer_kwargs={'lr': 5e-4},
        obfuscator_l2_norm_penalty=norm_penalty,
        obfuscator_batch_size_multiplier=8,
        normalize_erasure_probs_for_classifier=True
    )
    trainer = Trainer(
        max_epochs=STEP_COUNT*1024//100000,
        default_root_dir=logging_dir,
        accelerator='gpu',
        devices=1,
        logger=TensorBoardLogger(logging_dir, name='lightning_output')
    )
    trainer.fit(training_module, datamodule=data_module)
    trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
    ea.Reload()
    training_curves = {
        key: extract_trace(ea.Scalars(key)) for key in ['classifier-train-loss_epoch', 'classifier-val-loss', 'obfuscator-train-loss_epoch', 'obfuscator-val-loss', 'train-rank', 'val-rank']
    }
    erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
    data_module.train_dataset.transform = data_module.train_dataset.target_transform = None
    with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
        pickle.dump({'training_curves': training_curves, 'erasure_probs': erasure_probs, 'train_dataset': data_module.train_dataset}, f)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].plot(*training_curves['classifier-train-loss_epoch'], color='red', linestyle='--')
    axes[0].plot(*training_curves['classifier-val-loss'], color='red', linestyle='-')
    axes[1].plot(*training_curves['train-rank'], color='red', linestyle='--')
    axes[1].plot(*training_curves['val-rank'], color='red', linestyle='-')
    axes[2].plot(*training_curves['obfuscator-train-loss_epoch'], color='blue', linestyle='--')
    axes[2].plot(*training_curves['obfuscator-val-loss'], color='blue', linestyle='-')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Classifier loss')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Mean rank')
    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel('Obfuscator loss')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.pdf'))

    fig, ax = plt.subplots(figsize=(4, 4))
    for cycle in data_module.train_dataset.leaking_subbytes_cycles:
        ax.axvline(cycle, color='orange')
    ax.plot(training_module.obfuscator_l2_norm_penalty*erasure_probs, color='blue', linestyle='none', marker='.')
    ax.set_ylim(0, training_module.obfuscator_l2_norm_penalty)
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Leakage assessment $\lambda \gamma_t^*$')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'leakage_assessment.pdf'))