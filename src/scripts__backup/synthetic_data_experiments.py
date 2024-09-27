# The goal of these experiments is to demonstrate that our technique works in theory in the precense of masking and hiding countermeasures.

from typing import *
import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule
from utils.calculate_snr import calculate_snr

TIMESTEPS_PER_TRACE = 100
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 5000
MAX_EPOCHS = 10000
RUN_ALL = True
RUN_SNR = True

def load_data_module(countermeasure_type: Literal['unprotected', 'no-ops', 'shuffling', 'masking']):
    data_module = datasets.load(
        'synthetic-aes',
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        aug=False,
        dataset_kwargs={
            'timesteps_per_trace': TIMESTEPS_PER_TRACE,
            'leaking_timestep_count_1o': 0 if countermeasure_type == 'masking' else 1,
            'leaking_timestep_count_2o': 1 if countermeasure_type == 'masking' else 0,
            'shuffle_locs': 5 if countermeasure_type == 'shuffling' else 1,
            'max_no_ops': 5 if countermeasure_type == 'no-ops' else 0,
            'infinite_dataset': True,
            'data_var': 1.0,
            'residual_var': 0.5,
            'lpf_beta': 0.9,
        }
    )
    return data_module

def load_training_module(countermeasure_type: Literal['unprotected', 'no-ops', 'shuffling', 'masking']):
    training_module = AdversarialLocalizationModule(
        classifier_name='multilayer-perceptron',
        classifier_optimizer_name='AdamW',
        obfuscator_optimizer_name='AdamW',
        obfuscator_l2_norm_penalty=1e0 if countermeasure_type == 'unprotected' else 1e-1 if countermeasure_type in ['shuffling', 'masking'] else 5e-2,
        classifier_kwargs={'input_shape': (2, TIMESTEPS_PER_TRACE), 'xor_output': countermeasure_type == 'masking'},
        classifier_optimizer_kwargs={'lr': 1e-5},
        obfuscator_optimizer_kwargs={'lr': 2e-4},
        obfuscator_batch_size_multiplier=8
    )
    return training_module

for countermeasure_type in ['no-ops', 'masking', 'shuffling', 'unprotected']:
    logging_dir = os.path.join(get_trial_dir(), countermeasure_type)
    os.makedirs(logging_dir, exist_ok=True)
    data_module = load_data_module(countermeasure_type)
    if RUN_ALL:
        training_module = load_training_module(countermeasure_type)
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            default_root_dir=logging_dir,
            logger=TensorBoardLogger(os.path.join(logging_dir, 'lightning_output')),
            accelerator='gpu',
            devices=1,
            enable_checkpointing=True
        )
        trainer.fit(training_module, datamodule=data_module)
        
        ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'lightning_logs', 'version_0'))
        ea.Reload()
        #print(ea.Tags()['scalars'])
        classifier_train_loss = ea.Scalars('classifier-train-loss_epoch')
        classifier_val_loss = ea.Scalars('classifier-val-loss')
        obfuscator_train_loss = ea.Scalars('obfuscator-train-loss_epoch')
        obfuscator_val_loss = ea.Scalars('obfuscator-val-loss')
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        #axes[0].plot(
        #    [x.step for x in classifier_train_loss], [x.value for x in classifier_train_loss],
        #    color='red', linestyle='--'
        #)
        axes[0].plot(
            [x.step for x in classifier_val_loss], [x.value for x in classifier_val_loss],
            color='red', linestyle='-', label='classifier'
        )
        tax0 = axes[0].twinx()
        #tax0.plot(
        #    [x.step for x in obfuscator_train_loss], [x.value for x in obfuscator_train_loss],
        #    color='blue', linestyle='--'
        #)
        tax0.plot(
            [x.step for x in obfuscator_val_loss], [x.value for x in obfuscator_val_loss],
            color='blue', linestyle='-', label='obfuscator'
        )
        dataset = data_module.train_dataset
        if countermeasure_type in ['unprotected', 'shuffling']:
            for cycle in dataset.leaking_subbytes_cycles:
                axes[1].axvline(cycle, color='blue', label='ground truth')
        elif countermeasure_type == 'no-ops':
            for cycle in dataset.leaking_subbytes_cycles:
                axes[1].axvspan(cycle, cycle+dataset.max_no_ops, color='blue', alpha=0.5, label='ground truth')
        elif countermeasure_type == 'masking':
            for cycle in dataset.leaking_mask_cycles:
                axes[1].axvline(cycle, color='red', label='ground truth (mask)')
            for cycle in dataset.leaking_masked_subbytes_cycles:
                axes[1].axvline(cycle, color='purple', label='ground truth (masked subbytes)')
        obfuscation_weights = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
        axes[1].plot(obfuscation_weights, color='blue', linestyle='none', marker='.', label='Obfuscation weight')
        axes[1].set_ylim(0, 1)
        axes[0].set_xlabel('Training step')
        axes[1].set_xlabel('Timestep')
        axes[0].set_ylabel('Loss (classifier)')
        tax0.set_ylabel('Loss (obfuscator)')
        axes[0].legend()
        tax0.legend()
        axes[1].legend()
        axes[1].set_ylabel('Obfuscation weight')
        fig.tight_layout()
        fig.savefig(os.path.join(logging_dir, 'result.pdf'))
    if RUN_SNR:
        data_module.setup('fit')
        dataset = data_module.train_dataset
        snr = calculate_snr(
            dataset,
            targets=['subbytes', 'mask', 'masked_subbytes'] if countermeasure_type == 'masking' else 'subbytes'
        )
        fig, ax = plt.subplots(figsize=(4, 4))
        if countermeasure_type in ['unprotected', 'shuffling']:
            for cycle in dataset.leaking_subbytes_cycles:
                ax.axvline(cycle, color='blue', label='ground truth')
        elif countermeasure_type == 'no-ops':
            for cycle in dataset.leaking_subbytes_cycles:
                ax.axvspan(cycle, cycle+dataset.max_no_ops, color='blue', alpha=0.5, label='ground truth')
        elif countermeasure_type == 'masking':
            for cycle in dataset.leaking_mask_cycles:
                ax.axvline(cycle, color='red', label='ground truth (mask)')
            for cycle in dataset.leaking_masked_subbytes_cycles:
                ax.axvline(cycle, color='purple', label='ground truth (masked subbytes)')
        if countermeasure_type == 'masking':
            ax.plot(snr['mask'], color='red', linestyle='none', marker='.', label='mask')
            ax.plot(snr['masked_subbytes'], color='purple', linestyle='none', marker='.', label='masked subbytes')
            ax.plot(snr['subbytes'], color='blue', linestyle='none', marker='.', label='subbytes')
        else:
            ax.plot(snr['subbytes'], color='blue', linestyle='none', marker='.', label='subbytes')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Signal-noise ratio')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(logging_dir, 'snr.pdf'))