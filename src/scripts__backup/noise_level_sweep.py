# Synthetic data experiment.
#   Goal: determine how our technique scales with the 'noise level' of a dataset.
#   Let's use an infinite-sized dataset. We should vary $\lambda$. It would be good to vary the learning rates too, but that's a low-priority revision.

TIMESTEPS_PER_TRACE = 100
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 10000
MAX_EPOCHS = 100

data_vars = [0.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
lambda_vals = [1e-2, 1e-1, 1e0]
seeds = [0, 1, 2]

import os
import pickle
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

for data_var in data_vars:
    for seed in seeds:
        data_module = datasets.load(
            'synthetic-aes',
            train_batch_size=TRAIN_BATCH_SIZE,
            eval_batch_size=EVAL_BATCH_SIZE,
            train_dataset_size=int(1e5),
            val_dataset_size=int(1e4),
            test_dataset_size=int(1e4),
            aug=False,
            dataset_kwargs = {
                'timesteps_per_trace': TIMESTEPS_PER_TRACE,
                'leaking_timestep_count_1o': 1,
                'leaking_timestep_count_2o': 0,
                'shuffle_locs': 1,
                'max_no_ops': 0,
                'infinite_dataset': True, # we'll evaluate overfitting using the real datasets
                'data_var': data_var,
                'residual_var': 1.0,
                'lpf_beta': 0.9
            }
        )
        for lambda_val in lambda_vals:
            logging_dir = os.path.join(get_trial_dir(), f'data_var={data_var}__lambda={lambda_val}__seed={seed}')
            os.makedirs(logging_dir, exist_ok=True)
            training_module = AdversarialLocalizationModule(
                classifier_name='multilayer-perceptron',
                classifier_optimizer_name='AdamW',
                obfuscator_optimizer_name='AdamW',
                obfuscator_l2_norm_penalty=lambda_val,
                classifier_kwargs={'input_shape': (2, TIMESTEPS_PER_TRACE)},
                classifier_optimizer_kwargs={'lr': 1e-3},
                obfuscator_optimizer_kwargs={'lr': 1e-2},
                obfuscator_batch_size_multiplier=8
            )
            trainer = Trainer(
                max_epochs=MAX_EPOCHS,
                default_root_dir=logging_dir,
                logger=TensorBoardLogger(os.path.join(logging_dir, 'lightning_output')),
                accelerator='gpu',
                devices=1,
                enable_checkpointing=True
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            
            ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'lightning_logs', 'version_0'))
            ea.Reload()
            classifier_train_loss = extract_trace(ea.Scalars('classifier-train-loss_epoch'))
            classifier_val_loss = extract_trace(ea.Scalars('classifier-val-loss'))
            obfuscator_train_loss = extract_trace(ea.Scalars('obfuscator-train-loss_epoch'))
            obfuscator_val_loss = extract_trace(ea.Scalars('obfuscator-val-loss'))
            train_rank = extract_trace(ea.Scalars('train-rank'))
            val_rank = extract_trace(ea.Scalars('val-rank'))
            erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
            leaking_timestep = data_module.train_dataset.leaking_subbytes_cycles[0]
            leaking_prob = erasure_probs[leaking_timestep]
            nonleaking_probs = erasure_probs[np.array([x for x in np.arange(len(erasure_probs)) if x != leaking_timestep])]
            with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
                pickle.dump({
                    'classifier-train-loss': classifier_train_loss,
                    'classifier-val-loss': classifier_val_loss,
                    'obfuscator-train-loss': obfuscator_train_loss,
                    'obfuscator-val-loss': obfuscator_val_loss,
                    'train-rank': train_rank,
                    'val-rank': val_rank,
                    'erasure_probs': erasure_probs,
                    'leaking_prob': leaking_prob,
                    'nonleaking_probs': nonleaking_probs
                }, f)
            with open(os.path.join(logging_dir, 'dataset_properties.pickle'), 'wb') as f:
                pickle.dump({
                    attr_name: getattr(data_module.train_dataset, attr_name)
                    for attr_name in [
                        'leaking_subbytes_cycles', 'leaking_mask_cycles', 'leaking_masked_subbytes_cycles',
                        'per_operation_power_consumption', 'operations', 'fixed_noise_profile'
                    ]
                }, f)
        
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].plot(*classifier_train_loss, color='red', linestyle='--')
            axes[0].plot(*classifier_val_loss, color='red', linestyle='-')
            axes[1].plot(*train_rank, color='red', linestyle='--')
            axes[1].plot(*val_rank, color='red', linestyle='-')
            axes[2].plot(*obfuscator_train_loss, color='blue', linestyle='--')
            axes[2].plot(*obfuscator_val_loss, color='blue', linestyle='-')
            axes[0].set_xlabel('Step')
            axes[1].set_xlabel('Step')
            axes[2].set_xlabel('Step')
            axes[0].set_ylabel('Classifier loss')
            axes[1].set_ylabel('Correct-key rank')
            axes[2].set_ylabel('Obfuscator loss')
            fig.tight_layout()
            fig.savefig(os.path.join(logging_dir, f'data_var={data_var}__lambda={lambda_val}__seed={seed}.pdf'))
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axvline(leaking_timestep, color='orange', label='Leaking point')
            ax.plot(lambda_val*erasure_probs, color='blue', marker='.', linestyle='none', markersize=3, label='$\lambda \gamma_t^*$')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$\lambda \gamma_t^*$')
            ax.set_ylim(0, lambda_val)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(logging_dir, 'leakage_assessment.pdf'))
            
            plt.close('all')