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
from datasets.ascadv1 import ASCADv1, to_key_preds
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer
from utils.template_attack import TemplateAttack
from utils.metrics.rank import _process_dataloader_for_rank_accumulation, _accumulate_ranks

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1')
RUN_SWEEPS = True
EVAL_SWEEPS = False

profiling_dataset = ASCADv1(
    root=ROOT, train=True
)
attack_dataset = ASCADv1(
    root=ROOT, train=False
)

data_module = datasets.load('ascadv1f', train_batch_size=256, eval_batch_size=2048, root=ROOT)
learning_rates = [1e-5, 1e-6]
lambdas = [1e-2, 1e-1, 1e0]
if RUN_SWEEPS:
    for lbda in lambdas:
        for learning_rate in learning_rates:
            logging_dir = os.path.join(get_trial_dir(), f'lambda_{lbda}__lr_{learning_rate}')
            training_module = DiscreteAdversarialLocalizationTrainer(
                classifier_name='sca-cnn',
                classifier_kwargs={'input_shape': (2, profiling_dataset.timesteps_per_trace), 'head_kwargs': {'xor_output': False}},
                classifier_optimizer_name='AdamW',
                obfuscator_optimizer_name='AdamW',
                obfuscator_l2_norm_penalty=lbda,
                log_likelihood_baseline_ema=0.9,
                classifier_learning_rate=learning_rate,
                obfuscator_learning_rate=1e-3,
                obfuscator_batch_size_multiplier=8,
                normalize_erasure_probs_for_classifier=True
            )
            trainer = Trainer(
                max_epochs=1000,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            
            ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
            ea.Reload()
            classifier_train_loss = ea.Scalars('classifier-train-loss_epoch')
            classifier_val_loss = ea.Scalars('classifier-val-loss')
            obfuscator_train_loss = ea.Scalars('obfuscator-train-loss_epoch')
            obfuscator_val_loss = ea.Scalars('obfuscator-val-loss')
            train_rank = ea.Scalars('train-rank')
            val_rank = ea.Scalars('val-rank')
            
            fig, axes = plt.subplots(1, 3, figsize=(3*4, 1*4))
            axes[0].plot([x.step for x in classifier_train_loss], [x.value for x in classifier_train_loss], color='red', linestyle='--')
            axes[0].plot([x.step for x in classifier_val_loss], [x.value for x in classifier_val_loss], color='red', linestyle='-')
            axes[1].plot([x.step for x in train_rank], [x.value for x in train_rank], color='red', linestyle='--')
            axes[1].plot([x.step for x in val_rank], [x.value for x in val_rank], color='red', linestyle='-')
            axes[2].plot([x.step for x in obfuscator_train_loss], [x.value for x in obfuscator_train_loss], color='blue', linestyle='--')
            axes[2].plot([x.step for x in obfuscator_val_loss], [x.value for x in obfuscator_val_loss], color='blue', linestyle='-')
            axes[0].set_xlabel('Step')
            axes[1].set_xlabel('Step')
            axes[2].set_xlabel('Step')
            axes[0].set_ylabel('Classifier loss')
            axes[1].set_ylabel('Classifier mean correct-key rank')
            axes[2].set_ylabel('Obfuscator loss')
            fig.tight_layout()
            fig.savefig(os.path.join(logging_dir, 'training_curves.pdf'))
            
            fig, ax = plt.subplots(figsize=(4, 4))
            erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
            ax.plot(erasure_probs, color='blue', linestyle='none', marker='.')
            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(logging_dir, 'erasure_probs.pdf'))

if EVAL_SWEEPS:
    for lbda in lambdas:
        ep_fig, ep_axes = plt.subplots(2, len(learning_rates)//2, figsize=(2*len(learning_rates)//2, 4), sharex=True, sharey=True)
        rot_fig, rot_axes = plt.subplots(2, len(learning_rates)//2, figsize=(2*len(learning_rates)//2, 4), sharex=True, sharey=True)
        loss_fig, loss_axes = plt.subplots(2, len(learning_rates)//2, figsize=(2*len(learning_rates)//2, 4), sharex=True, sharey=True)
        mean_ranks = [[], [], []]
        for learning_rate, ep_ax, rot_ax, loss_ax in zip(learning_rates, ep_axes.flatten(), rot_axes.flatten(), loss_axes.flatten()):
            try:
                logging_dir = os.path.join(get_trial_dir(), f'lambda_{lbda}__lr_{learning_rate}')
                training_module = DiscreteAdversarialLocalizationTrainer.load_from_checkpoint(
                    os.path.join(logging_dir, 'final_checkpoint.ckpt'),
                    classifier_name='sca-cnn',
                    classifier_kwargs={'input_shape': (2, profiling_dataset.timesteps_per_trace)},
                    classifier_optimizer_name='AdamW',
                    obfuscator_optimizer_name='AdamW',
                    obfuscator_l2_norm_penalty=lbda,
                    log_likelihood_baseline_ema=0.9,
                    classifier_learning_rate=2e-5,
                    obfuscator_learning_rate=learning_rate,
                    obfuscator_batch_size_multiplier=8
                )
                trainer = Trainer(
                    max_epochs=1000,
                    default_root_dir=logging_dir,
                    accelerator='gpu',
                    devices=1,
                    logger=TensorBoardLogger(logging_dir, name='lightning_output')
                )
                trainer.validate(training_module, datamodule=data_module)
                erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
                ep_ax.plot(training_module.obfuscator_l2_norm_penalty*erasure_probs, color='blue', linestyle='none', marker='.', markersize=2)
                ep_ax.set_ylim(0, training_module.obfuscator_l2_norm_penalty)
                ep_ax.set_xlabel('$t$')
                ep_ax.set_ylabel('$\lambda \gamma_t^*$')
                ep_ax.set_title(f'Learning rate: {learning_rate:.2e}')
            except:
                continue
                
            ranking = erasure_probs.argsort()[::-1]
            for x in range(3):
                template_attack = TemplateAttack(ranking[10*x:10*(x+1)])
                template_attack.profile(profiling_dataset)
                color = ['blue', 'red', 'green'][x]
                predictions = template_attack.get_predictions(attack_dataset, n_repetitions=1000, n_traces=len(attack_dataset), int_var_to_key_fn=to_key_preds)
                indices = np.stack([np.random.choice(len(attack_dataset), len(attack_dataset), replace=False) for _ in range(1000)])
                rv = _process_dataloader_for_rank_accumulation(training_module, metadata_keys=['plaintext', 'offset'], constants=['mask'], skip_forward_passes=True)
                _, keys, args, constants = rv
                args = np.stack(args, axis=1)
                rank_over_time = _accumulate_ranks(predictions, keys, args, constants, indices, int_var_to_key_fn=to_key_preds)
                rot_ax.plot(np.arange(1, rank_over_time.shape[1]+1), np.median(rank_over_time, axis=0), color=color)
                rot_ax.fill_between(np.arange(1, rank_over_time.shape[1]+1), np.percentile(rank_over_time, 25, axis=0), np.percentile(rank_over_time, 75, axis=0), color=color, alpha=0.5)
                rot_ax.set_xlabel('Traces seen')
                rot_ax.set_ylabel('Correct key rank')
                rot_ax.set_xscale('log')
                mean_ranks[x].append(np.mean(rank_over_time))
            
        ep_fig.tight_layout()
        ep_fig.savefig(os.path.join(get_trial_dir(), f'erasure_probs_sweep__lambda={lbda:.2e}.pdf'))
        rot_fig.tight_layout()
        rot_fig.savefig(os.path.join(get_trial_dir(), f'ta_rot__lambda={lbda:.2e}.pdf'))
        
        fig, ax = plt.subplots(figsize=(4, 4))
        for _mean_ranks in mean_ranks:
            ax.plot(learning_rates[:len(_mean_ranks)], _mean_ranks)
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Mean rank')
        fig.savefig(os.path.join(get_trial_dir(), f'mean_ranks__lambda={lbda:.2e}.pdf'))