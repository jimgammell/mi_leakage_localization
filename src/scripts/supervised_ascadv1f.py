import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.ascadv1 import ASCADv1, to_key_preds
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from utils.metrics.rank import accumulate_ranks
import datasets
from training_modules.supervised_classification import SupervisedClassificationModule

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization','downloads', 'ascadv1')
RUN_STATISTICAL_EVALUATIONS = True
RUN_LR_SWEEP = False
EVAL_LR_SWEEP = False

profiling_dataset = ASCADv1(
    root=ROOT,
    train=True
)
attack_dataset = ASCADv1(
    root=ROOT,
    train=False
)

if RUN_STATISTICAL_EVALUATIONS:
    cpa = calculate_cpa(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    axes[0].plot(cpa[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[1].plot(cpa[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[2].plot(cpa[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[3].plot(cpa[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[4].plot(cpa[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    for ax in axes:
        ax.set_xlabel('Timestep')
        ax.set_ylabel('CPA')
    axes[0].set_title('$Y$')
    axes[1].set_title('$Y \oplus R$')
    axes[2].set_title('$R$')
    axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
    axes[4].set_title('$R_{\mathrm{out}}$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'cpa.pdf'), **SAVEFIG_KWARGS)

    snr = calculate_snr(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    axes[0].plot(snr[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[1].plot(snr[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[2].plot(snr[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[3].plot(snr[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[4].plot(snr[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    for ax in axes:
        ax.set_xlabel('Timestep')
        ax.set_ylabel('SNR')
    axes[0].set_title('$Y$')
    axes[1].set_title('$Y \oplus R$')
    axes[2].set_title('$R$')
    axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
    axes[4].set_title('$R_{\mathrm{out}}$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'), **SAVEFIG_KWARGS)

    sosd = calculate_sosd(profiling_dataset, targets=['subbytes', 'subbytes__r', 'r', 'subbytes__r_out', 'r_out'], bytes=2)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    axes[0].plot(sosd[('subbytes', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[1].plot(sosd[('subbytes__r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[2].plot(sosd[('r', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[3].plot(sosd[('subbytes__r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    axes[4].plot(sosd[('r_out', 2)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
    for ax in axes:
        ax.set_xlabel('Timestep')
        ax.set_ylabel('SOSD')
    axes[0].set_title('$Y$')
    axes[1].set_title('$Y \oplus R$')
    axes[2].set_title('$R$')
    axes[3].set_title('$Y \oplus R_{\mathrm{out}}$')
    axes[4].set_title('$R_{\mathrm{out}}$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'sosd.pdf'), **SAVEFIG_KWARGS)

data_module = datasets.load('ascadv1f', train_batch_size=256, eval_batch_size=2048, root=ROOT)
learning_rates = np.logspace(-7, -3, 20)
if RUN_LR_SWEEP:
    for learning_rate in learning_rates:
        logging_dir = os.path.join(get_trial_dir(), f'lr_{learning_rate}')
        training_module = SupervisedClassificationModule(
            model_name='sca-cnn',
            optimizer_name='AdamW',
            model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace), 'head_kwargs': {'xor_output': True}},
            optimizer_kwargs={'lr': learning_rate}
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
        train_loss = ea.Scalars('train-loss')
        val_loss = ea.Scalars('val-loss')
        train_rank = ea.Scalars('train-rank')
        val_rank = ea.Scalars('val-rank')
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
            pickle.dump({
                'train_loss': train_loss, 'val_loss': val_loss, 'train_rank': train_rank, 'val_rank': val_rank
            }, f)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(
            [x.step for x in train_loss], [x.value for x in train_loss], color='red', linestyle='--', label='train'
        )
        axes[0].plot(
            [x.step for x in val_loss], [x.value for x in val_loss], color='red', linestyle='-', label='val'
        )
        axes[1].plot(
            [x.step for x in train_rank], [x.value for x in train_rank], color='red', linestyle='--', label='train'
        )
        axes[1].plot(
            [x.step for x in val_rank], [x.value for x in val_rank], color='red', linestyle='-', label='val'
        )
        axes[0].set_xlabel('Training step')
        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Training step')
        axes[1].set_ylabel('Rank')
        fig.savefig(os.path.join(logging_dir, 'loss_curves.pdf'))

if EVAL_LR_SWEEP:
    fig, axes = plt.subplots(4, 5, figsize=(2*5, 2*4), sharey=True)
    final_ranks = []
    for (learning_rate, ax) in zip(learning_rates, axes.flatten()):
        try:
            logging_dir = os.path.join(get_trial_dir(), f'lr_{learning_rate}')
            training_module = SupervisedClassificationModule.load_from_checkpoint(
                os.path.join(logging_dir, 'final_checkpoint.ckpt'),
                model_name='sca-cnn',
                optimizer_name='AdamW',
                model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace), 'head_kwargs': {'xor_output': True}},
                optimizer_kwargs={'lr': learning_rate}
            )
            trainer = Trainer(
                max_epochs=1000,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.test(training_module, datamodule=data_module)
            with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
            val_rank = training_curves['val_rank']
            final_rank = [x.value for x in val_rank][-1]
            if final_rank == -1:
                final_rank = 255
            final_ranks.append(final_rank)
            rank_over_time = accumulate_ranks(training_module, int_var_to_key_fn=to_key_preds)
            ax.plot(np.arange(1, rank_over_time.shape[1]+1), np.median(rank_over_time, axis=0), color='blue')
            ax.fill_between(
                np.arange(1, rank_over_time.shape[1]+1), np.percentile(rank_over_time, 25, axis=0), np.percentile(rank_over_time, 75, axis=0),
                color='blue', alpha=0.25
            )
            ax.set_xlabel('Traces seen')
            ax.set_ylabel('Correct-key rank')
            ax.set_xscale('log')
            ax.set_title(f'Learning rate: {learning_rate:.02e}')
        except:
            continue
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'rank_over_time.pdf'))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(learning_rates[:len(final_ranks)], final_ranks, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Mean correct key rank')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'lr_sweep.pdf'))