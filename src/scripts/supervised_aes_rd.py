import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.aes_rd import AES_RD, to_key_preds
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from utils.metrics.rank import accumulate_ranks
import datasets
from training_modules.supervised_classification import SupervisedClassificationModule

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_rd')
RUN_STATISTICAL_EVALUATIONS = False
RUN_LR_SWEEP = False
EVAL_LR_SWEEP = True

profiling_dataset = AES_RD(
    root=ROOT, train=True
)
attack_dataset = AES_RD(
    root=ROOT, train=False
)

if RUN_STATISTICAL_EVALUATIONS:
    cpa = calculate_cpa(profiling_dataset, targets=['subbytes'])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cpa[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('CPA')
    ax.set_title('$Y$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'cpa.pdf'))

    snr = calculate_snr(profiling_dataset, targets=['subbytes'])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(snr[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SNR')
    ax.set_title('$Y$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'snr.pdf'))

    sosd = calculate_sosd(profiling_dataset, targets=['subbytes'])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(sosd[('subbytes', None)].squeeze(), color='blue', linestyle='none', marker='.', markersize=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SOSD')
    ax.set_title('$Y$')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'sosd.pdf'))

data_module = datasets.load('aes-rd', train_batch_size=256, eval_batch_size=2048, root=ROOT)
learning_rates = np.logspace(-7, -3, 20)
if RUN_LR_SWEEP:
    for learning_rate in learning_rates:
        logging_dir = os.path.join(get_trial_dir(), f'lr_{learning_rate}')
        training_module = SupervisedClassificationModule(
            model_name='sca-cnn',
            optimizer_name='AdamW',
            model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace)},
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
        logging_dir = os.path.join(get_trial_dir(), f'lr_{learning_rate}')
        training_module = SupervisedClassificationModule.load_from_checkpoint(
            os.path.join(logging_dir, 'final_checkpoint.ckpt'),
            model_name='sca-cnn',
            optimizer_name='AdamW',
            model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace)},
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
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'rank_over_time.pdf'))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(learning_rates, final_ranks, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Mean correct key rank')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), 'lr_sweep.pdf'))