import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from lightning.pytorch.callbacks import ModelCheckpoint

from _common import *
from datasets.ascadv1 import ASCADv1, to_key_preds
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from utils.metrics.rank import accumulate_ranks
import datasets
from training_modules.supervised_classification import SupervisedClassificationModule

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization','downloads', 'ascadv1')
RUN_LR_SWEEP = True
STEP_COUNT = 10000

profiling_dataset = ASCADv1(
    root=ROOT,
    train=True
)
attack_dataset = ASCADv1(
    root=ROOT,
    train=False
)

data_module = datasets.load('ascadv1f', train_batch_size=1024, eval_batch_size=10000, root=ROOT)
learning_rates = [6e-5] #[x*1e-6 for x in range(1, 10)] + [x*1e-5 for x in range(1, 10)] + [1e-4]
weight_decays = [0.0] #[0.0, 1e-2]
additive_noises = [0.25] #[0.0, 0.25]
seeds = [0, 1, 2, 3, 4]
if RUN_LR_SWEEP:
    for seed in seeds:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                for additive_noise in additive_noises:
                    logging_dir = os.path.join(get_trial_dir(), f'learning_rate={learning_rate}__weight_decay={weight_decay}__additive_noise={additive_noise}__seed={seed}')
                    training_module = SupervisedClassificationModule(
                        model_name='sca-cnn',
                        optimizer_name='AdamW',
                        model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace)},
                        optimizer_kwargs={'lr': learning_rate, 'weight_decay': weight_decay},
                        additive_noise_augmentation=additive_noise
                    )
                    checkpoint = ModelCheckpoint(
                        filename='best_model',
                        monitor='val-rank',
                        save_top_k=1,
                        mode='min'
                    )
                    trainer = Trainer(
                        max_epochs=int(STEP_COUNT*1024/len(profiling_dataset)),
                        default_root_dir=logging_dir,
                        accelerator='gpu',
                        devices=1,
                        logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                        callbacks=[checkpoint]
                    )
                    trainer.fit(training_module, datamodule=data_module)
                    trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
                    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
                    ea.Reload()
                    training_curves = {
                        key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-rank', 'val-rank']
                    }
                    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
                        pickle.dump(training_curves, f)
                    rank_over_time = accumulate_ranks(training_module, int_var_to_key_fn=to_key_preds, args=['plaintext'], traces_per_attack=np.inf)
                    with open(os.path.join(logging_dir, 'rank_over_time.pickle'), 'wb') as f:
                        pickle.dump(rank_over_time, f)
                        
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].plot(*training_curves['train-loss'], color='red', linestyle='--')
                    axes[0].plot(*training_curves['val-loss'], color='red', linestyle='-')
                    axes[1].plot(*training_curves['train-rank'], color='red', linestyle='--')
                    axes[1].plot(*training_curves['val-rank'], color='red', linestyle='-')
                    axes[0].set_xlabel('Training step')
                    axes[0].set_ylabel('Loss')
                    axes[1].set_xlabel('Training step')
                    axes[1].set_ylabel('Rank')
                    fig.tight_layout()
                    fig.savefig(os.path.join(logging_dir, 'training_curves.png'))
                    
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.plot(np.median(rank_over_time, axis=0), color='blue')
                    ax.fill_between(np.arange(1, rank_over_time.shape[-1]+1), np.percentile(rank_over_time, 25, axis=0), np.percentile(rank_over_time, 75, axis=0), color='blue', alpha=0.25)
                    ax.set_xlabel('Traces seen')
                    ax.set_ylabel('Correct key rank')
                    ax.set_xscale('log')
                    fig.tight_layout()
                    fig.savefig(os.path.join(logging_dir, 'rank_over_time.png'))