import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
import datasets
from utils.metrics.rank import accumulate_ranks
from utils.aes import subbytes_to_keys
from training_modules.supervised_classification import SupervisedClassificationModule

RUN_LR_SWEEP = True
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
learning_rates = [x*1e-6 for x in range(1, 10)] + [x*1e-5 for x in range(1, 10)] + [1e-4]
if RUN_LR_SWEEP:
    for learning_rate in learning_rates:
        logging_dir = os.path.join(get_trial_dir(), f'learning_rate={learning_rate}')
        training_module = SupervisedClassificationModule(
            model_name='multilayer-perceptron',
            optimizer_name='AdamW',
            model_kwargs={'input_shape': (1, TIMESTEPS_PER_TRACE)},
            optimizer_kwargs={'lr': learning_rate}
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
            key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-rank', 'val-rank']
        }
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
            pickle.dump(training_curves, f)
        rank_over_time = accumulate_ranks(training_module, int_var_to_key_fn=subbytes_to_keys, args=['plaintext'], traces_per_attack=np.inf)
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