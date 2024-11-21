import os
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.ascadv1 import ASCADv1, ASCADv1_DataModule
from training_modules.supervised_classification import SupervisedClassificationModule

LEARNING_RATES = np.logspace(-6, -2, 25)
EPOCHS = 100

min_val_ranks = []
kwargs = dict(
    root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
    variable_keys=True
)

for learning_rate in LEARNING_RATES:
    logging_dir = os.path.join(get_trial_dir(), f'learning_rate={learning_rate}')
    if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
            training_curves = pickle.load(f)
    else:
        data_module = ASCADv1_DataModule(
            root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
            dataset_kwargs={'variable': True}
        )
        data_module.setup('')
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
        training_module = SupervisedClassificationModule(
            model_name='sca-cnn',
            optimizer_name='AdamW',
            model_kwargs={'input_shape': (1, data_module.train_dataset.dataset.timesteps_per_trace)},
            optimizer_kwargs={'lr': learning_rate}
        )
        trainer = Trainer(
            max_epochs=EPOCHS,
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
    plt.close('all')
    min_val_ranks.append(training_curves['val-rank'][-1].min())

fig, ax = plt.subplots()
ax.set_xlabel('Learning rate')
ax.set_ylabel('Min val rank')
ax.set_xscale('log')
ax.plot(LEARNING_RATES, min_val_ranks, color='blue')
fig.savefig(os.path.join(get_trial_dir(), 'lr_sweeps.png'))
optimal_learning_rate = LEARNING_RATES[np.argmin(min_val_ranks)]