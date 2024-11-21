import os
import shutil
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.aes_pt_v2 import AES_PTv2, AES_PTv2_DataModule
from training_modules.supervised_classification import SupervisedClassificationModule

DEVICE_IDS = ['D1', 'D2', 'D3', 'D4']
LEARNING_RATES = np.logspace(-6, -2, 25)
EPOCHS = 100

datasets = {
    dev_id: {
        'train': AES_PTv2(
            root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_pt_v2'),
            devices=dev_id,
            countermeasure='Unprotected',
            train=True
        ), 'test': AES_PTv2(
            root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_pt_v2'),
            devices=dev_id,
            countermeasure='Unprotected',
            train=False
        )
    } for dev_id in DEVICE_IDS
}

min_val_ranks = {}

# Do a grid search to find the best learning rate for supervised classification
for dev_id in DEVICE_IDS:
    min_val_ranks[dev_id] = []
    train_dataset = datasets[dev_id]['train']
    test_dataset = datasets[dev_id]['test']
    for lr in LEARNING_RATES:
        logging_dir = os.path.join(get_trial_dir(), f'dev_id={dev_id}__learning_rate={lr}')
        if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
        else:
            data_module = AES_PTv2_DataModule(train_dataset, test_dataset)
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module = SupervisedClassificationModule(
                model_name='sca-cnn',
                optimizer_name='AdamW',
                model_kwargs={'input_shape': (1, train_dataset.timesteps_per_trace)},
                optimizer_kwargs={'lr': lr}
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
            #print(ea.Tags()['scalars'])
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

        min_val_ranks[dev_id].append(training_curves['val-rank'][-1].min())

fig, ax = plt.subplots()
ax.set_xlabel('Learning rate')
ax.set_ylabel('Min val rank')
ax.set_xscale('log')
for key, val in min_val_ranks.items():
    ax.plot(LEARNING_RATES, val, label=key)
ax.legend()
fig.savefig(os.path.join(get_trial_dir(), 'lr_sweeps.png'))
optimal_learning_rates = {key: LEARNING_RATES[np.argmin(val)] for key, val in min_val_ranks.items()}
optimal_learning_rate = np.min(optimal_learning_rates.values())