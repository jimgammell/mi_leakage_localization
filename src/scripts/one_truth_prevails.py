import os
import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from lightning.pytorch.callbacks import ModelCheckpoint

from _common import *
from datasets.ascadv1 import ASCADv1, ASCADv1_DataModule
from training_modules.supervised_classification import SupervisedClassificationModule
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map

LEARNING_RATES = np.logspace(-6, -2, 25)
EPOCHS = 100

min_val_ranks = []

# Do a grid search to find the best learning rate for supervised classification
for learning_rate in LEARNING_RATES:
    logging_dir = os.path.join(get_trial_dir(), f'learning_rate={learning_rate}')
    if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
            training_curves = pickle.load(f)
    else:
        data_module = ASCADv1_DataModule(
            root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
            dataset_kwargs={'variable_keys': True}
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

# Train a standard + ALL-style classifier with the optimal learning rate and early stopping based on validation rank
logging_dir = os.path.join(get_trial_dir(), 'supervised_classifier')
data_module = ASCADv1_DataModule(
    root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
    dataset_kwargs={'variable_keys': True}
)
data_module.setup('')
if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
        training_curves = pickle.load(f)
else:
    if os.path.exists(logging_dir):
        shutil.rmtree(logging_dir)
    os.makedirs(logging_dir)
    training_module = SupervisedClassificationModule(
        model_name='sca-cnn',
        optimizer_name='AdamW',
        model_kwargs={'input_shape': (1, data_module.train_dataset.dataset.timesteps_per_trace)},
        optimizer_kwargs={'lr': optimal_learning_rate}
    )
    checkpoint = ModelCheckpoint(
        filename='best',
        monitor='val-rank',
        save_top_k=1,
        mode='min'
    )
    trainer = Trainer(
        max_epochs=EPOCHS,
        default_root_dir=logging_dir,
        accelerator='gpu',
        devices=1,
        logger=TensorBoardLogger(logging_dir, name='lightning_output'),
        callbacks=[checkpoint]
    )
    trainer.fit(training_module, datamodule=data_module)
    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
    ea.Reload()
    training_curves = {
        key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-rank', 'val-rank']
    }
    training_module = SupervisedClassificationModule.load_from_checkpoint(
        os.path.join(logging_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt'),
        model_name='sca-cnn',
        optimizer_name='AdamW',
        model_kwargs={'input_shape': (1, data_module.train_dataset.dataset.timesteps_per_trace)},
        optimizer_kwargs={'lr': optimal_learning_rate}
    )
    training_curves.update({'early_stop_epoch': training_module.current_epoch})
    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
        pickle.dump(training_curves, f)
    torch.save(training_module.model.state_dict(), os.path.join(logging_dir, 'standard_classifier_state_dict.pth'))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(*training_curves['train-loss'], color='red', linestyle='--')
axes[0].plot(*training_curves['val-loss'], color='red', linestyle='-')
axes[1].plot(*training_curves['train-rank'], color='red', linestyle='--')
axes[1].plot(*training_curves['val-rank'], color='red', linestyle='-')
axes[0].axvline(training_curves['early_stop_epoch']*len(data_module.train_dataloader()), color='red', linestyle='--')
axes[1].axvline(training_curves['early_stop_epoch']*len(data_module.train_dataloader()), color='red', linestyle='--')
axes[0].set_xlabel('Training step')
axes[0].set_ylabel('Loss')
axes[1].set_xlabel('Training step')
axes[1].set_ylabel('Rank')
fig.tight_layout()
fig.savefig(os.path.join(logging_dir, 'training_curves.png'))
plt.close('all')

if not os.path.exists(os.path.join(get_trial_dir(), 'nn_explainability_results.pickle')):
    training_module = SupervisedClassificationModule(
        model_name='sca-cnn',
        optimizer_name='AdamW',
        model_kwargs={'input_shape': (1, data_module.train_dataset.dataset.timesteps_per_trace)},
        optimizer_kwargs={'lr': optimal_learning_rate}
    )
    training_module.model.load_state_dict(torch.load(os.path.join(logging_dir, 'standard_classifier_state_dict.pth'), map_location='cpu', weights_only=True))
    ablation_assessment = compute_feature_ablation_map(training_module, data_module.train_dataset.dataset)
    gradvis_assessment = compute_gradvis(training_module, data_module.train_dataset.dataset)
    input_x_grad_assessment = compute_input_x_gradient(training_module, data_module.train_dataset.dataset)
    with open(os.path.join(get_trial_dir(), 'nn_explainability_results.pickle'), 'wb') as f:
        pickle.dump({'ablation': ablation_assessment, 'gradvis': gradvis_assessment, 'input_x_gradient': input_x_grad_assessment}, f)
else:
    with open(os.path.join(get_trial_dir(), 'nn_explainability_results.pickle'), 'rb') as f:
        res = pickle.load(f)
        ablation_assessment = res['ablation']
        gradvis_assessment = res['gradvis']
        input_x_grad_assessment = res['input_x_gradient']
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(ablation_assessment.squeeze(), color='blue', linestyle='none', marker='.', rasterized=True)
axes[1].plot(gradvis_assessment.squeeze(), color='blue', linestyle='none', marker='.', rasterized=True)
axes[2].plot(input_x_grad_assessment.squeeze(), color='blue', linestyle='none', marker='.', rasterized=True)
for ax in axes:
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Estimated leakage of $X_t$')
axes[0].set_title('Ablation')
axes[1].set_title('GradVis')
axes[2].set_title('Input $*$ Gradient')
fig.tight_layout()
fig.savefig(os.path.join(get_trial_dir(), 'nn_explainability_results.pdf'))