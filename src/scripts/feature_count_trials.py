import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))
import shutil
import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import torch
from torch import nn
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tensorboard.backend.event_processing import event_accumulator

from common import *
from datasets.two_spirals import TwoSpiralsDataset, TwoSpiralsDataModule
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from training_modules.supervised_classification import SupervisedClassificationModule
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer as ALLTrainer
from utils.localization_via_interpretability import compute_feature_ablation_map, compute_gradvis, compute_input_x_gradient

set_verbosity(True)

def extract_trace(trace):
    x = np.array([u.step for u in trace])
    y = np.array([u.value for u in trace])
    return (x, y)

DATASET_SIZE = 10000
TOTAL_FEATURE_COUNT = 100
TRAINING_EPOCHS = 100
L2_NORM_PENALTY = np.log(2)
SAVE_DIR = os.path.join(OUTPUT_DIR, 'feature_count_trials')
os.makedirs(SAVE_DIR, exist_ok=True)

easy_feature_counts = np.array([1, 2, 4, 8, 16, 32, 64])
seed_count = 1
if not os.path.exists(os.path.join(SAVE_DIR, 'results.pickle')):
    snr_vals, sosd_vals, cpa_vals, gradvis_vals, inputxgrad_vals, ablation_vals, all_vals = map(
        lambda _: np.full((len(easy_feature_counts), seed_count, TOTAL_FEATURE_COUNT), np.nan, dtype=float), range(7)
    )
    for easy_feature_count_idx, easy_feature_count in enumerate(easy_feature_counts):
        for seed in range(seed_count):
            dataset_kwargs = {
                'datapoint_count': DATASET_SIZE,
                'random_feature_count': TOTAL_FEATURE_COUNT - easy_feature_count,
                'easy_feature_count': easy_feature_count,
                'no_hard_feature': True,
                'easy_feature_sigma': 100.
            }
            train_dataset = TwoSpiralsDataset(**dataset_kwargs)
            val_dataset = TwoSpiralsDataset(**dataset_kwargs)
            mean = train_dataset.x.mean(axis=0)
            std = train_dataset.x.std(axis=0)
            transform = lambda x: (x - mean) / std
            train_dataset.transform = val_dataset.transform = transform
            
            train_dataset.ret_mdata = True
            snr = calculate_snr(train_dataset, train_dataset, ['target'])[('target', None)]
            snr_vals[easy_feature_count_idx, seed, :] = snr
            sosd = calculate_sosd(train_dataset, train_dataset, ['target'])[('target', None)]
            sosd_vals[easy_feature_count_idx, seed, :] = sosd
            cpa = calculate_cpa(train_dataset, train_dataset, ['target'])[('target', None)]
            cpa_vals[easy_feature_count_idx, seed, :] = cpa
            
            data_module = TwoSpiralsDataModule(train_dataset=train_dataset, eval_dataset=val_dataset, train_batch_size=256)
            logging_dir = os.path.join(SAVE_DIR, f'supervised_lightning_output__count_{easy_feature_count}__seed_{seed}')
            if os.path.exists(logging_dir):
                training_module = SupervisedClassificationModule.load_from_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'),
                    model_name='multilayer-perceptron',
                    optimizer_name='AdamW',
                    model_kwargs={'input_shape': (train_dataset.timesteps_per_trace,), 'output_classes': 2},
                    optimizer_kwargs={'lr': 2e-4},
                    #lr_scheduler_name='CosineDecayLRSched',
                    additive_noise_augmentation=0.0
                )
                data_module.setup('')
            else:
                training_module = SupervisedClassificationModule(
                    model_name='multilayer-perceptron',
                    optimizer_name='AdamW',
                    model_kwargs={'input_shape': (train_dataset.timesteps_per_trace,), 'output_classes': 2},
                    optimizer_kwargs={'lr': 2e-4},
                    #lr_scheduler_name='CosineDecayLRSched',
                    additive_noise_augmentation=0.0
                )
                checkpoint = ModelCheckpoint(
                    filename='best_model',
                    monitor='val-loss',
                    save_top_k=1,
                    mode='max'
                )
                trainer = Trainer(
                    max_epochs=TRAINING_EPOCHS,
                    default_root_dir=logging_dir,
                    accelerator='gpu',
                    devices=1,
                    logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                    callbacks=[checkpoint]
                )
                trainer.fit(training_module, datamodule=data_module)
                trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            gradvis_val = compute_gradvis(training_module, data_module.train_dataset)
            gradvis_vals[easy_feature_count_idx, seed, :] = gradvis_val
            inputxgrad_val = compute_input_x_gradient(training_module, data_module.train_dataset)
            inputxgrad_vals[easy_feature_count_idx, seed, :] = inputxgrad_val
            ablation_val = compute_feature_ablation_map(training_module, data_module.train_dataset)
            ablation_vals[easy_feature_count_idx, seed, :] = ablation_val
            
            ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
            ea.Reload()
            training_curves = {
                key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-acc', 'val-acc']
            }
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].plot(*training_curves['train-loss'], color='blue', linestyle='--')
            axes[0].plot(*training_curves['val-loss'], color='blue', linestyle='-')
            axes[1].plot(*training_curves['train-acc'], color='blue', linestyle='--')
            axes[1].plot(*training_curves['val-acc'], color='blue', linestyle='-')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Loss')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Accuracy')
            axes[0].set_yscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(SAVE_DIR, logging_dir, 'training_curves.png'))
            plt.close('all')
            
    
            logging_dir = os.path.join(SAVE_DIR, f'all_lightning_output__count_{easy_feature_count}__seed_{seed}')
            if os.path.exists(logging_dir):
                training_module = ALLTrainer.load_from_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'),
                    classifier_name='multilayer-perceptron',
                    classifier_kwargs={'input_shape': (train_dataset.timesteps_per_trace,), 'output_classes': 2},
                    classifier_optimizer_name='AdamW',
                    obfuscator_optimizer_name='AdamW',
                    obfuscator_l2_norm_penalty=L2_NORM_PENALTY,
                    split_training_steps=TRAINING_EPOCHS*len(data_module.train_dataloader()),
                    classifier_optimizer_kwargs={'lr': 2e-4},
                    obfuscator_optimizer_kwargs={'lr': 1e-2},
                    #classifier_lr_scheduler_name='CosineDecayLRSched',
                    #obfuscator_lr_scheduler_name='CosineDecayLRSched',
                    obfuscator_batch_size_multiplier=8,
                    normalize_erasure_probs_for_classifier=True,
                    additive_noise_augmentation=0.0
                )
            else:
                training_module = ALLTrainer(
                    classifier_name='multilayer-perceptron',
                    classifier_kwargs={'input_shape': (train_dataset.timesteps_per_trace,), 'output_classes': 2},
                    classifier_optimizer_name='AdamW',
                    obfuscator_optimizer_name='AdamW',
                    obfuscator_l2_norm_penalty=L2_NORM_PENALTY,
                    split_training_steps=TRAINING_EPOCHS*len(data_module.train_dataloader()),
                    classifier_optimizer_kwargs={'lr': 2e-4},
                    obfuscator_optimizer_kwargs={'lr': 1e-2},
                    #classifier_lr_scheduler_name='CosineDecayLRSched',
                    #obfuscator_lr_scheduler_name='CosineDecayLRSched',
                    obfuscator_batch_size_multiplier=8,
                    normalize_erasure_probs_for_classifier=True,
                    additive_noise_augmentation=0.0
                )
                trainer = Trainer(
                    max_epochs=2*TRAINING_EPOCHS,
                    default_root_dir=logging_dir,
                    accelerator='gpu',
                    devices=1,
                    logger=TensorBoardLogger(logging_dir, name='lightning_output')
                )
                trainer.fit(training_module, datamodule=data_module)
                trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            all_val = (
                training_module.obfuscator_l2_norm_penalty*nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
            )
            all_vals[easy_feature_count_idx, seed, :] = all_val
            ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
            ea.Reload()
            training_curves = {
                key: extract_trace(ea.Scalars(key)) for key in [
                    'classifier-train-loss_epoch', 'classifier-val-loss', 'obfuscator-train-loss_epoch', 'obfuscator-val-loss', 'min-obf-weight', 'max-obf-weight',
                    'train-acc_epoch', 'val-acc'
                ]
            }
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].plot(*training_curves['classifier-train-loss_epoch'], color='blue', linestyle='--')
            axes[0].plot(*training_curves['classifier-val-loss'], color='blue', linestyle='-')
            axes[1].plot(*training_curves['obfuscator-train-loss_epoch'], color='blue', linestyle='--')
            axes[1].plot(*training_curves['obfuscator-val-loss'], color='blue', linestyle='-')
            axes[2].plot(*training_curves['min-obf-weight'], color='red', linestyle='-')
            axes[2].plot(*training_curves['max-obf-weight'], color='blue', linestyle='-')
            axes[3].plot(*training_curves['train-acc_epoch'], color='blue', linestyle='--')
            axes[3].plot(*training_curves['val-acc'], color='blue', linestyle='-')
            axes[0].set_xlabel('Step')
            axes[1].set_xlabel('Step')
            axes[2].set_xlabel('Step')
            axes[0].set_ylabel('Classifier loss')
            axes[1].set_ylabel('Obfuscator loss')
            axes[2].set_ylabel('Obfuscation weights')
            axes[3].set_xlabel('Step')
            axes[3].set_ylabel('Accuracy')
            axes[2].set_yscale('log')
            axes[1].set_yscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(SAVE_DIR, logging_dir, 'training_curves.png'))
            plt.close('all')
    with open(os.path.join(SAVE_DIR, 'results.pickle'), 'wb') as f:
        pickle.dump({
            'snr_vals': snr_vals, 'sosd_vals': sosd_vals, 'cpa_vals': cpa_vals,
            'gradvis_vals': gradvis_vals, 'inputxgrad_vals': inputxgrad_vals, 'ablation_vals': ablation_vals,
            'all_vals': all_vals
        }, f)
else:
    with open(os.path.join(SAVE_DIR, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)
        snr_vals = results['snr_vals']
        sosd_vals = results['sosd_vals']
        cpa_vals = results['cpa_vals']
        gradvis_vals = results['gradvis_vals']
        inputxgrad_vals = results['inputxgrad_vals']
        ablation_vals = results['ablation_vals']
        all_vals = results['all_vals']

kwargss = [
    {'color': 'red', 'label': 'XOR (`hard\') feature: dim 0'}, {'color': 'blue', 'label': 'XOR (`hard\') feature: dim 1'},
    {'color': 'green', 'label': 'GMM (`easy\') feature'}, {'color': 'orange', 'label': 'Random feature'}
]
fig, axes = plt.subplots(3, 3, figsize=(4*3, 4*3))
axes[-1, 0].axis('off')
axes[-1, -1].axis('off')
_axes = axes
axes = [axes.flatten()[idx] for idx in range(9) if not(idx in [6, 8])]
for method, title, ax in zip([snr_vals, sosd_vals, cpa_vals, gradvis_vals, inputxgrad_vals, ablation_vals, all_vals], ['SNR', 'SOSD', 'CPA', 'GradVis', 'Input*Grad', 'Ablation', 'ALL (ours)'], axes):
    for easy_feature_count_idx, easy_feature_count in enumerate(easy_feature_counts):
        easy = method[easy_feature_count_idx, :, :easy_feature_count].flatten()
        hard = method[easy_feature_count_idx, :, easy_feature_count:].flatten()
        ax.plot(easy_feature_count*np.ones(len(easy)), easy, color='blue', linestyle='none', marker='.')
        ax.plot(easy_feature_count*np.ones(len(hard)), hard, color='red', linestyle='none', marker='.')
    ax.set_ylabel(title)
    ax.set_xlabel('Predictive feature count')
    ax.set_xscale('log')
    if title in ['SNR', 'SOSD', 'GradVis', 'Input*Grad', 'Ablation', 'ALL (ours)']:
        ax.set_yscale('log')
handles = [mlines.Line2D([], [], **kwargs) for kwargs in kwargss]
handles.append(mlines.Line2D([], [], color='gray', label='Median (5 seeds)'))
handles.append(mpatches.Patch(color='gray', label='Min -- Max (5 seeds)', alpha=0.25))
_axes[-1, -1].legend(handles=handles, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'results_vs_count.pdf'))