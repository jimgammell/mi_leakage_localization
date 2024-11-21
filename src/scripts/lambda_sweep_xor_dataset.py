import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')))
import shutil
from copy import deepcopy
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
RANDOM_FEATURE_COUNT = 1
TRAINING_EPOCHS = 100
SAVE_DIR = os.path.join(OUTPUT_DIR, 'lambda_sweep_xor')
os.makedirs(SAVE_DIR, exist_ok=True)
SIGMA = 1e1
SEED_COUNT = 1
LAMBDA_VALS = np.log(2) * np.concatenate([np.logspace(-2, 2, 100)])
if not os.path.exists(os.path.join(SAVE_DIR, 'results.pickle')):
    all_vals = np.full((len(LAMBDA_VALS), SEED_COUNT, 4), np.nan, dtype=float)
    for seed in range(SEED_COUNT):
        train_dataset = TwoSpiralsDataset(DATASET_SIZE, easy_feature_sigma=SIGMA, xor_hard_feature=True, random_feature_count=RANDOM_FEATURE_COUNT)
        val_dataset = TwoSpiralsDataset(DATASET_SIZE, easy_feature_sigma=SIGMA, xor_hard_feature=True, random_feature_count=RANDOM_FEATURE_COUNT)
        mean = train_dataset.x.mean(axis=0)
        std = train_dataset.x.std(axis=0)
        transform = lambda x: (x - mean) / std
        train_dataset.transform = val_dataset.transform = transform
        data_module = TwoSpiralsDataModule(train_dataset=train_dataset, eval_dataset=val_dataset, train_batch_size=1024)
        logging_dir = os.path.join(SAVE_DIR, f'classifiers_pretrain__seed_{seed}.ckpt')
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
        training_module = ALLTrainer(
            classifier_name='multilayer-perceptron',
            classifier_kwargs={'input_shape': (3+RANDOM_FEATURE_COUNT,), 'output_classes': 2},
            classifier_optimizer_name='AdamW',
            obfuscator_optimizer_name='LBFGS',
            obfuscator_l2_norm_penalty=np.log(2),
            split_training_steps=TRAINING_EPOCHS*len(data_module.train_dataloader()),
            classifier_optimizer_kwargs={'lr': 2e-4},
            obfuscator_optimizer_kwargs={'lr': 0.1},
            obfuscator_batch_size_multiplier=8,
            normalize_erasure_probs_for_classifier=True,
            additive_noise_augmentation=0.0
        )
        trainer = Trainer(
            max_epochs=TRAINING_EPOCHS,
            default_root_dir=logging_dir,
            accelerator='gpu',
            devices=1,
            logger=TensorBoardLogger(logging_dir, name='lightning_output')
        )
        trainer.fit(training_module, datamodule=data_module)
        classifiers_state = deepcopy(training_module.classifier.state_dict())
        
        for lambda_idx, lambda_val in enumerate(LAMBDA_VALS):
            logging_dir = os.path.join(SAVE_DIR, f'erasure_probs_train__seed_{seed}__lambda_{lambda_val}.ckpt')
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module = ALLTrainer(
                classifier_name='multilayer-perceptron',
                classifier_kwargs={'input_shape': (3+RANDOM_FEATURE_COUNT,), 'output_classes': 2},
                classifier_optimizer_name='AdamW',
                obfuscator_optimizer_name='LBFGS',
                obfuscator_l2_norm_penalty=lambda_val,
                split_training_steps=0,
                classifier_optimizer_kwargs={'lr': 2e-4},
                obfuscator_optimizer_kwargs={'lr': 0.1},
                obfuscator_batch_size_multiplier=8,
                normalize_erasure_probs_for_classifier=True,
                additive_noise_augmentation=0.0
            )
            trainer = Trainer(
                max_epochs=10,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            training_module.classifier.load_state_dict(classifiers_state)
            trainer.fit(training_module, datamodule=data_module)
            all_val = (
                training_module.obfuscator_l2_norm_penalty*nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
            )
            all_vals[lambda_idx, seed, :] = all_val
            print(all_val)
    with open(os.path.join(SAVE_DIR, 'results.pickle'), 'wb') as f:
        pickle.dump(all_vals, f)
else:
    with open(os.path.join(SAVE_DIR, 'results.pickle'), 'rb') as f:
        all_vals = pickle.load(f)

kwargss = [
    {'color': 'red', 'label': 'XOR (`hard\') feature: dim 0'}, {'color': 'blue', 'label': 'XOR (`hard\') feature: dim 1'},
    {'color': 'green', 'label': 'GMM (`easy\') feature'}, {'color': 'orange', 'label': 'Random feature'}
]
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xscale('log')
ax.set_xlabel('Norm penalty: $\lambda$')
ax.set_ylabel('ALL (ours): $\lambda \gamma_t^*$')
ax.plot(LAMBDA_VALS, LAMBDA_VALS, color='black', linestyle='--', label='$\lambda$ (saturation value)')
for feature_idx, kwargs in zip(range(4), kwargss):
    ax.plot(LAMBDA_VALS, np.median(all_vals[:, :, feature_idx], axis=1), marker='.', linestyle='none', markersize=2, **kwargs)
    ax.fill_between(LAMBDA_VALS, all_vals[:, :, feature_idx].min(axis=1), all_vals[:, :, feature_idx].max(axis=1), alpha=0.25, **kwargs)
ax.set_yscale('log')
handles = [mlines.Line2D([], [], **kwargs) for kwargs in kwargss]
handles.append(mlines.Line2D([], [], color='gray', label='Median (5 seeds)'))
handles.append(mpatches.Patch(color='gray', label='Min -- Max (5 seeds)', alpha=0.25))
ax.legend(handles=handles, loc='lower right')
ax.set_ylim(1e-5, 1e2)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'lambda_sweep.pdf'))