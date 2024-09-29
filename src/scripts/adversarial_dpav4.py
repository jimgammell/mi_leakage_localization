import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.dpav4 import DPAv4, to_key_preds
from utils.metrics.rank import accumulate_ranks
import datasets
from training_modules import AdversarialLocalizationModule

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization','downloads', 'dpav4')
STEP_COUNT = 100000
BATCH_SIZE = 256

profiling_dataset = DPAv4(
    root=ROOT,
    train=True
)
attack_dataset = DPAv4(
    root=ROOT,
    train=False
)

data_module = datasets.load('dpav4', train_batch_size=BATCH_SIZE, eval_batch_size=1024, root=ROOT)

norm_penalties = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
for norm_penalty in norm_penalties:
    logging_dir = os.path.join(get_trial_dir(), f'lambda={norm_penalty}')
    training_module = AdversarialLocalizationModule(
        classifier_name='sca-cnn',
        classifier_kwargs={'input_shape': (2, profiling_dataset.timesteps_per_trace)},
        classifier_optimizer_name='AdamW',
        obfuscator_optimizer_name='AdamW',
        classifier_optimizer_kwargs={'lr': 6e-6},
        obfuscator_optimizer_kwargs={'lr': 1e-4},
        obfuscator_l2_norm_penalty=norm_penalty,
        obfuscator_batch_size_multiplier=8,
        normalize_erasure_probs_for_classifier=True,
        additive_noise_augmentation=0.25
    )
    trainer = Trainer(
        max_epochs=STEP_COUNT*BATCH_SIZE//len(profiling_dataset),
        val_check_interval=STEP_COUNT//100,
        check_val_every_n_epoch=None,
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
        key: extract_trace(ea.Scalars(key)) for key in ['classifier-train-loss_epoch', 'classifier-val-loss', 'obfuscator-train-loss_epoch', 'obfuscator-val-loss']
    }
    erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
    with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
        pickle.dump({'training_curves': training_curves, 'erasure_probs': erasure_probs}, f)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(*training_curves['classifier-train-loss_epoch'], color='red', linestyle='--')
    axes[0].plot(*training_curves['classifier-val-loss'], color='red', linestyle='-')
    axes[1].plot(*training_curves['obfuscator-train-loss_epoch'], color='blue', linestyle='--')
    axes[1].plot(*training_curves['obfuscator-val-loss'], color='blue', linestyle='-')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Classifier loss')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Obfuscator loss')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.pdf'))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(training_module.obfuscator_l2_norm_penalty*erasure_probs, color='blue', linestyle='none', marker='.')
    ax.set_ylim(0, training_module.obfuscator_l2_norm_penalty)
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Leakage assessment $\lambda \gamma_t^*$')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'leakage_assessment.pdf'))