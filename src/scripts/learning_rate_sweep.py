import os
import shutil
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

r"""import ray
from ray import train, tune
os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = '0'"""

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule
from utils.template_attack import TemplateAttack

TIMESTEPS_PER_TRACE = 1000
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 2048
MAX_EPOCHS = 1000

classifier_learning_rates = np.logspace(-5, -1, 5)
obfuscator_learning_rates = np.logspace(-4, -2, 5)
if PART is not None:
    obfuscator_learning_rates = [obfuscator_learning_rates[PART]]

data_module = datasets.load(
    'synthetic-aes',
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    aug=False,
    dataset_kwargs = {
        'timesteps_per_trace': TIMESTEPS_PER_TRACE,
        'max_no_ops': 2,
        'shuffle_locs': 2,
        'lpf_beta': 0.9,
        'residual_var': 0.5,
        'infinite_dataset': False
    }
)
data_module.setup('fit')
data_module.setup('test')
train_dataset = data_module.train_dataset
val_dataset = data_module.val_dataset
test_dataset = data_module.test_dataset
train_dataset.transform = train_dataset.target_transform = val_dataset.transform = val_dataset.target_transform = test_dataset.transform = test_dataset.target_transform = None
with open(os.path.join(get_trial_dir(), 'datasets.pickle'), 'wb') as f:
    pickle.dump({'train_dataset': train_dataset, 'val_dataset': val_dataset, 'test_dataset': test_dataset}, f)

def run_trial(config):
    classifier_learning_rate = config['classifier_learning_rate']
    obfuscator_learning_rate = config['obfuscator_learning_rate']
    logging_dir = os.path.join(get_trial_dir(), f'clr_{classifier_learning_rate}__olr_{obfuscator_learning_rate}')
    os.makedirs(logging_dir, exist_ok=True)
    training_module = AdversarialLocalizationModule(
        classifier_name='multilayer-perceptron',
        classifier_optimizer_name='AdamW',
        obfuscator_optimizer_name='AdamW',
        obfuscator_l2_norm_penalty=1e-1,
        classifier_kwargs={'input_shape': (2, TIMESTEPS_PER_TRACE), 'xor_output': False},
        classifier_optimizer_kwargs={'lr': classifier_learning_rate},
        obfuscator_optimizer_kwargs={'lr': obfuscator_learning_rate},
        obfuscator_batch_size_multiplier=8,
        classifier_step_prob=0.1,
        obfuscator_step_prob=1.0
    )
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        default_root_dir=logging_dir,
        logger=TensorBoardLogger(os.path.join(logging_dir, 'lightning_output')),
        accelerator='gpu',
        devices=1
    )
    trainer.fit(training_module, datamodule=data_module)
    trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
    obfuscation_weights = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).squeeze().detach().cpu().numpy()
    with open(os.path.join(logging_dir, 'trained_erasure_probs.pickle'), 'wb') as f:
        pickle.dump(obfuscation_weights, f)
        
    ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'lightning_logs', 'version_0'))
    ea.Reload()
    classifier_train_loss = ea.Scalars('classifier-train-loss_epoch')
    classifier_val_loss = ea.Scalars('classifier-val-loss')
    obfuscator_train_loss = ea.Scalars('obfuscator-train-loss_epoch')
    obfuscator_val_loss = ea.Scalars('obfuscator-val-loss')
    classifier_train_rank = ea.Scalars('train-rank')
    classifier_val_rank = ea.Scalars('val-rank')
    with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
        pickle.dump({
            'classifier_train_loss': classifier_train_loss, 'classifier_val_loss': classifier_val_loss,
            'obfuscator_train_loss': obfuscator_train_loss, 'obfuscator_val_loss': obfuscator_val_loss,
            'train_rank': classifier_train_rank, 'val_rank': classifier_val_rank
        }, f)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(
        [x.step for x in classifier_train_loss], [x.value for x in classifier_train_loss], color='red', linestyle='--', label='classifier-train'
    )
    axes[0].plot(
        [x.step for x in classifier_val_loss], [x.value for x in classifier_val_loss], color='red', linestyle='-', label='classifier-val'
    )
    tax = axes[0].twinx()
    tax.plot(
        [x.step for x in obfuscator_train_loss], [x.value for x in obfuscator_val_loss], color='blue', linestyle='--', label='obfuscator-train'
    )
    tax.plot(
        [x.step for x in obfuscator_val_loss], [x.value for x in obfuscator_val_loss], color='blue', linestyle='-', label='obfuscator-val'
    )
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Classifier loss')
    tax.set_ylabel('Obfuscator loss')
    axes[1].plot(
        [x.step for x in classifier_train_rank], [x.value for x in classifier_train_rank], color='red', linestyle='--', label='classifier-train'
    )
    axes[1].plot(
        [x.step for x in classifier_val_rank], [x.value for x in classifier_val_rank], color='red', linestyle='-', label='classifier-val'
    )
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Rank')
    fig.savefig(os.path.join(logging_dir, 'training_curves.pdf'))
        
    fig, ax = plt.subplots(figsize=(4, 4))
    for idx, cycle in enumerate(train_dataset.leaking_subbytes_cycles):
        ax.axvspan(cycle, cycle+train_dataset.max_no_ops, color='blue', alpha=0.5, label='Ground truth' if idx == 0 else None)
    ax.plot(obfuscation_weights, color='blue', linestyle='none', marker='.', label='Erasure probability')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Power measurement')
    ax.set_ylabel('Erasure probability')
    ax.legend()
    fig.savefig(os.path.join(logging_dir, 'erasure_probs.pdf'))
    
    points_of_interest = obfuscation_weights.argsort()[-10:]
    template_attack = TemplateAttack(points_of_interest)
    template_attack.profile(train_dataset)
    rank_over_time = template_attack.attack(test_dataset, n_repetitions=1000, n_traces=1000)
    with open(os.path.join(logging_dir, 'ta_rank_over_time.pickle'), 'wb') as f:
        pickle.dump(rank_over_time, f)
    fig, ax = plt.subplots(figsize=(4, 4))
    mean = rank_over_time.mean(axis=0)
    std = rank_over_time.std(axis=0)
    ax.plot(mean, color='blue')
    ax.fill_between(np.arange(len(mean)), mean-std, mean+std, color='blue', alpha=0.5)
    ax.set_xlabel('Traces observed')
    ax.set_ylabel('Correct key rank')
    fig.savefig(os.path.join(logging_dir, 'ta_rank_over_time.pdf'))

for classifier_learning_rate in classifier_learning_rates:
    for obfuscator_learning_rate in obfuscator_learning_rates:
        run_trial({'classifier_learning_rate': classifier_learning_rate, 'obfuscator_learning_rate': obfuscator_learning_rate})