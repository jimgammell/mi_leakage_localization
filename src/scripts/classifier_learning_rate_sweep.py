import os
import pickle
import numpy as np
from torch import nn
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule

TIMESTEPS_PER_TRACE = 1000
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 2048
MAX_EPOCHS = 500

classifier_learning_rates = np.logspace(-6, -3, 100)
obfuscator_learning_rates = [1e-3]

data_module = datasets.load(
    'synthetic-aes',
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    aug=False,
    dataset_kwargs = {
        'timesteps_per_trace': TIMESTEPS_PER_TRACE,
        'max_no_ops': 2,
        'shuffle_locs': 2,
        'lpf_beta': 0.9
    }
)
data_module.setup('train')

train_dataset = data_module.train_dataset
dataset_summary = {
    'leaking_subbytes_cycles': train_dataset.leaking_subbytes_cycles,
    'leaking_mask_cycles': train_dataset.leaking_mask_cycles,
    'leaking_masked_subbytes_cycles': train_dataset.leaking_masked_subbytes_cycles,
    **train_dataset.settings
}
with open(os.path.join(get_trial_dir(), 'dataset_summary.pickle'), 'wb') as f:
    pickle.dump(dataset_summary, f)

for classifier_learning_rate in classifier_learning_rates:
    for obfuscator_learning_rate in obfuscator_learning_rates:
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
            obfuscator_batch_size_multiplier=8
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
        with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
            pickle.dump({
                'classifier_train_loss': classifier_train_loss, 'classifier_val_loss': classifier_val_loss,
                'obfuscator_train_loss': obfuscator_train_loss, 'obfuscator_val_loss': obfuscator_val_loss
            }, f)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            [x.step for x in classifier_train_loss], [x.value for x in classifier_train_loss], color='red', linestyle='--', label='classifier-train'
        )
        ax.plot(
            [x.step for x in classifier_val_loss], [x.value for x in classifier_val_loss], color='red', linestyle='-', label='classifier-val'
        )
        tax = ax.twinx()
        tax.plot(
            [x.step for x in obfuscator_train_loss], [x.value for x in obfuscator_val_loss], color='blue', linestyle='--', label='obfuscator-train'
        )
        tax.plot(
            [x.step for x in obfuscator_val_loss], [x.value for x in obfuscator_val_loss], color='blue', linestyle='-', label='obfuscator-val'
        )
        ax.set_xlabel('Training step')
        ax.set_ylabel('Classifier loss')
        tax.set_ylabel('Obfuscator loss')
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