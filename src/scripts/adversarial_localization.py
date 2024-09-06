from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule

data_module = datasets.load('synthetic-aes', train_batch_size=512, eval_batch_size=5000, aug=False, dataset_kwargs={'num_leaking_subbytes_cycles': 1, 'lpf_beta': 0.0})
training_module = AdversarialLocalizationModule(
    classifier_name='sca-cnn',
    classifier_optimizer_name='AdamW',
    obfuscator_optimizer_name='SGD',
    obfuscator_l2_norm_penalty=1e0,
    classifier_lr_scheduler_name=None,
    obfuscator_lr_scheduler_name=None,
    classifier_kwargs={'input_shape': (2, 1000), 'base_channels': 64},
    classifier_optimizer_kwargs={'lr': 1e-3},
    obfuscator_optimizer_kwargs={'lr': 1e0, 'momentum': 0.99},
    obfuscator_batch_size_multiplier=8
)
trainer = Trainer(
    max_epochs=-1,
    default_root_dir=get_trial_dir(),
    accelerator='gpu',
    devices=1,
    enable_checkpointing=True,
    logger=TensorBoardLogger(get_trial_dir(), name='lightning_output')
)
trainer.fit(training_module, datamodule=data_module)