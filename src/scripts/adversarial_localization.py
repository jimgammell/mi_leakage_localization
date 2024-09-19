from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule

TIMESTEPS_PER_TRACE = 100

data_module = datasets.load(
    'synthetic-aes', train_batch_size=512, eval_batch_size=5000, aug=False,
    dataset_kwargs={'timesteps_per_trace': TIMESTEPS_PER_TRACE, 'leaking_timestep_count_1o': 0, 'leaking_timestep_count_2o': 1, 'infinite_dataset': False, 'data_var': 10.0, 'residual_var': 0.1, 'lpf_beta': 0.0} ## should fix infinite_dataset setting
)
training_module = AdversarialLocalizationModule(
    classifier_name='sca-cnn',
    classifier_optimizer_name='AdamW',
    obfuscator_optimizer_name='AdamW',
    obfuscator_l2_norm_penalty=1e-1,
    classifier_lr_scheduler_name=None,
    obfuscator_lr_scheduler_name=None,
    classifier_kwargs={'input_shape': (2, TIMESTEPS_PER_TRACE), 'head_kwargs': {'xor_output': True}},
    classifier_optimizer_kwargs={'lr': 1e-5},
    obfuscator_optimizer_kwargs={'lr': 1e-3},
    obfuscator_batch_size_multiplier=8
)
trainer = Trainer(
    max_epochs=500,
    default_root_dir=get_trial_dir(),
    accelerator='gpu',
    devices=1,
    enable_checkpointing=True,
    logger=TensorBoardLogger(get_trial_dir(), name='lightning_output')
)
trainer.fit(training_module, datamodule=data_module)