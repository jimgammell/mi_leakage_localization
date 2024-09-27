from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from _common import *
import datasets
from training_modules import AdversarialLocalizationModule

data_module = datasets.load('simple-gaussian', train_batch_size=2048, eval_batch_size=8192)
training_module = AdversarialLocalizationModule(
    classifier_name='multilayer-perceptron',
    classifier_kwargs=dict(input_shape=(2, 2), hidden_layers=1, hidden_dims=512, output_classes=2),
    classifier_optimizer_name='AdamW',
    classifier_optimizer_kwargs=dict(lr=2e-4),
    obfuscator_optimizer_name='AdamW',
    obfuscator_optimizer_kwargs=dict(lr=1e-3),
    obfuscator_batch_size_multiplier=8,
    obfuscator_l2_norm_penalty=1.0,
    log_likelihood_baseline_ema=0.9
)
print(training_module.classifier)
trainer = Trainer(
    max_epochs=-1,
    default_root_dir=get_trial_dir(),
    accelerator='gpu',
    devices=1,
    enable_checkpointing=True,
    logger=TensorBoardLogger(get_trial_dir(), name='lightning_output')
)
trainer.fit(training_module, datamodule=data_module)