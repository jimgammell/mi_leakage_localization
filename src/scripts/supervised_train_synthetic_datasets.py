import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from _common import *
import datasets
from training_modules.supervised_classification import SupervisedClassificationModule
from utils.recalibrate_batchnorm_stats import recalibrate_batchnorm_stats
from utils.lightning_callbacks import GradientTracker

TUNE_BATCH_SIZES = False
TUNE_LEARNING_RATE = True

data_module = datasets.load('synthetic-aes', train_batch_size=1024, eval_batch_size=10000, aug=True, dataset_kwargs={'num_leaking_subbytes_cycles': 5})
training_module = SupervisedClassificationModule(
    model_name='sca-cnn',
    optimizer_name='AdamW',
    optimizer_kwargs={'lr': 2e-4, 'weight_decay': 1e-2},
    lr_scheduler_name='CosineDecayLRSched',
    post_train_epoch_callbacks=[recalibrate_batchnorm_stats]
)
trainer = Trainer(
    max_epochs=100,
    default_root_dir=get_trial_dir(),
    accelerator='gpu',
    devices=1,
    enable_checkpointing=True,
    logger=TensorBoardLogger(get_trial_dir(), name='tensorboard'),
    callbacks=[LearningRateMonitor(logging_interval='step'), GradientTracker()]
)
tuner = Tuner(trainer)
if TUNE_BATCH_SIZES:
    tuner.scale_batch_size(training_module, datamodule=data_module, mode='power', method='fit', batch_arg_name='train_batch_size')
    tuner.scale_batch_size(training_module, datamodule=data_module, mode='power', method='validate', batch_arg_name='eval_batch_size')
    print(data_module.train_batch_size)
    print(data_module.eval_batch_size)
if TUNE_LEARNING_RATE:
    tuner.lr_find(training_module, datamodule=data_module, method='fit', num_training=100, mode='exponential')
    print(training_module.learning_rate)
trainer.fit(training_module, datamodule=data_module)