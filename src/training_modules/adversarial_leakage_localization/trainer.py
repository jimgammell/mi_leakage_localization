import os
from copy import copy
from typing import *
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from trials.utils import *
from datasets.data_module import DataModule
from .training_module import AdversarialLeakageLocalizationModule
from .plot_things import *

class AdversarialLeakageLocalizationTrainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        theta_pretrain_epochs: int = 0,
        adversarial_train_epochs: int = 100,
        gammap_posttrain_epochs: int = 0,
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {}
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.theta_pretrain_epochs = theta_pretrain_epochs
        self.adversarial_train_epochs = adversarial_train_epochs
        self.gammap_posttrain_epochs = gammap_posttrain_epochs
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        
        self.data_module = DataModule(self.profiling_dataset, self.attack_dataset, adversarial_mode=True, **self.default_data_module_kwargs)
    
    def get_training_module_kwargs(self, override_kwargs: dict = {}):
        kwargs = copy(self.default_training_module_kwargs)
        kwargs.update(override_kwargs)
        assert not any(x in override_kwargs for x in ['theta_pretrain_steps', 'adversarial_train_steps'])
        steps_per_epoch = len(self.data_module.train_dataloader())
        kwargs['theta_pretrain_steps'] = self.theta_pretrain_epochs*steps_per_epoch
        kwargs['alternating_train_steps'] = self.adversarial_train_epochs*steps_per_epoch
        return kwargs
    
    def pretrain_classifiers(self, logging_dir: Union[str, os.PathLike], override_kwargs: dict = {}):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module = AdversarialLeakageLocalizationModule(**self.get_training_module_kwargs(override_kwargs))
            checkpoint = ModelCheckpoint(
                filename='best',
                monitor='val_theta_loss',
                mode='min'
            )
            trainer = Trainer(
                max_epochs=self.theta_pretrain_epochs,
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
            training_module = AdversarialLeakageLocalizationModule.load_from_checkpoint(os.path.join(logging_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt'))
            classifier_state = training_module.classifiers.state_dict()
            torch.save(classifier_state, os.path.join(logging_dir, 'classifier_state.pth'))
        plot_theta_pretraining_curves(logging_dir)
    
    def train_gamma(self, logging_dir: Union[str, os.PathLike], starting_module_path: Optional[Union[str, os.PathLike]] = None, override_kwargs: dict = {}):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            if starting_module_path is not None:
                training_module = AdversarialLeakageLocalizationModule.load_from_checkpoint(starting_module_path, **override_kwargs)
            else:
                training_module = AdversarialLeakageLocalizationModule(**self.get_training_module_kwargs(override_kwargs))
            trainer = Trainer(
                max_epochs=self.theta_pretrain_epochs+self.adversarial_train_epochs+self.gammap_posttrain_epochs,
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        plot_gammap_training_curves(logging_dir)