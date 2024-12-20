import os
from copy import copy
from typing import *
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from lightning import LightningModule
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner

from common import *
from trials.utils import *
from datasets.data_module import DataModule
from .plot_things import *

class AdversarialLeakageLocalizationTrainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        max_epochs: int = 100,
        gradient_estimation_strategy: Literal['REINFORCE', 'Relaxed', 'Dependent', 'WinnerTakeAll'] = 'WinnerTakeAll',
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {}
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.max_epochs = max_epochs
        self.gradient_estimation_strategy = gradient_estimation_strategy
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        
        self.data_module = DataModule(self.profiling_dataset, self.attack_dataset, adversarial_mode=False, **self.default_data_module_kwargs)
        if self.gradient_estimation_strategy == 'REINFORCE':
            from .reinforce_training_module import AdversarialLeakageLocalizationModule
        elif self.gradient_estimation_strategy == 'Relaxed':
            from .relaxed_training_module import AdversarialLeakageLocalizationModule
        elif self.gradient_estimation_strategy == 'Dependent':
            from .dependent_training_module import AdversarialLeakageLocalizationModule
        elif self.gradient_estimation_strategy == 'WinnerTakeAll':
            from .winner_take_all_module import AdversarialLeakageLocalizationModule
        else:
            raise NotImplementedError
        self.training_module_class = AdversarialLeakageLocalizationModule
    
    def get_training_module_kwargs(self, override_kwargs: dict = {}):
        kwargs = copy(self.default_training_module_kwargs)
        return kwargs
    
    def smith_lr_sweep(self, logging_dir, override_kwargs: dict = {}):
        training_module = self.training_module_class(**self.get_training_module_kwargs(override_kwargs))
        class ModuleWrapper(LightningModule):
            def __init__(self, all_module, lr=None):
                super().__init__()
                self.all_module = all_module
                self.lr = lr
            def forward(self, *args):
                return self.all_module.classifiers(*args)
            def training_step(self, batch, batch_idx):
                x, y = batch
                gamma = self.all_module.get_gamma()
                noise = self.all_module.sample_noise(x, gamma, training_theta=True)
                logits = self.all_module.get_logits(x, noise, calibrate=False)
                loss = nn.functional.cross_entropy(logits, y)
                return loss
            def configure_optimizers(self):
                rv = self.all_module.configure_optimizers()
                theta_optimizer = rv[0]['optimizer']
                if self.lr is not None:
                    for g in theta_optimizer.param_groups:
                        g['lr'] = self.lr
                return theta_optimizer
        smith_compatible_module = ModuleWrapper(training_module)
        trainer = Trainer(
            max_epochs=self.theta_pretrain_epochs,
            val_check_interval=1.,
            default_root_dir=logging_dir,
            accelerator='gpu',
            devices=1,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='value',
            logger=TensorBoardLogger(logging_dir, name='lightning_output')
        )
        trainer.datamodule = self.data_module
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            smith_compatible_module,
            train_dataloaders=self.data_module.train_dataloader()[0],
            val_dataloaders=self.data_module.val_dataloader(),
            min_lr=1e-6, max_lr=1e-1, num_training=1000
        )
        fig = lr_finder.plot(suggest=True)
        fig.savefig(os.path.join(logging_dir, 'lr_finder_results.png'))
    
    def pretrain_classifiers(self, logging_dir: Union[str, os.PathLike], override_kwargs: dict = {}):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module = self.training_module_class(**self.get_training_module_kwargs(override_kwargs))
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
            training_module = self.training_module_class.load_from_checkpoint(os.path.join(logging_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt'))
            classifier_state = training_module.classifiers.state_dict()
            torch.save(classifier_state, os.path.join(logging_dir, 'classifier_state.pth'))
        plot_theta_pretraining_curves(logging_dir)
    
    def train_gamma(self, 
            logging_dir: Union[str, os.PathLike],
            starting_module_path: Optional[Union[str, os.PathLike]] = None,
            override_kwargs: dict = {},
            anim_gammas: bool = True
        ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            override_kwargs.update({'theta_pretrain_steps': 0})
            training_module = self.training_module_class(**self.get_training_module_kwargs(override_kwargs))
            if starting_module_path is not None:
                override_training_module = self.training_module_class.load_from_checkpoint(starting_module_path)
                training_module.classifiers.load_state_dict(override_training_module.classifiers.state_dict())
            trainer = Trainer(
                max_epochs=self.max_epochs,
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
        plot_gammap_training_curves(logging_dir, anim_gammas=anim_gammas)