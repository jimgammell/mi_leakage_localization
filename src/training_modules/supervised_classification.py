from typing import *
import numpy as np
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.classification import Accuracy

import models
import utils.lr_schedulers
from utils.metrics import Rank

class SupervisedClassificationModule(L.LightningModule):
    def __init__(self,
        model_name: str,
        optimizer_name: Union[str, optim.Optimizer] = 'Adam',
        lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        model_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        lr_scheduler_kwargs: dict = {},
        learning_rate: Optional[float] = None,
        additive_noise_augmentation: float = 0.0
    ):
        for key, val in locals().items():
            if not key in ('self', 'key', 'val'):
                setattr(self, key, val)
        super().__init__()
        self.model = models.load(self.model_name, **self.model_kwargs)
        self.train_accuracy, self.val_accuracy, self.test_accuracy = map(lambda _: Accuracy(task='multiclass', num_classes=self.model.output_classes), range(3))
        self.train_rank, self.val_rank, self.test_rank = map(lambda _: Rank(), range(3))
        if self.learning_rate is None:
            assert 'lr' in self.optimizer_kwargs.keys()
            self.learning_rate = self.optimizer_kwargs['lr']
        self.epochs_seen = 0
    
    def configure_optimizers(self):
        if isinstance(self.optimizer_name, str):
            optimizer_constructor = getattr(optim, self.optimizer_name)
        else:
            optimizer_constructor = self.optimizer_name
        weight_decay = self.optimizer_kwargs['weight_decay'] if 'weight_decay' in self.optimizer_kwargs.keys() else 0.0
        self.optimizer_kwargs = {key: val for key, val in self.optimizer_kwargs.items() if key not in ['lr', 'weight_decay']}
        yes_weight_decay, no_weight_decay = [], []
        for name, param in self.model.named_parameters():
            if ('weight' in name) and not('norm' in name):
                yes_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [
            {'params': yes_weight_decay, 'weight_decay': weight_decay},
            {'params': no_weight_decay, 'weight_decay': 0.0}
        ]
        optimizer = optimizer_constructor(param_groups, lr=self.learning_rate, **self.optimizer_kwargs)
        rv = {'optimizer': optimizer}
        if self.lr_scheduler_name is not None:
            if isinstance(self.lr_scheduler_name, str):
                try:
                    lr_scheduler_constructor = getattr(utils.lr_schedulers, self.lr_scheduler_name)
                except:
                    lr_scheduler_constructor = getattr(optim.lr_scheduler, self.lr_scheduler_name)
            else:
                lr_scheduler_constructor = self.lr_scheduler_name
            lr_scheduler = lr_scheduler_constructor(
                optimizer, total_steps=self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader()), **self.lr_scheduler_kwargs
            )
            rv.update({'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}})
        return rv

    def forward(self, inputs):
        return self.model(inputs)
    
    def _step(self, batch, batch_idx, aug=False):
        inputs, targets = batch
        if aug:
            inputs = inputs + self.additive_noise_augmentation*torch.randn_like(inputs)
        logits = self(inputs)
        logits = logits.view(-1, logits.size(-1))
        if len(targets.shape) > 1:
            loss_multiplier = targets.size(-2)
            targets = targets.view(-1, targets.size(-1))
        else:
            loss_multiplier = 1
        return logits, targets, loss_multiplier
    
    def training_step(self, batch, batch_idx):
        logits, targets, loss_multiplier = self._step(batch, batch_idx, aug=True)
        loss = loss_multiplier*nn.functional.cross_entropy(logits, targets)
        self.train_accuracy(logits, targets)
        self.train_rank(logits, targets)
        self.log('train-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train-acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train-rank', self.train_rank, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, targets, loss_multiplier = self._step(batch, batch_idx)
        loss = loss_multiplier*nn.functional.cross_entropy(logits, targets)
        self.val_accuracy(logits, targets)
        self.val_rank(logits, targets)
        self.log('val-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val-acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-rank', self.val_rank, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        logits, targets, loss_multiplier = self._step(batch, batch_idx)
        loss = loss_multiplier*nn.functional.cross_entropy(logits, targets)
        self.test_accuracy(logits, targets)
        self.test_rank(logits, targets)
        self.log('test-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test-acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test-rank', self.test_rank, on_step=False, on_epoch=True, prog_bar=True)