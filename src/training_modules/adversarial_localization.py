from typing import *
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.classification import Accuracy

from common import *
import models
import utils.lr_schedulers

class AdversarialLocalizationTrainer(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        obfuscator_name: str,
        classifier_optimizer_name: Union[str, optim.Optimizer],
        obfuscator_optimizer_name: Union[str, optim.Optimizer],
        classifier_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        obfuscator_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        classifier_kwargs: dict = {},
        obfuscator_kwargs: dict = {},
        classifier_optimizer_kwargs: dict = {},
        obfuscator_optimizer_kwargs: dict = {},
        classifier_lr_scheduler_kwargs: dict = {},
        obfuscator_lr_scheduler_kwargs: dict = {},
        classifier_learning_rate: Optional[float] = None,
        obfuscator_learning_rate: Optional[float] = None,
        obfuscator_l2_norm_penalty: float = 1e-2,
        classifier_update_prob: float = 1.0,
        obfuscator_update_prob: float = 1.0
    ):
        for key, val in locals().items():
            if key not in ('key', 'val', 'self'):
                if hasattr(self, key):
                    raise Exception(f'Name clash for variable: {key}={val}')
                setattr(self, key, val)
        super().__init__()
        self.automatic_optimization = False
        self.classifier = models.load(classifier_name, **self.classifier_kwargs)
        self.obfuscator = models.load(obfuscator_name, **self.obfuscator_kwargs)
        for model_name in ('classifier', 'obfuscator'):
            for phase_name in ('train', 'val', 'test'):
                setattr(self, f'{phase_name}_{model_name}_accuracy', Accuracy(task='multitask', num_classes=self.classifier.output_classes))
    
    def _configure_optimizers(self, model_name: str):
        optimizer_name = getattr(self, f'{model_name}_optimizer_name')
        optimizer_kwargs = getattr(self, f'{model_name}_optimizer_kwargs')
        lr_scheduler_name = getattr(self, f'{model_name}_lr_scheduler_name')
        lr_scheduler_kwargs = getattr(self, f'{model_name}_lr_scheduler_kwargs')
        learning_rate = getattr(self, f'{model_name}_learning_rate')
        if learning_rate is None:
            assert 'lr' in optimizer_kwargs.keys()
            learning_rate = optimizer_kwargs['lr']
            setattr(self, f'{model_name}_learning_rate', learning_rate)
        if isinstance(optimizer_name, optim.Optimizer):
            optimizer_constructor = optimizer_name
        else:
            optimizer_constructor = getattr(optim, optimizer_name)
        weight_decay = optimizer_kwargs['weight_decay'] if 'weight_decay' in optimizer_kwargs.keys() else 0.0
        optimizer_kwargs = {key: val for key, val in optimizer_kwargs.items() if key not in ('lr', 'weight_decay')}
        yes_weight_decay, no_weight_decay = [], []
        for name, param in self.model.named_parameters():
            if ('weight' in name) and not('norm' in name):
                yes_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [
            {'params': yes_weight_decay, 'weight_decay': weight_decay},
            {'params': no_weight_decay, 'weight_decay': 0.}
        ]
        optimizer = optimizer_constructor(param_groups, lr=learning_rate, **optimizer_kwargs)
        rv = {'optimizer': optimizer}
        if lr_scheduler_name is not None:
            if isinstance(lr_scheduler_name, optim.lr_scheduler.LRScheduler):
                lr_scheduler_constructor = lr_scheduler_name
            else:
                try:
                    lr_scheduler_constructor = getattr(utils.lr_schedulers, lr_scheduler_name)
                except:
                    lr_scheduler_constructor = getattr(optim.lr_scheduler, lr_scheduler_name)
            lr_scheduler = lr_scheduler_constructor(
                optimizer, total_steps=self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader()), **lr_scheduler_kwargs
            )
            rv.update({'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}})
        return rv
    
    def configure_optimizers(self):
        rv = (self._configure_optimizers('classifier'), self._configure_optimizers('obfuscator'))
        return rv
    
    def training_step(self, batch):
        trace, target = batch
        if len(target.shape) > 1:
            target = target.view(-1, target.size(-1))
        classifier_optimizer, obfuscator_optimizer = self.optimizers()
        
        if NUMPY_RNG.uniform() <= self.classifier_update_prob:
            self.toggle_optimizer(classifier_optimizer)
            with torch.no_grad():
                obfuscated_trace = self.obfuscator(trace)
            logits = self.classifier(obfuscated_trace)
            logits = logits.view(-1, logits.size(-1))
            loss = nn.functional.cross_entropy(logits, target)
            self.train_classifier_accuracy(logits, target)
            self.log('classifier-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('classifier-acc', self.train_classifier_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
            self.untoggle_optimizer(classifier_optimizer)
        
        if NUMPY_RNG.uniform() <= self.obfuscator_update_prob:
            self.toggle_optimizer(obfuscator_optimizer)
            obfuscated_trace = self.obfuscator(trace)
            logits = self.classifier(obfuscated_trace)
            logits = logits.view(-1, logits.size(-1))
            loss = (
                0.5*self.obfuscator_l2_norm_penalty*nn.functional.mse_loss(trace - obfuscated_trace, torch.zeros_like(trace))
                - nn.functional.cross_entropy(logits, target)
            )
            self.train_obfuscator_accuracy(logits, target)
            self.log('obfuscator-loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('obfuscator-acc', self.train_obfuscator_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            obfuscator_optimizer.zero_grad()
            self.manual_backward(loss)
            obfuscator_optimizer.step()
            self.untoggle_optimizer(obfuscator_optimizer)
    
    def validation_step(self, batch):
        trace, target = batch
        if len(target.shape) > 1:
            target = target.view(-1, target.size(-1))
        obfuscated_trace = self.obfuscator(trace)
        clean_logits = self.classifier(trace)
        obfuscated_logits = self.classifier(obfuscated_trace)
        clean_loss = nn.functional.cross_entropy(clean_logits, target)
        obfuscated_loss = nn.functional.cross_entropy(obfuscated_logits, target)