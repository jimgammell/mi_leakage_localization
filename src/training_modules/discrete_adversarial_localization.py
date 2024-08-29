# To do:
#   - Implement different batch sizes for the classifier and obfuscator dataloaders

from typing import *
import numpy as np
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.classification import Accuracy

from common import *
import models
import utils.lr_schedulers

class DiscreteAdversarialLocalizationTrainer(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        classifier_optimizer_name: Union[str, optim.Optimizer],
        obfuscator_optimizer_name: Union[str, optim.Optimizer],
        classifier_lr_scheduler: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        obfuscator_lr_scheduler: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        classifier_kwargs: dict = {},
        classifier_optimizer_kwargs: dict = {},
        obfuscator_optimizer_kwargs: dict = {},
        classifier_lr_scheduler_kwargs: dict = {},
        obfuscator_lr_scheduler_kwargs: dict = {},
        classifier_learning_rate: Optional[float] = None,
        obfuscator_learning_rate: Optional[float] = None
    ):
        for key, val in locals().items():
            if key not in ('key', 'val', 'self'):
                if hasattr(self, key):
                    raise Exception(f'Name clash for variable: {key}={val}')
                setattr(self, key, val)
        self.automatic_optimization = False
        self.classifier = models.load(classifier_name, **self.classifier_kwargs)
        self.unsquashed_obfuscation_weights = nn.Parameter(torch.zeros(*self.classifier.input_shape, dtype=torch.float32), requires_grad=True)
        for phase_name in ('train', 'val'):
            for target_name in ('clean', 'obfuscated'):
                setattr(self, f'{phase_name}_{target_name}_accuracy', Accuracy(task='multitask', num_classes=self.classifier.output_classes))
    
    def _compute_logits(self, trace):
        logits = self.classifier(trace)
        logits = logits.view(-1, logits.size(-1))
        return logits
    
    @torch.no_grad()
    def _sample_binary_noise(self, trace):
        batch_size = trace.size(0)
        return nn.functional.sigmoid(self.unsquashed_obfuscation_weights).repeat(batch_size, *((len(trace.shape)-1)*[1])).bernoulli()
    
    @torch.no_grad()
    def _obfuscate_trace(self, trace, binary_noise):
        return torch.cat([binary_noise*trace, binary_noise], dim=1)
    
    def _classifier_training_step(self, trace, target):
        classifier_optimizer, _ = self.optimizers()
        classifier_lr_scheduler, _ = self.lr_schedulers()
        self.toggle_optimizer(classifier_optimizer)
        binary_noise = self._sample_binary_noise(trace)
        obfuscated_trace = self._obfuscate_trace(trace, binary_noise)
        logits = self._compute_logits(obfuscated_trace)
        loss = nn.functional.cross_entropy(logits, target)
        classifier_optimizer.zero_grad()
        self.manual_backward(loss)
        classifier_optimizer.step()
        classifier_lr_scheduler.step()
        self.untoggle_optimizer(classifier_optimizer)
        return logits, loss

    @torch.no_grad()
    def _obfuscator_training_step(self, trace, target):
        batch_size = target.size(0)
        _, obfuscator_optimizer = self.optimizers()
        _, obfuscator_lr_scheduler = self.lr_schedulers()
        self.toggle_optimizer(obfuscator_optimizer)
        obfuscation_weights = nn.functional.sigmoid(self.unsquashed_obfuscation_weights)
        l2_norm = obfuscation_weights.norm(p=2)**2
        l2_norm_grad = obfuscation_weights*obfuscation_weights*(1-obfuscation_weights)
        binary_noise = self._sample_binary_noise(trace)
        obfuscation_weights = obfuscation_weights.repeat(batch_size, *((len(trace.shape)-1)*[1]))
        obfuscated_trace = self._obfuscate_trace(trace, binary_noise)
        logits = self._compute_logits(obfuscated_trace)
        log_likelihood = -nn.functional.cross_entropy(logits, target, reduction='none')
        log_likelihood_grad = (
            log_likelihood*((1-binary_noise)*(1-obfuscation_weights) - binary_noise*obfuscation_weights)
        ).mean(dim=0)
        loss = 0.5*self.obfuscator_l2_norm_penalty*l2_norm + log_likelihood.mean()
        grad = self.obfuscator_l2_norm_penalty*l2_norm_grad + log_likelihood_grad
        self.unsquashed_obfuscation_weights.grad = grad
        obfuscator_optimizer.step()
        obfuscator_lr_scheduler.step()
        self.untoggle_optimizer(obfuscator_optimizer)
        return loss
    
    def training_step(self, batch):
        trace, target = batch
        if len(target.shape) > 1:
            target = target.view(-1, target.size(-1))
        classifier_logits, classifier_loss = self._classifier_training_step(trace, target)
        with torch.no_grad():
            classifier_clean_logits = self._compute_logits(torch.cat([trace, torch.zeros_like(trace)], dim=1))
        obfuscator_loss = self._obfuscator_training_step(trace, target)
        self.train_obfuscated_accuracy(classifier_logits, target)
        self.train_clean_accuracy(classifier_clean_logits, target)
        self.log('classifier-train-loss', classifier_loss, on_epoch=True, prog_bar=True)
        self.log('obfuscator-train-loss', obfuscator_loss, on_epoch=True, prog_bar=True)
        self.log('train-acc', self.train_obfuscated_accuracy, on_epoch=True, prog_bar=True)
        self.log('train-clean-acc', self.train_clean_accuracy, on_epoch=True, prog_bar=True)