from typing import *
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Subset
import lightning as L
from torchmetrics.classification import Accuracy

from common import *
import models
import utils.lr_schedulers
from utils.metrics import Rank

class DiscreteAdversarialLocalizationTrainer(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        classifier_optimizer_name: Union[str, optim.Optimizer],
        obfuscator_optimizer_name: Union[str, optim.Optimizer],
        obfuscator_l2_norm_penalty: float = 1.0,
        classifier_step_prob: float = 1.0,
        obfuscator_step_prob: float = 1.0,
        split_training_steps: Optional[int] = None, # If not None, we will first train classifier only for this number of steps, then train obfuscator only for this number of steps.
        log_likelihood_baseline_ema: Optional[float] = 0.9,
        classifier_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        obfuscator_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        classifier_kwargs: dict = {},
        classifier_optimizer_kwargs: dict = {},
        obfuscator_optimizer_kwargs: dict = {},
        classifier_lr_scheduler_kwargs: dict = {},
        obfuscator_lr_scheduler_kwargs: dict = {},
        classifier_learning_rate: Optional[float] = None,
        obfuscator_learning_rate: Optional[float] = None,
        obfuscator_batch_size_multiplier: int = 1, # In general the obfuscator can be trained with a larger batch size than the discriminator. We will use the same training data but more binary noise samples.
        normalize_erasure_probs_for_classifier: bool = False,
        additive_noise_augmentation: float = 0.0
    ):
        for key, val in locals().items():
            if key not in ('key', 'val', 'self', '__class__'):
                if hasattr(self, key):
                    raise Exception(f'Name clash for variable: {key}={val}')
                setattr(self, key, val)
        super().__init__()
        self.automatic_optimization = False
        self.classifier = models.load(classifier_name, noise_conditional=True, **self.classifier_kwargs)
        self.unsquashed_obfuscation_weights = nn.Parameter(torch.zeros(self.classifier.input_shape[0], *self.classifier.input_shape[1:], dtype=torch.float32), requires_grad=True)
        if self.log_likelihood_baseline_ema is not None:
            self.register_buffer('log_likelihood_mean', torch.tensor(0.))
        for phase_name in ('train', 'val'):
            for target_name in ('clean', 'obfuscated'):
                setattr(self, f'{phase_name}_{target_name}_accuracy', Accuracy(task='multiclass', num_classes=self.classifier.output_classes))
                setattr(self, f'{phase_name}_{target_name}_rank', Rank())
    
    def _configure_optimizers(self, prefix: Literal['classifier', 'obfuscator']):
        if prefix == 'classifier':
            model = self.classifier
        elif prefix == 'obfuscator':
            model = self.unsquashed_obfuscation_weights
        else:
            assert False
        optimizer_name = getattr(self, f'{prefix}_optimizer_name')
        optimizer_kwargs = getattr(self, f'{prefix}_optimizer_kwargs')
        lr_scheduler_name = getattr(self, f'{prefix}_lr_scheduler_name')
        lr_scheduler_kwargs = getattr(self, f'{prefix}_lr_scheduler_kwargs')
        learning_rate = getattr(self, f'{prefix}_learning_rate')
        if isinstance(optimizer_name, optim.Optimizer):
            optimizer_constructor = optimizer_name
        else:
            assert isinstance(optimizer_name, str)
            optimizer_constructor = getattr(optim, optimizer_name)
        if lr_scheduler_name is None:
            lr_scheduler_constructor = None
        elif isinstance(lr_scheduler_name, optim.lr_scheduler.LRScheduler):
            lr_scheduler_constructor = lr_scheduler_name
        else:
            assert isinstance(lr_scheduler_name, str)
            try:
                lr_scheduler_constructor = getattr(utils.lr_schedulers, lr_scheduler_name)
            except:
                lr_scheduler_constructor = getattr(optim.lr_scheduler, lr_scheduler_name)
        if learning_rate is None:
            assert 'lr' in optimizer_kwargs.keys()
            learning_rate = optimizer_kwargs['lr']
            setattr(self, f'{prefix}_learning_rate', learning_rate)
        if 'weight_decay' in optimizer_kwargs.keys():
            assert prefix == 'classifier'
            weight_decay = optimizer_kwargs['weight_decay']
        else:
            weight_decay = 0.0
        optimizer_kwargs = {key: val for key, val in optimizer_kwargs.items() if key not in ('lr', 'weight_decay')}
        if optimizer_constructor not in [optim.LBFGS]:
            optimizer_kwargs.update({'weight_decay': 0.0})
        if prefix == 'classifier':
            yes_weight_decay, no_weight_decay = [], []
            for name, param in model.named_parameters():
                if ('weight' in name) and not('norm' in name):
                    yes_weight_decay.append(param)
                else:
                    no_weight_decay.append(param)
            param_groups = [{'params': yes_weight_decay, 'weight_decay': weight_decay}, {'params': no_weight_decay, 'weight_decay': 0.}]
            optimizer = optimizer_constructor(param_groups, lr=learning_rate, **optimizer_kwargs)
        else:
            optimizer = optimizer_constructor([model], lr=learning_rate, **optimizer_kwargs)
        if lr_scheduler_constructor is not None:
            if self.split_training_steps is None:
                lr_scheduler = lr_scheduler_constructor(
                    optimizer, total_steps=self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader()), **lr_scheduler_kwargs
                )
            elif prefix == 'classifier':
                lr_scheduler = lr_scheduler_constructor(
                    optimizer, total_steps=self.split_training_steps, **lr_scheduler_kwargs
                )
            elif prefix == 'obfuscator':
                lr_scheduler = lr_scheduler_constructor(
                    optimizer, total_steps=self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader()) - self.split_training_steps, **lr_scheduler_kwargs
                )
        else:
            lr_scheduler = None
        rv = {'optimizer': optimizer}
        if lr_scheduler is not None:
            rv.update({'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}})
        return rv
    
    def configure_optimizers(self):
        rv = (self._configure_optimizers('classifier'), self._configure_optimizers('obfuscator'))
        return rv
    
    def _compute_logits(self, trace: torch.Tensor, noise: torch.Tensor):
        logits = self.classifier(trace, noise)
        logits = logits.view(-1, logits.size(-1))
        return logits
    
    @torch.no_grad()
    def _sample_binary_noise(self, trace: torch.Tensor, normalize: bool = False):
        batch_size = trace.size(0)
        if normalize:
            p = torch.rand(batch_size, 1, 1, device=trace.device, dtype=trace.dtype)
            erasure_probs = p*torch.ones((1, 1, trace.shape[-1]), device=trace.device, dtype=trace.dtype)
        else:
            erasure_probs = nn.functional.sigmoid(self.unsquashed_obfuscation_weights).repeat(batch_size, *(len(trace.shape[1:])*[1]))
        binary_noise = (1 - erasure_probs).bernoulli()
        return binary_noise
    
    #@torch.no_grad()
    #def _obfuscate_trace(self, trace, binary_noise):
    #    return torch.cat([binary_noise*trace, binary_noise], dim=1)
    
    def _classifier_step(self, trace, target, train=False):
        if train:
            classifier_optimizer, _ = self.optimizers()
            if self.classifier_lr_scheduler_name is not None:
                classifier_lr_scheduler, _ = self.lr_schedulers()
            self.toggle_optimizer(classifier_optimizer)
        with torch.set_grad_enabled(train):
            binary_noise = self._sample_binary_noise(trace, self.normalize_erasure_probs_for_classifier)
            if train:
                trace = trace + self.additive_noise_augmentation*torch.randn_like(trace)
            #obfuscated_trace = self._obfuscate_trace(trace, binary_noise)
            logits = self._compute_logits(binary_noise*trace, binary_noise)
            loss = nn.functional.cross_entropy(logits, target, label_smoothing=0.0)
        if train:
            classifier_optimizer.zero_grad()
            self.manual_backward(loss)
            classifier_optimizer.step()
            if self.classifier_lr_scheduler_name is not None:
                classifier_lr_scheduler.step()
            self.untoggle_optimizer(classifier_optimizer)
        return logits, loss

    @torch.no_grad()
    def _obfuscator_step(self, trace, target, train=False, first_batch=False):
        trace = trace.repeat(self.obfuscator_batch_size_multiplier, *((len(trace.shape)-1)*[1]))
        target = target.repeat(self.obfuscator_batch_size_multiplier, *((len(target.shape)-1)*[1]))
        batch_size = target.size(0)
        loss_rv = None

        def closure():
            nonlocal loss_rv
            self.unsquashed_obfuscation_weights.data = torch.clamp(self.unsquashed_obfuscation_weights.data, min=-1e6, max=1e6)
            obfuscation_weights = nn.functional.sigmoid(self.unsquashed_obfuscation_weights)
            l2_norm = obfuscation_weights.norm(p=2)**2
            binary_noise = self._sample_binary_noise(trace)
            logits = self._compute_logits(binary_noise*trace, binary_noise)
            log_likelihood = -nn.functional.cross_entropy(logits, target, reduction='none')
            loss = 0.5*self.obfuscator_l2_norm_penalty*l2_norm + log_likelihood.mean()
            if train: # manually compute gradient
                l2_norm_grad = obfuscation_weights*obfuscation_weights*(1-obfuscation_weights)
                obfuscation_weights = obfuscation_weights.repeat(batch_size, *((len(trace.shape)-1)*[1]))
                log_likelihood = log_likelihood.view(-1, *((len(trace.shape)-1)*[1]))
                if self.log_likelihood_baseline_ema is not None:
                    if not hasattr(self, 'log_likelihood_mean'):
                        self.log_likelihood_mean = log_likelihood.mean()
                    else:
                        self.log_likelihood_mean = (
                            self.log_likelihood_baseline_ema*self.log_likelihood_mean + (1-self.log_likelihood_baseline_ema)*log_likelihood.mean()
                        )
                    log_likelihood -= self.log_likelihood_mean
                log_likelihood_grad = (
                    log_likelihood*((1-binary_noise)*(1-obfuscation_weights) - binary_noise*obfuscation_weights)
                ).mean(dim=0)
                grad = self.obfuscator_l2_norm_penalty*l2_norm_grad + log_likelihood_grad
                self.unsquashed_obfuscation_weights.grad = grad
            loss_rv = loss
            return loss

        if train:
            _, obfuscator_optimizer = self.optimizers()
            if self.obfuscator_lr_scheduler_name is not None:
                scheduler_rv = self.lr_schedulers()
                if self.classifier_lr_scheduler_name is not None:
                    obfuscator_lr_scheduler = scheduler_rv[-1]
                else:
                    obfuscator_lr_scheduler = scheduler_rv
            self.toggle_optimizer(obfuscator_optimizer)
            obfuscator_optimizer.step(closure)
            if self.obfuscator_lr_scheduler_name is not None:
                obfuscator_lr_scheduler.step()
            self.untoggle_optimizer(obfuscator_optimizer)
        if loss_rv is None:
            closure()
        return loss_rv
    
    @torch.no_grad()
    def _dontuse_obfuscator_step(self, trace, target, train=False, first_batch=False):
        trace = trace.repeat(self.obfuscator_batch_size_multiplier, *((len(trace.shape)-1)*[1]))
        target = target.repeat(self.obfuscator_batch_size_multiplier, *((len(target.shape)-1)*[1]))
        batch_size = target.size(0)
        if train:
            _, obfuscator_optimizer = self.optimizers()
            if self.obfuscator_lr_scheduler_name is not None:
                scheduler_rv = self.lr_schedulers()
                if self.classifier_lr_scheduler_name is not None:
                    obfuscator_lr_scheduler = scheduler_rv[-1]
                else:
                    obfuscator_lr_scheduler = scheduler_rv
            self.toggle_optimizer(obfuscator_optimizer)
        obfuscation_weights = nn.functional.sigmoid(self.unsquashed_obfuscation_weights)
        l2_norm = obfuscation_weights.norm(p=2)**2
        if train: l2_norm_grad = obfuscation_weights*obfuscation_weights*(1-obfuscation_weights)
        binary_noise = self._sample_binary_noise(trace)
        obfuscation_weights = obfuscation_weights.repeat(batch_size, *((len(trace.shape)-1)*[1]))
        logits = self._compute_logits(binary_noise*trace, binary_noise)
        log_likelihood = -nn.functional.cross_entropy(logits, target, reduction='none', label_smoothing=0.0)
        if train:
            ll = log_likelihood.view(-1, *((len(trace.shape)-1)*[1]))
            if self.log_likelihood_baseline_ema is not None:
                if first_batch:
                    self.log_likelihood_mean = log_likelihood.mean()
                else:
                    self.log_likelihood_mean = (
                        (self.log_likelihood_baseline_ema)*self.log_likelihood_mean + (1-self.log_likelihood_baseline_ema)*log_likelihood.mean()
                    )
                ll -= self.log_likelihood_mean
            log_likelihood_grad = (
                ll*((1-binary_noise)*(1-obfuscation_weights) - binary_noise*obfuscation_weights)
            ).mean(dim=0)
        loss = 0.5*self.obfuscator_l2_norm_penalty*l2_norm + log_likelihood.mean()
        if train:
            grad = self.obfuscator_l2_norm_penalty*l2_norm_grad + log_likelihood_grad
            self.unsquashed_obfuscation_weights.grad = grad
            obfuscator_optimizer.step()
            if self.obfuscator_lr_scheduler_name is not None:
                obfuscator_lr_scheduler.step()
            self.untoggle_optimizer(obfuscator_optimizer)
        return loss
    
    def training_step(self, batch, batch_idx):
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        trace, target = batch
        if len(target.shape) > 1:
            target = target.view(-1, target.size(-1))
        if self.split_training_steps is not None:
            if self.step_count < self.split_training_steps:
                train_classifier = True
                train_obfuscator = False
            else:
                train_classifier = False
                train_obfuscator = True
        else:
            train_classifier = np.random.rand() < self.classifier_step_prob
            train_obfuscator = np.random.rand() < self.obfuscator_step_prob
        classifier_logits, classifier_loss = self._classifier_step(trace, target, train=train_classifier)
        with torch.no_grad():
            classifier_clean_logits = self._compute_logits(trace, torch.zeros_like(trace))
        obfuscator_loss = self._obfuscator_step(trace, target, train=train_obfuscator, first_batch=batch_idx==0)
        self.train_obfuscated_accuracy(classifier_logits, target)
        self.train_clean_accuracy(classifier_clean_logits, target)
        self.train_obfuscated_rank(classifier_logits, target)
        self.train_clean_rank(classifier_clean_logits, target)
        self.log('classifier-train-loss', classifier_loss, on_epoch=True, prog_bar=True)
        self.log('obfuscator-train-loss', obfuscator_loss, on_epoch=True, prog_bar=True)
        self.log('train-acc', self.train_obfuscated_accuracy, on_epoch=True, prog_bar=False)
        self.log('train-clean-acc', self.train_clean_accuracy, on_epoch=True, prog_bar=False)
        self.log('train-rank', self.train_obfuscated_rank, on_epoch=True, on_step=False, prog_bar=False)
        self.log('train-clean-rank', self.train_clean_rank, on_epoch=True, on_step=False, prog_bar=False)
        self.log('min-obf-weight', nn.functional.sigmoid(self.unsquashed_obfuscation_weights).min(), on_epoch=True, on_step=False, prog_bar=False)
        self.log('max-obf-weight', nn.functional.sigmoid(self.unsquashed_obfuscation_weights).max(), on_epoch=True, on_step=False, prog_bar=False)
        self.log('mean-obf-weight', nn.functional.sigmoid(self.unsquashed_obfuscation_weights).mean(), on_epoch=True, on_step=False, prog_bar=False)
        self.step_count += 1
    
    def validation_step(self, batch):
        trace, target = batch
        if len(target.shape) > 1:
            target = target.view(-1, target.size(-1))
        classifier_logits, classifier_loss = self._classifier_step(trace, target)
        with torch.no_grad():
            classifier_clean_logits = self._compute_logits(trace, torch.zeros_like(trace))
        obfuscator_loss = self._obfuscator_step(trace, target)
        self.val_obfuscated_accuracy(classifier_logits, target)
        self.val_clean_accuracy(classifier_clean_logits, target)
        self.val_obfuscated_rank(classifier_logits, target)
        self.val_clean_rank(classifier_clean_logits, target)
        self.log('classifier-val-loss', classifier_loss, on_epoch=True, prog_bar=True)
        self.log('obfuscator-val-loss', obfuscator_loss, on_epoch=True, prog_bar=True)
        self.log('val-acc', self.val_obfuscated_accuracy, on_epoch=True, prog_bar=False)
        self.log('val-clean-acc', self.val_clean_accuracy, on_epoch=True, prog_bar=False)
        self.log('val-rank', self.val_obfuscated_rank, on_epoch=True, prog_bar=False)
        self.log('val-clean-rank', self.val_clean_rank, on_epoch=True, prog_bar=False)
    
    def on_train_epoch_end(self):
        obfuscation_weights = nn.functional.sigmoid(self.unsquashed_obfuscation_weights).squeeze()
        self.trainer.logger.experiment.add_histogram('obfuscation-weights-hist', obfuscation_weights, self.trainer.current_epoch)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_ylim(0, self.obfuscator_l2_norm_penalty)
        train_dataset = self.trainer.datamodule.train_dataloader().dataset
        while isinstance(train_dataset, Subset):
            train_dataset = train_dataset.dataset
        if hasattr(train_dataset, 'leaking_timestep_count_1o') and (train_dataset.leaking_timestep_count_1o > 0):
            for cycle in train_dataset.leaking_subbytes_cycles:
                ax.axvline(cycle, color='red')
        if hasattr(train_dataset, 'leaking_timestep_count_2o') and (train_dataset.leaking_timestep_count_2o > 0):
            for cycle in train_dataset.leaking_mask_cycles:
                ax.axvline(cycle, color='green')
            for cycle in train_dataset.leaking_masked_subbytes_cycles:
                ax.axvline(cycle, color='orange')
        ax.plot(self.obfuscator_l2_norm_penalty*obfuscation_weights.detach().cpu().numpy(), color='blue', linestyle='none', marker='.', markersize=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Obfuscation weight')
        fig.tight_layout()
        self.trainer.logger.experiment.add_figure('obfuscation-weights', fig, self.trainer.current_epoch)