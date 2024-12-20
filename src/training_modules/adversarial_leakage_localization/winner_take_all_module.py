from typing import *
import numpy as np
import torch
from torch import nn, optim
import lightning as L

from common import *
import utils.lr_schedulers
from utils.metrics import get_rank
import models
from models.calibrated_model import CalibratedModel

class AdversarialLeakageLocalizationModule(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_lr_scheduler_name: str = None,
        theta_lr_scheduler_kwargs: dict = {},
        gammap_lr_scheduler_name: str = None,
        gammap_lr_scheduler_kwargs: dict = {},
        theta_lr: float = 2e-4,
        gammap_lr: float = 1e-2,
        gumbel_tau: Union[float, Callable] = 1.0,
        occluded_point_count: int = 1,
        theta_weight_decay: float = 0.0,
        calibrate_classifiers: bool = False,
        timesteps_per_trace: int = None,
        entropy_weight: float = 1.0,
        eps: float = 1e-12,
        gradient_estimator: Literal['CONCRETE', 'REINFORCE'] = 'CONCRETE'
    ):
        assert timesteps_per_trace is not None
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.gumbel_tau = gumbel_tau
        self.classifiers = models.load(self.hparams.classifiers_name, noise_conditional=True, input_shape=(1, self.hparams.timesteps_per_trace), **self.hparams.classifiers_kwargs)
        if self.hparams.calibrate_classifiers:
            self.classifiers = CalibratedModel(self.classifiers)
        self.gammap = nn.Parameter(0.01*torch.randn((self.hparams.occluded_point_count, 1, self.hparams.timesteps_per_trace), dtype=torch.float), requires_grad=True)
    
    def configure_optimizers(self):
        self.gammap_optimizer = optim.Adam([self.gammap], lr=self.hparams.gammap_lr)
        theta_yes_weight_decay, theta_no_weight_decay = [], []
        for name, param in self.classifiers.named_parameters():
            if ('weight' in name) and not('norm' in name):
                theta_yes_weight_decay.append(param)
            else:
                theta_no_weight_decay.append(param)
        theta_param_groups = [{'params': theta_yes_weight_decay, 'weight_decay': self.hparams.theta_weight_decay}, {'params': theta_no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim.AdamW(theta_param_groups, lr=self.hparams.theta_lr)
        theta_lr_scheduler_constructor, gammap_lr_scheduler_constructor = map(
            lambda x: (
                x if isinstance(x, (optim.lr_scheduler.LRScheduler, type(None)))
                else getattr(utils.lr_schedulers, x) if hasattr(utils.lr_schedulers, x)
                else getattr(optim.lr_scheduler, x)
            ), (self.hparams.theta_lr_scheduler_name, self.hparams.gammap_lr_scheduler_name)
        )
        step_count = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        if theta_lr_scheduler_constructor is not None:
            self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=step_count, **self.hparams.theta_lr_scheduler_kwargs)
        else:
            self.theta_lr_scheduler = optim.lr_scheduler.LambdaLR(self.theta_optimizer, lr_lambda=lambda _: 1.0)
        if gammap_lr_scheduler_constructor is not None:
            self.gammap_lr_scheduler = gammap_lr_scheduler_constructor(self.gammap_optimizer, total_steps=step_count, **self.hparams.gammap_lr_scheduler_kwargs)
        else:
            self.gammap_lr_scheduler = optim.lr_scheduler.LambdaLR(self.gammap_optimizer, lr_lambda=lambda _: 1.0)
        return [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.gammap_optimizer, 'lr_scheduler': {'scheduler': self.gammap_lr_scheduler, 'interval': 'step'}}
        ]

    def rand_like(self, x):
        return self.hparams.eps + (1-2*self.hparams.eps)*torch.rand_like(x)
    
    def get_mutinf(self, logits, labels):
        rv = (
            torch.full((logits.size(0),), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
            + (nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        )
        return rv
    
    def sample_mask(self, batch_size):
        if isinstance(self.gumbel_tau, float):
            tau = self.gumbel_tau
        else:
            if not hasattr(self, 'total_steps'):
                assert self.trainer.max_epochs is not None
                self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
            tau = self.gumbel_tau(self.global_step / self.total_steps)
        logits = self.gammap.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        masks = nn.functional.gumbel_softmax(
            logits.squeeze(-2), tau=tau, dim=-1, hard=self.hparams.gradient_estimator == 'REINFORCE'
        ).unsqueeze(-2)
        masks = masks.sum(dim=1).clamp(max=1.0)
        return masks
    
    def get_entropy(self):
        p = nn.functional.softmax(self.gammap.squeeze(-2), dim=-1).mean(dim=0)
        log_p = torch.logsumexp(self.gammap.squeeze(-2), 0) - torch.tensor(np.log(self.hparams.occluded_point_count), dtype=torch.float, device=self.device)
        return -(p*log_p).sum()
    
    def step_gammap(self, trace, labels, train: bool = False):
        if train:
            _, gammap_optimizer = self.optimizers()
            _, gammap_lr_scheduler = self.lr_schedulers()
            gammap_optimizer.zero_grad()
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'CONCRETE':
            mask = self.sample_mask(batch_size)
            noise = torch.randn_like(trace)
            classifier_logits = self.classifiers(mask*trace + (1-mask)*noise, mask)
            classifier_logits = classifier_logits.view(-1, classifier_logits.size(-1))
            mutinf_loss = -self.get_mutinf(classifier_logits, labels).mean()
            entropy_loss = -self.get_entropy()
            loss = mutinf_loss + self.hparams.entropy_weight*entropy_loss
        else:
            raise NotImplementedError
        rv.update({
            'loss': loss, 'mutinf_loss': mutinf_loss, 'identity_loss': entropy_loss
        })
        if train:
            self.manual_backward(loss)
            gammap_optimizer.step()
            gammap_lr_scheduler.step()
        assert torch.all(torch.isfinite(self.gammap))
        return rv
    
    def step_theta(self, trace, label, train: bool = False):
        batch_size = trace.size(0)
        rv = {}
        if train:
            theta_optimizer, _ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
            theta_optimizer.zero_grad()
        mask = self.sample_mask(batch_size)
        noise = torch.randn_like(trace)
        logits = self.classifiers(mask*trace + (1-mask)*noise, mask)
        logits = logits.view(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits, label)
        if train:
            self.manual_backward(loss)
            theta_optimizer.step()
            theta_lr_scheduler.step()
        rv.update({'loss': loss, 'rank': get_rank(logits, label).mean()})
        return rv
    
    def extract_batch(self, batch):
        trace, label = batch
        return trace, label
    
    def training_step(self, batch):
        trace, label = self.extract_batch(batch)
        theta_rv = self.step_theta(trace, label, train=True)
        gammap_rv = self.step_gammap(trace, label, train=True)
        for key, val in theta_rv.items():
            self.log(f'train_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in gammap_rv.items():
            self.log(f'train_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        trace, label = self.extract_batch(batch)
        theta_rv = self.step_theta(trace, label, train=False)
        gammap_rv = self.step_gammap(trace, label, train=False)
        for key, val in theta_rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in gammap_rv.items():
            self.log(f'val_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        if self.hparams.calibrate_classifiers:
            raise NotImplementedError
        gamma = nn.functional.softmax(self.gammap.squeeze(-2), dim=-1).mean(dim=0)
        theta_save_dir = os.path.join(self.logger.log_dir, 'gamma_log')
        os.makedirs(theta_save_dir, exist_ok=True)
        np.save(os.path.join(theta_save_dir, f'gamma__step={self.global_step}.npy'), gamma.detach().cpu().numpy().squeeze())