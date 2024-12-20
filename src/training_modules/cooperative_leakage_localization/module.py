import numpy as np
import torch
from torch import nn, optim
import lightning as L

from common import *
import models
from .utils import *
import utils.lr_schedulers
from utils.metrics import get_rank

class Module(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_lr_scheduler_name: str = None,
        theta_lr_scheduler_kwargs: dict = {},
        etat_lr_scheduler_name: str = None,
        etat_lr_scheduler_kwargs: str = None,
        theta_lr: float = 2e-4,
        eteat_lr: float = 2e-4,
        theta_weight_decay: float = 0.0,
        etat_weight_decay: float = 0.0,
        budget: float = 1.0,
        timesteps_per_trace: Optional[int] = None,
        gradient_estimator: Literal['REINFORCE'] = 'REINFORCE',
        noise_scale: Optional[float] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        if self.hparams.theta_lr_scheduler_name is None:
            self.hparams.theta_lr_scheduler_name = 'NoOpLRSched'
        if self.hparams.etat_lr_scheduler_name is None:
            self.hparams.etat_lr_scheduler_name = 'NoOpLrSched'
        assert self.hparams.timesteps_per_trace is not None
        
        self.cmi_estimator = CondMutInfEstimator(
            self.hparams.classifiers_name,
            input_shape=(1, self.hparams.timesteps_per_trace),
            **self.hparams.classifiers_kwargs
        )
        self.selection_mechanism = SelectionMechanism(
            self.hparams.timesteps_per_trace,
            C=self.hparams.budget
        )
    
    def configure_optimizers(self):
        self.etat_optimizer = optim.AdamW(self.selection_mechanism.parameters(), lr=self.hparams.etat_lr, weight_decay=self.hparams.etat_weight_decay)
        theta_yes_weight_decay, theta_no_weight_decay = [], []
        for name, param in self.cmi_estimator.named_parameters():
            if ('weight' in name) and not('norm' in name):
                theta_yes_weight_decay.append(param)
            else:
                theta_no_weight_decay.append(param)
        theta_param_groups = [{'params': theta_yes_weight_decay, 'weight_decay': self.hparams.theta_weight_decay}, {'params': theta_no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim.AdamW(theta_param_groups, lr=self.hparams.theta_lr)
        theta_lr_scheduler_constructor, etat_lr_scheduler_constructor = map(
            lambda x: (
                x if isinstance(x, (optim.lr_scheduler.LRScheduler))
                else getattr(utils.lr_schedulers, x) if hasattr(utils.lr_schedulers, x)
                else getattr(optim.lr_scheduler, x)
            ), (self.hparams.theta_lr_scheduler_name, self.hparams.etat_lr_scheduler_name)
        )
        if self.trainer.max_epochs is not None:
            total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            total_steps = self.trainer.max_steps
        else:
            assert False
        self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=total_steps, **self.hparams.theta_lr_scheduler_kwargs)
        self.etat_lr_scheduler = etat_lr_scheduler_constructor(self.etat_optimizer, total_steps=total_steps, **self.hparams.etat_lr_scheduler_kwargs)
        return [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etat_optimizer, 'lr_scheduler': {'scheduler': self.etat_lr_scheduler, 'interval': 'step'}}
        ]

    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def step_etat(self, batch, train: bool = False):
        if train:
            _, etat_optimizer = self.optimizers()
            _, etat_lr_scheduler = self.lr_schedulers()
            etat_optimizer.zero_grad()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'REINFORCE':
            alpha = self.selection_mechanism.sample(batch_size)
            log_prob_mass = self.selection_mechanism.log_pmf(alpha)
            with torch.no_grad():
                mutinf = self.cmi_estimator.get_mutinf_estimate(trace, alpha, label)
            loss = -(mutinf*log_prob_mass).mean()
        else:
            assert False
        rv.update({'loss': loss.detach()})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
            etat_optimizer.step()
            etat_lr_scheduler.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.selection_mechanism.parameters())
        return rv
    
    def step_theta(self, batch, train: bool = False):
        if train:
            theta_optimizer, _ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        alpha = self.selection_mechanism.sample(batch_size)
        logits = self.cmi_estimator.get_logits(trace, alpha)
        loss = nn.functional.cross_entropy(logits, label)
        rv.update({'loss': loss.detach(), 'rank': get_rank(logits, label).mean()})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.cmi_estimator)})
            theta_optimizer.step()
            theta_lr_scheduler.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.cmi_estimator.parameters())
        return rv
    
    def training_step(self, batch):
        theta_rv = self.step_theta(batch, train=True)
        etat_rv = self.step_etat(batch, train=True)
        for key, val in theta_rv.items():
            self.log(f'train_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in etat_rv.items():
            self.log(f'train_etat_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        theta_rv = self.step_theta(batch, train=False)
        etat_rv = self.step_etat(batch, train=False)
        for key, val in theta_rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in etat_rv.items():
            self.log(f'val_etat_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        log_gamma = self.selection_mechanism.get_log_gamma().detach().cpu().numpy().squeeze()
        log_gamma_save_dir = os.path.join(self.logger.log_dir, 'log_gamma_over_time')
        os.makedirs(log_gamma_save_dir, exist_ok=True)
        np.save(os.path.join(log_gamma_save_dir, f'log_gamma__step={self.global_step}.npy'), log_gamma)