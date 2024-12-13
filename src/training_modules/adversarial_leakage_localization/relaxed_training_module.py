import numpy as np
import torch
from torch import nn, optim
import lightning as L

from common import *
import utils.lr_schedulers
from utils.metrics import get_rank
import models
from models.calibrated_model import CalibratedModel

class AdversarialLeakageLocalization(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_lr_scheduler_name: str = None,
        theta_lr_scheduler_kwargs: dict = {},
        gammap_lr_scheduler_name: str = None,
        gammap_lr_scheduler_kwargs: dict = {},
        identity_lambda: float = 1e0,
        theta_lr: float = 2e-4,
        gammap_lr: float = 1e-3,
        initial_taup: float = 0.0,
        initial_etap: float = 0.0,
        train_taup_etap: bool = False,
        theta_weight_decay: float = 0.0,
        calibrate_classifiers: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.classifiers = models.load(self.hparams.classifiers_name, noise_conditional=True, **self.hparams.classifiers_kwargs)
        if self.hparams.calibrate_classifiers:
            self.classifiers = CalibratedModel(self.classifiers)
        self.gammap = nn.Parameter(torch.zeros((1, self.hparams.timesteps_per_trace), dtype=torch.float), requires_grad=True)
        self.etap = nn.Parameter(torch.tensor(self.hparams.initial_etap, dtype=torch.float), requires_grad=True)
        self.taup = nn.Parameter(torch.tensor(self.hparams.initial_taup, dtype=torch.float), requires_grad=True)
    
    def configure_optimizers(self):
        self.etap_taup_optimizer = optim.Adam([self.etap, self.taup], lr=10*self.hparams.gammap_lr, betas=(0.9, 0.99999))
        self.gammap_optimizer = optim.Adam([self.gammap], lr=self.hparams.gammap_lr, betas=(0.9, 0.99999))
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
        step_count = None
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
            {'optimizer': self.gammap_optimizer, 'lr_scheduler': {'scheduler': self.gammap_lr_scheduler, 'interval': 'step'}},
            self.etap_taup_optimizer
        ]
    
    def get_gamma(self):
        return nn.functional.sigmoid(self.gammap)
    
    def get_eta(self):
        return nn.functional.sigmoid(self.etap)
    
    def get_tau(self):
        return nn.functional.softplus(self.taup)
    
    def get_mutinf(self, logits):
        return (
            torch.full((logits.size(0)), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
            + (nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        )
        
    def get_identity_loss(self):
        return 0.5*(self.get_gamma()**2).sum()
    
    # Uses the REBAR gradient estimator: https://arxiv.org/pdf/1703.07370
    def step_gammap(self, trace, train: bool = False, gradient_estimator: Literal['CONCRETE', 'REBAR'] = 'REBAR'):
        if train:
            _, gammap_optimizer, etap_taup_optimizer = self.optimizers()
            _, gammap_lr_scheduler = self.lr_schedulers()
        gamma = self.get_gamma()
        rv = {}
        if gradient_estimator == 'CONCRETE':
            tau = self.get_tau()
            u = torch.rand_like(gamma)
            z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
            soft_b = nn.functional.sigmoid(z/tau)
            noise = torch.randn_like(trace)
            obfuscated_trace = soft_b*noise + (1-soft_b)*trace
            logits = self.classifiers(obfuscated_trace, soft_b)
            mutinf = self.get_mutinf(logits)
            mutinf_loss = mutinf.mean()
            identity_loss = self.get_identity_loss()
            loss = self.hparams.identity_lambda*identity_loss + mutinf_loss
        elif gradient_estimator == 'REBAR':
            tau = self.get_tau()
            eta = self.get_eta()
            u = torch.rand_like(gamma)
            v = torch.rand_like(gamma)
            v = torch.where(u <= gamma, v*gamma, gamma + v*(1-gamma))
            z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
            b = torch.where(z >= 0, torch.ones_like(gamma), torch.zeros_like(gamma))
            z_tilde = torch.where(
                b == 1, (v.log() - (1-v).log() - (1-gamma).log()).exp().log1p(), -(v.log() - (1-v).log() - gamma.log()).exp().log1p()
            )
            soft_b = nn.functional.sigmoid(z/tau)
            soft_b_tilde = nn.functional.sigmoid(z_tilde/tau)
            noise = torch.randn_like(trace)
            hard_obfuscated_trace = b*noise + (1-b)*trace
            soft_obfuscated_trace = soft_b*noise + (1-soft_b)*trace
            soft_obfuscated_trace_tilde = soft_b_tilde*noise + (1-soft_b_tilde)*trace
            with torch.no_grad():
                hard_logits = self.classifiers(hard_obfuscated_trace, b)
                hard_mutinf = self.get_mutinf(hard_logits)
            soft_logitss = self.classifiers(
                torch.cat([soft_obfuscated_trace, soft_obfuscated_trace_tilde], dim=0), torch.cat([soft_b, soft_b_tilde], dim=0)
            )
            soft_logits = soft_logitss[:soft_logitss.size(0)//2, ...]
            soft_logits_tilde = soft_logitss[soft_logitss.size(0)//2:, ...]
            soft_mutinf = self.get_mutinf(soft_logits)
            soft_mutinf_tilde = self.get_mutinf(soft_logits_tilde)
            log_p_b = ((1-b)*gamma.log() + b*(1-gamma).log()).sum(dim=-1)
            mutinf_loss = (hard_mutinf - eta*soft_mutinf_tilde).detach()*log_p_b + eta*soft_mutinf - eta*soft_mutinf_tilde
            identity_loss = self.get_identity_loss()
            loss = self.hparams.identity_lambda*identity_loss + mutinf_loss
        else:
            assert False
        rv.update({
            'loss': loss, 'mutinf_loss': mutinf_loss, 'identity_loss': identity_loss
        })
        if train:
            gammap_optimizer.zero_grad()
            self.manual_backward(loss)
            gammap_optimizer.step()
            gammap_lr_scheduler.step()
        return rv
    
    def step_theta(self, trace, label, train=False):
        rv = {}
        if train:
            theta_optimizer, *_ = self.optimizers()
            theta_lr_scheduler, *_ = self.lr_schedulers()
        gamma = self.get_gamma()
        tau = self.get_tau()
        u = torch.rand_like(gamma)
        z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
        hard_b = torch.where(z >= 0, torch.ones_like(gamma), torch.zeros_like(gamma))
        soft_b = nn.functional.sigmoid(z/tau)
        b = torch.cat([hard_b, soft_b], dim=0)
        obfuscated_trace = b*torch.randn_like(trace) + (1-b)*trace
        logits = self.classifiers(obfuscated_trace, b)
        loss = nn.functional.cross_entropy(logits, label)
        if train:
            theta_optimizer.zero_grad()
            self.manual_backward(loss)
            theta_optimizer.step()
            theta_lr_scheduler.step()
        rv.update({'loss': loss, 'rank': get_rank(logits, label).mean()})
        return rv
    
    def train_step(self, batch):
        trace, label = batch
        theta_rv = self.step_theta(trace, label, train=True)
        gammap_rv = self.step_gammap(trace, train=True)
        for key, val in theta_rv.items():
            self.log(f'train_theta_{key}', val, on_step=True, on_epoch=False)
        for key, val in gammap_rv.items():
            self.log(f'train_gammap_{key}', val, on_step=True, on_epoch=False)
    
    def val_step(self, batch):
        trace, label = batch
        theta_rv = self.step_theta(trace, label, train=False)
        gammap_rv = self.step_gammap(trace, train=False)
        for key, val in theta_rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in gammap_rv.items():
            self.log(f'val_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        gamma = self.get_gamma()
        if self.hparams.calibrate_classifiers:
            def get_classifier_args(trace):
                tau = self.get_tau()
                u = torch.randn_like(gamma)
                z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
                hard_b = torch.where(z >= 0, torch.ones_like(gamma), torch.zeros_like(gamma))
                soft_b = nn.functional.sigmoid(z/tau)
                b = torch.cat([hard_b, soft_b], dim=0)
                obfuscated_trace = b*torch.randn_like(trace) + (1-b)*trace
                classifier_args = (obfuscated_trace, b)
                return classifier_args
            self.classifiers.calibrate_temperature(
                self.trainer.datamodule.val_dataloader(),
                get_classifier_args
            )
        theta_save_dir = os.path.join(self.logger.log_dir, 'gamma_log')
        os.makedirs(theta_save_dir, exist_ok=True)
        np.save(os.path.join(theta_save_dir, f'gamma__step={self.global_step}.npy'), gamma.detach().cpu().numpy().squeeze())