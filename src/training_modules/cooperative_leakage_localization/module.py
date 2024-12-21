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
        etat_lr_scheduler_kwargs: dict = {},
        theta_lr: float = 2e-4,
        etat_lr: float = 1e-3,
        theta_weight_decay: float = 0.0,
        etat_weight_decay: float = 0.0,
        budget: float = 50.0,
        timesteps_per_trace: Optional[int] = None,
        gradient_estimator: Literal['REINFORCE', 'REBAR'] = 'REBAR',
        noise_scale: Optional[float] = None,
        eps: float = 1e-6,
        classifiers_pretrain_phase: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        if self.hparams.theta_lr_scheduler_name is None:
            self.hparams.theta_lr_scheduler_name = 'NoOpLRSched'
        if self.hparams.etat_lr_scheduler_name is None:
            self.hparams.etat_lr_scheduler_name = 'NoOpLRSched'
        assert self.hparams.timesteps_per_trace is not None
        
        self.cmi_estimator = CondMutInfEstimator(
            self.hparams.classifiers_name,
            input_shape=(1, self.hparams.timesteps_per_trace),
            classifiers_kwargs=self.hparams.classifiers_kwargs
        )
        self.selection_mechanism = SelectionMechanism(
            self.hparams.timesteps_per_trace,
            C=self.hparams.budget
        )
        if self.hparams.gradient_estimator == 'REBAR':
            self.etat = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)
            self.taut = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)
        
    def rand_like(self, x):
        return self.hparams.eps + (1 - 2*self.hparams.eps)*torch.rand_like(x)
        
    def get_eta_and_tau(self): # FIXME: modify this so that we can optimize the eta and tau parameters
        assert self.hparams.gradient_estimator == 'REBAR'
        eta = self.hparams.eps + (1 - 2*self.hparams.eps)*nn.functional.sigmoid(self.etat)
        tau = self.hparams.eps + nn.functional.softplus(self.taut)
        return eta, tau
    
    def configure_optimizers(self):
        self.etat_optimizer = optim.AdamW(
            self.selection_mechanism.parameters(), lr=self.hparams.etat_lr, weight_decay=self.hparams.etat_weight_decay,
            betas=(0.9, 0.99999) if self.hparams.gradient_estimator == 'REBAR' else (0.9, 0.999)
        )
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
        rv = [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etat_optimizer, 'lr_scheduler': {'scheduler': self.etat_lr_scheduler, 'interval': 'step'}}
        ]
        if self.hparams.gradient_estimator == 'REBAR':
            self.rebar_params_optimizer = optim.Adam([self.etat, self.taut], lr=10*self.hparams.etat_lr, betas=(0.9, 0.99999))
            rv.append({'optimizer': self.rebar_params_optimizer})
        return rv

    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def step_etat(self, batch, train: bool = False):
        if train:
            if self.hparams.gradient_estimator == 'REBAR':
                _, etat_optimizer, rebar_params_optimizer = self.optimizers()
                rebar_params_optimizer.zero_grad()
            else:
                _, etat_optimizer = self.optimizers()
            _, etat_lr_scheduler = self.lr_schedulers()
            etat_optimizer.zero_grad()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'REINFORCE':
            batch_size *= 8 # since we aren't backproping through the neural net, we can fit a larger minibatch in memory
            trace = trace.repeat(8, 1, 1)
            label = label.repeat(8)
            alpha = self.selection_mechanism.sample(batch_size)
            log_prob_mass = self.selection_mechanism.log_pmf(alpha)
            with torch.no_grad():
                mutinf = self.cmi_estimator.get_mutinf_estimate(trace, alpha, label)
            if not hasattr(self, 'mutinf_ema'): # control variate to reduce variance of the gradient estimator
                self.mutinf_ema = mutinf.detach().mean()
            else:
                self.mutinf_ema = 0.9*self.mutinf_ema + 0.1*mutinf.detach().mean()
            loss = -((mutinf - self.mutinf_ema)*log_prob_mass).mean()
            with torch.no_grad():
                display_loss = -mutinf.mean()
        elif self.hparams.gradient_estimator == 'REBAR':
            eta, tau = self.get_eta_and_tau()
            log_gamma = self.selection_mechanism.get_log_gamma().unsqueeze(0)
            log_1mgamma = self.selection_mechanism.get_log_1mgamma().unsqueeze(0)
            assert torch.all(torch.isfinite(log_gamma))
            assert torch.all(torch.isfinite(log_1mgamma))
            u = self.rand_like(trace)
            z = log_gamma - log_1mgamma + u.log() - (1-u).log()
            assert torch.all(torch.isfinite(z))
            b = torch.where(z >= 0, torch.ones_like(z), torch.zeros_like(z))
            uprime = 1 - log_gamma.exp()
            v = self.rand_like(trace)
            v = torch.where(b == 1, uprime + v*(1 - uprime), v*uprime)
            v = v.clamp_(self.hparams.eps, 1-self.hparams.eps)
            z_tilde = torch.where(b == 1, (v.log() - (1-v).log() - log_1mgamma).exp().log1p(), -(v.log() - (1-v).log() - log_gamma).exp().log1p())
            assert torch.all(torch.isfinite(z_tilde))
            rb = nn.functional.sigmoid(z/tau)
            rb_tilde = nn.functional.sigmoid(z_tilde/tau)
            with torch.no_grad():
                mutinf_b = self.cmi_estimator.get_mutinf_estimate(trace, b)
            mutinf_rb = self.cmi_estimator.get_mutinf_estimate(trace, rb)
            mutinf_rb_tilde = self.cmi_estimator.get_mutinf_estimate(trace, rb_tilde)
            log_p_b = self.selection_mechanism.log_pmf(b)
            with torch.no_grad():
                display_loss = -mutinf_b.mean()
            if not hasattr(self, 'mutinf_ema'):
                self.mutinf_ema = mutinf_b.detach().mean()
            else:
                self.mutinf_ema = 0.9*self.mutinf_ema + 0.1*mutinf_b.detach().mean()
            mutinf_b = mutinf_b - self.mutinf_ema
            mutinf_rb = mutinf_rb - self.mutinf_ema
            mutinf_rb_tilde = mutinf_rb_tilde - self.mutinf_ema
            loss = -((mutinf_b - eta*mutinf_rb_tilde).detach()*log_p_b + eta*mutinf_rb - eta*mutinf_rb_tilde).mean()
        else:
            assert False
        rv.update({'loss': display_loss.detach()})
        if train:
            if self.hparams.gradient_estimator == 'REBAR':
                (etat_grad,) = torch.autograd.grad(loss, self.selection_mechanism.etat, create_graph=True)
                self.selection_mechanism.etat.grad = etat_grad
                rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
                if not hasattr(self, 'etat_grad_ema'):
                    self.etat_grad_ema = etat_grad.detach()
                else:
                    self.etat_grad_ema = 0.95*self.etat_grad_ema + 0.05*etat_grad.detach()
                rebar_params_loss = ((etat_grad - self.etat_grad_ema)**2).mean()
                (self.etat.grad, self.taut.grad) = torch.autograd.grad(rebar_params_loss, [self.etat, self.taut])
                etat_optimizer.step()
                etat_lr_scheduler.step()
                rebar_params_optimizer.step()
            else:
                self.manual_backward(loss)
                rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
                etat_optimizer.step()
                etat_lr_scheduler.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.selection_mechanism.parameters())
        return rv
    
    def step_theta(self, batch, train: bool = False):
        if train:
            theta_optimizer, *_ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'REBAR': # we need to train it on the interpolated masks, not just hard masks
            _, tau = self.get_eta_and_tau()
            log_gamma = self.selection_mechanism.get_log_gamma().unsqueeze(0)
            log_1mgamma = self.selection_mechanism.get_log_1mgamma().unsqueeze(0)
            u = self.rand_like(trace)
            z = log_gamma - log_1mgamma + u.log() - (1-u).log()
            z_b = z[:batch_size//2]
            z_rb = z[batch_size//2:]
            b = torch.where(z_b >= 0, torch.ones_like(z_b), torch.zeros_like(z_b))
            rb = nn.functional.sigmoid(z_rb/tau)
            alpha = torch.cat([b, rb], dim=0)
        else:
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
        if not self.hparams.classifiers_pretrain_phase:
            etat_rv = self.step_etat(batch, train=True)
            for key, val in etat_rv.items():
                self.log(f'train_etat_{key}', val, on_step=False, on_epoch=True)
        for key, val in theta_rv.items():
            self.log(f'train_theta_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        theta_rv = self.step_theta(batch, train=False)
        if not self.hparams.classifiers_pretrain_phase:
            etat_rv = self.step_etat(batch, train=False)
            for key, val in etat_rv.items():
                self.log(f'val_etat_{key}', val, on_step=False, on_epoch=True)
        for key, val in theta_rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        log_gamma = self.selection_mechanism.get_log_gamma().detach().cpu().numpy().squeeze()
        log_gamma_save_dir = os.path.join(self.logger.log_dir, 'log_gamma_over_time')
        os.makedirs(log_gamma_save_dir, exist_ok=True)
        np.save(os.path.join(log_gamma_save_dir, f'log_gamma__step={self.global_step}.npy'), log_gamma)