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
        identity_lambda: float = 1e0,
        theta_lr: float = 2e-4,
        gammap_lr: float = 1e-3,
        initial_taup: float = 0.0,
        initial_etap: float = 0.0,
        theta_weight_decay: float = 0.0,
        calibrate_classifiers: bool = False,
        timesteps_per_trace: int = None,
        eps: float = 1e-12,
        train_eta_and_tau: bool = True,
        gaussian_noise_scale: Optional[float] = None,
        mi_decay_rate: Optional[float] = None,
        identity_loss_fn: Literal['l1', 'l2', 'ent', 'logsumexp'] = 'l2',
        direction: Literal['minimax', 'maximin'] = 'maximin'
    ):
        assert timesteps_per_trace is not None
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.classifiers = models.load(self.hparams.classifiers_name, noise_conditional=True, input_shape=(1, self.hparams.timesteps_per_trace), **self.hparams.classifiers_kwargs)
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
            {'optimizer': self.gammap_optimizer, 'lr_scheduler': {'scheduler': self.gammap_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etap_taup_optimizer}
        ]
    
    @torch.no_grad()
    def clip_gammap(self):
        #lim = np.log(self.hparams.eps) - np.log(1-self.hparams.eps)
        #self.gammap.data.clip_(-lim, lim)
        pass
    
    def get_gamma(self):
        return self.hparams.eps + (1-2*self.hparams.eps)*nn.functional.sigmoid(self.gammap)
    
    def get_eta(self):
        return self.hparams.eps + (1-2*self.hparams.eps)*nn.functional.sigmoid(self.etap)
    
    def get_tau(self):
        return self.hparams.eps + nn.functional.softplus(self.taup)

    def rand_like(self, x):
        return self.hparams.eps + (1-2*self.hparams.eps)*torch.rand_like(x)
    
    def get_mutinf(self, logits):
        logits = logits.view(-1, logits.size(-1))
        rv = (
            torch.full((logits.size(0),), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
            + (nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        )
        return rv
        
    def get_identity_loss(self):
        if self.hparams.identity_loss_fn == 'l1':
            if self.hparams.direction == 'minimax':
                return self.get_gamma().sum()
            elif self.hparams.direction == 'maximin':
                return -(1 - self.gamma).sum()
            else:
                raise NotImplementedError
        elif self.hparams.identity_loss_fn == 'l2':
            if self.hparams.direction == 'minimax':
                return 0.5*(self.get_gamma()**2).sum()
            elif self.hparams.direction == 'maximin':
                return -0.5*((1 - self.get_gamma())**2).sum()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    # Uses the REBAR gradient estimator: https://arxiv.org/pdf/1703.07370
    def step_gammap(self, trace, train: bool = False, gradient_estimator: Literal['CONCRETE', 'REBAR'] = 'CONCRETE'):
        self.clip_gammap()
        if train:
            _, gammap_optimizer, etap_taup_optimizer = self.optimizers()
            _, gammap_lr_scheduler = self.lr_schedulers()
            gammap_optimizer.zero_grad()
            etap_taup_optimizer.zero_grad()
        batch_size = trace.size(0)
        gamma = self.get_gamma().unsqueeze(0).repeat(batch_size, 1, 1)
        rv = {}
        if gradient_estimator == 'CONCRETE':
            tau = self.get_tau()
            gammam1 = 1 - gamma
            u = self.rand_like(gammam1)
            z = gammam1.log() - (1-gammam1).log() + u.log() - (1-u).log()
            #z = nn.functional.logsigmoid(-self.gammap) - nn.functional.logsigmoid(self.gammap) + u.log() - (1-u).log()
            rb = nn.functional.sigmoid(z/tau)
            noise = torch.randn_like(trace)
            logits = self.classifiers(rb*trace + (1-rb)*noise, 1-rb)
            mutinf_loss = self.get_mutinf(logits)
            mean_mutinf_loss = mutinf_loss.mean()
            identity_loss = self.get_identity_loss()
            loss = self.hparams.identity_lambda*identity_loss + mean_mutinf_loss
        elif gradient_estimator == 'REBAR': # Based on https://arxiv.org/pdf/1703.07370
            tau = self.get_tau() # renamed from \lambda in the paper since we are already calling the identity penalty coefficient \lambda
            eta = self.get_eta()
            gammam1 = 1 - gamma # these are the Bernoulli parameters so that we can just plug it into the equations from the paper
            u = self.rand_like(gammam1)
            z = nn.functional.logsigmoid(-self.gammap) - nn.functional.logsigmoid(self.gammap) + u.log() - (1-u).log() # z = g(u, \theta)
            b = torch.where(z >= 0, torch.ones_like(z), torch.zeros_like(z)) # b = H(z)
            uprime = 1 - gammam1 # u' such that g(u', \theta) = 0
            v = self.rand_like(gammam1)
            v = torch.where(b == 1, uprime + v*(1 - uprime), v*uprime) # make u and v common random numbers
            z_tilde = torch.where(b == 1, (v.log() - (1-v).log() - (1-gammam1).log()).exp().log1p(), -(v.log() - (1-v).log() - gammam1.log()).exp().log1p()) # \tilde{z} \mid b = \tilde{g}(v, b, \theta)
            rb = nn.functional.sigmoid(z/tau) # \sigma_\lambda(z)
            rb_tilde = nn.functional.sigmoid(z_tilde/tau) # \sigma_\lambda(\tilde{x})
            noise = torch.randn_like(trace)
            with torch.no_grad():
                logits_b = self.classifiers(b*trace + (1-b)*noise, 1-b)
                mutinf_b = self.get_mutinf(logits_b) # f(H(z))
            logits_rb = self.classifiers(rb*trace + (1-rb)*noise, 1-rb)
            mutinf_rb = self.get_mutinf(logits_rb) # f(\sigma_\lambda(z))
            logits_rb_tilde = self.classifiers(rb_tilde*trace + (1-rb_tilde)*noise, 1-rb_tilde)
            mutinf_rb_tilde = self.get_mutinf(logits_rb_tilde) # f(\sigma_\lambda(\tilde{z}))
            log_p_b = (b*gammam1.log() + (1-b)*(1-gammam1).log()).squeeze(1).sum(dim=-1) # \log p(b)
            mutinf_loss = ((mutinf_b - eta*mutinf_rb_tilde).detach()*log_p_b + eta*mutinf_rb - eta*mutinf_rb_tilde)
            mean_mutinf_loss = mutinf_loss.mean()
            identity_loss = self.get_identity_loss() # this is another term in the loss which we compute explicitly rather than with REBAR
            loss = self.hparams.identity_lambda*identity_loss + mean_mutinf_loss
        else:
            assert False
        rv.update({
            'loss': loss, 'mutinf_loss': mean_mutinf_loss, 'identity_loss': identity_loss
        })
        if train:
            mutinf_grad = torch.autograd.grad(mean_mutinf_loss, [self.gammap], retain_graph=True)[0]
            identity_grad = torch.autograd.grad(identity_loss, [self.gammap], retain_graph=gradient_estimator=='REBAR' and self.hparams.train_eta_and_tau)[0]
            rv.update({
                'mutinf_rms_grad': (mutinf_grad.detach()**2).mean().sqrt(),
                'identity_rms_grad': (self.hparams.identity_lambda*identity_grad.detach()**2).mean().sqrt()
            })
            if self.hparams.direction == 'minimax':
                self.gammap.grad = mutinf_grad + self.hparams.identity_lambda*identity_grad
            elif self.hparams.direction == 'maximin':
                self.gammap.grad = -(mutinf_grad + self.hparams.identity_lambda*identity_grad)
            else:
                raise NotImplementedError
            gammap_optimizer.step()
            gammap_lr_scheduler.step()
            if gradient_estimator == 'REBAR' and self.hparams.train_eta_and_tau:
                loss_variance = mutinf_loss.var() # The original paper said variance of the gradient estimator, but that's super expensive to compute + backprop through. I feel like this should have a similar effect.
                self.etap.grad, self.taup.grad = torch.autograd.grad(loss_variance, [self.etap, self.taup])
                etap_taup_optimizer.step()
        assert torch.isfinite(self.etap)
        assert torch.isfinite(self.taup)
        assert torch.all(torch.isfinite(self.gammap))
        return rv
    
    def step_theta(self, trace, label, train=False):
        batch_size = trace.size(0)
        rv = {}
        if train:
            theta_optimizer, *_ = self.optimizers()
            theta_lr_scheduler, *_ = self.lr_schedulers()
        gamma = self.get_gamma().unsqueeze(0).repeat(batch_size, 1, 1)
        trace = trace.repeat(2, 1, 1)
        label = label.repeat(2)
        tau = self.get_tau()
        u = self.rand_like(gamma)
        z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
        hard_b = torch.where(z >= 0, torch.ones_like(gamma), torch.zeros_like(gamma))
        soft_b = nn.functional.sigmoid(z/tau)
        b = torch.cat([hard_b, soft_b], dim=0)
        obfuscated_trace = b*torch.randn_like(trace) + (1-b)*trace
        logits = self.classifiers(obfuscated_trace, b)
        logits = logits.view(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits, label)
        if train:
            theta_optimizer.zero_grad()
            self.manual_backward(loss)
            theta_optimizer.step()
            theta_lr_scheduler.step()
        rv.update({'loss': loss, 'rank': get_rank(logits, label).mean()})
        assert all(torch.all(torch.isfinite(param)) for param in self.classifiers.parameters())
        return rv
    
    def extract_batch(self, batch):
        trace, label = batch
        if self.hparams.gaussian_noise_scale is not None:
            trace = trace + self.hparams.gaussian_noise_scale*torch.randn_like(trace)
        return trace, label
    
    def training_step(self, batch):
        trace, label = self.extract_batch(batch)
        theta_rv = self.step_theta(trace, label, train=True)
        gammap_rv = self.step_gammap(trace, train=True)
        for key, val in theta_rv.items():
            self.log(f'train_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in gammap_rv.items():
            self.log(f'train_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        trace, label = self.extract_batch(batch)
        theta_rv = self.step_theta(trace, label, train=False)
        gammap_rv = self.step_gammap(trace, train=False)
        for key, val in theta_rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        for key, val in gammap_rv.items():
            self.log(f'val_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        tau = self.get_tau()
        eta = self.get_eta()
        self.log('tau', tau, on_step=False, on_epoch=True)
        self.log('eta', eta, on_step=False, on_epoch=True)
        if self.hparams.calibrate_classifiers:
            def get_classifier_args(trace):
                gamma = self.get_gamma().unsqueeze(0).repeat(trace.size(0), 1, 1)
                tau = self.get_tau()
                u = torch.randn_like(gamma)
                z = gamma.log() - (1-gamma).log() + u.log() - (1-u).log()
                hard_b = torch.where(z >= 0, torch.ones_like(gamma), torch.zeros_like(gamma))
                soft_b = nn.functional.sigmoid(z/tau)
                b = torch.cat([hard_b, soft_b], dim=0)
                trace = trace.repeat(2, 1, 1)
                obfuscated_trace = b*torch.randn_like(trace) + (1-b)*trace
                classifier_args = (obfuscated_trace, b)
                return classifier_args
            self.classifiers.calibrate_temperature(
                self.trainer.datamodule.val_dataloader(),
                get_classifier_args
            )
        gamma = self.get_gamma()
        theta_save_dir = os.path.join(self.logger.log_dir, 'gamma_log')
        os.makedirs(theta_save_dir, exist_ok=True)
        np.save(os.path.join(theta_save_dir, f'gamma__step={self.global_step}.npy'), gamma.detach().cpu().numpy().squeeze())