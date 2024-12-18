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
        psi_lr_scheduler_name: str = None,
        psi_lr_scheduler_kwargs: dict = {},
        identity_lambda: float = 1e0,
        tau: float = 1.0,
        theta_lr: float = 2e-4,
        psi_lr: float = 2e-4,
        theta_weight_decay: float = 0.0,
        psi_weight_decay: float = 0.0,
        calibrate_classifiers: bool = False,
        timesteps_per_trace: int = None,
        eps: float = 1e-12,
        gaussian_noise_scale: Optional[float] = None,
        gradient_estimator: Literal['REINFORCE', 'CONCRETE'] = 'CONCRETE',
        mask_intrinsic_dim: int = 512
    ):
        assert timesteps_per_trace is not None
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.classifiers = models.load(self.hparams.classifiers_name, noise_conditional=True, input_shape=(1, self.hparams.timesteps_per_trace), **self.hparams.classifiers_kwargs)
        if self.hparams.calibrate_classifiers:
            self.classifiers = CalibratedModel(self.classifiers)
        self.noise_generator = nn.Sequential(
            nn.Linear(self.hparams.mask_intrinsic_dim, self.hparams.mask_intrinsic_dim),
            nn.SELU(),
            nn.Linear(self.hparams.mask_intrinsic_dim, self.hparams.timesteps_per_trace)
        )
    
    def configure_optimizers(self):
        psi_yes_weight_decay, psi_no_weight_decay = [], []
        for name, param in self.noise_generator.named_parameters():
            if ('weight' in name) and not('norm' in name):
                psi_yes_weight_decay.append(param)
            else:
                psi_no_weight_decay.append(param)
        psi_param_groups = [{'params': psi_yes_weight_decay, 'weight_decay': self.hparams.psi_weight_decay}, {'params': psi_no_weight_decay, 'weight_decay': 0.0}]
        self.psi_optimizer = optim.AdamW(psi_param_groups, lr=self.hparams.psi_lr)
        theta_yes_weight_decay, theta_no_weight_decay = [], []
        for name, param in self.classifiers.named_parameters():
            if ('weight' in name) and not('norm' in name):
                theta_yes_weight_decay.append(param)
            else:
                theta_no_weight_decay.append(param)
        theta_param_groups = [{'params': theta_yes_weight_decay, 'weight_decay': self.hparams.theta_weight_decay}, {'params': theta_no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim.AdamW(theta_param_groups, lr=self.hparams.theta_lr)
        theta_lr_scheduler_constructor, psi_lr_scheduler_constructor = map(
            lambda x: (
                x if isinstance(x, (optim.lr_scheduler.LRScheduler, type(None)))
                else getattr(utils.lr_schedulers, x) if hasattr(utils.lr_schedulers, x)
                else getattr(optim.lr_scheduler, x)
            ), (self.hparams.theta_lr_scheduler_name, self.hparams.psi_lr_scheduler_name)
        )
        step_count = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        if theta_lr_scheduler_constructor is not None:
            self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=step_count, **self.hparams.theta_lr_scheduler_kwargs)
        else:
            self.theta_lr_scheduler = optim.lr_scheduler.LambdaLR(self.theta_optimizer, lr_lambda=lambda _: 1.0)
        if psi_lr_scheduler_constructor is not None:
            self.psi_lr_scheduler = psi_lr_scheduler_constructor(self.psi_optimizer, total_steps=step_count, **self.hparams.psi_lr_scheduler_kwargs)
        else:
            self.psi_lr_scheduler = optim.lr_scheduler.LambdaLR(self.psi_optimizer, lr_lambda=lambda _: 1.0)
        return [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.psi_optimizer, 'lr_scheduler': {'scheduler': self.psi_lr_scheduler, 'interval': 'step'}}
        ]

    def rand_like(self, x):
        return self.hparams.eps + (1-2*self.hparams.eps)*torch.rand_like(x)
    
    def get_mutinf(self, logits):
        logits = logits.view(-1, logits.size(-1))
        rv = (
            torch.full((logits.size(0),), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
            + (nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        )
        return rv
        
    def get_identity_loss(self, mask):
        return (1 - mask).mean(dim=0).sum()
    
    def sample_mask(self, batch_size):
        generator_latent = torch.randn(batch_size, 1, self.hparams.mask_intrinsic_dim, dtype=torch.float, device=self.device)
        u = self.rand_like(torch.empty((batch_size, 1, self.hparams.timesteps_per_trace), dtype=torch.float, device=self.device))
        generator_logits = self.noise_generator(generator_latent)
        z = nn.functional.logsigmoid(generator_logits) - nn.functional.logsigmoid(-generator_logits) + u.log() - (1-u).log()
        if self.hparams.gradient_estimator == 'CONCRETE':
            mask = nn.functional.sigmoid(z/self.hparams.tau)
        elif self.hparams.gradient_estimator == 'REINFORCE':
            mask = torch.where(z >= 0, torch.ones_like(z), torch.zeros_like(z))
        else:
            raise NotImplementedError
        return mask
    
    def step_gammap(self, trace, train: bool = False):
        if train:
            _, psi_optimizer = self.optimizers()
            _, psi_lr_scheduler = self.lr_schedulers()
            psi_optimizer.zero_grad()
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'CONCRETE':
            mask = self.sample_mask(batch_size)
            noise = torch.randn_like(trace)
            classifier_logits = self.classifiers(mask*trace + (1-mask)*noise, 1-mask)
            classifier_logits = classifier_logits.view(-1, classifier_logits.size(-1))
            mutinf_loss = self.get_mutinf(classifier_logits)
            mean_mutinf_loss = mutinf_loss.mean()
            identity_loss = self.get_identity_loss(mask)
            loss = self.hparams.identity_lambda*identity_loss + mean_mutinf_loss
        elif self.hparams.gradient_estimator == 'REINFORCE':
            raise NotImplementedError
        else:
            raise NotImplementedError
        rv.update({
            'loss': loss, 'mutinf_loss': mean_mutinf_loss, 'identity_loss': identity_loss
        })
        if train:
            self.manual_backward(loss)
            psi_optimizer.step()
            psi_lr_scheduler.step()
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
        logits = self.classifiers(mask*trace + (1-mask)*noise, 1-mask)
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
        if self.hparams.calibrate_classifiers:
            raise NotImplementedError
        with torch.no_grad():
            masks = self.sample_mask(100000)
            gamma = (1 - masks).mean(dim=0).detach().cpu().numpy().squeeze()
        save_dir = os.path.join(self.logger.log_dir, 'gamma_log')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'gamma__step={self.global_step}.npy'), gamma)