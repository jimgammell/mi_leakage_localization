import typing
from copy import copy
import numbers
import torch
from torch import nn, optim
import lightning as L

from common import *
import utils.lr_schedulers
from utils.metrics import get_rank
from utils.recalibrate_batchnorm import recalibrate_batchnorm
import models
from models.calibrated_model import CalibratedModel

def rms(val):
    return (val**2).mean().sqrt()

class AdversarialLeakageLocalizationModule(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_optimizer_name: Union[str, optim.Optimizer] = optim.AdamW,
        theta_optimizer_kwargs: dict = {},
        theta_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        theta_lr_scheduler_kwargs: dict = {},
        gammap_optimizer_name: Union[str, optim.Optimizer] = optim.SGD,
        gammap_optimizer_kwargs: dict = {},
        gammap_lr_scheduler_name: Optional[Union[str, optim.lr_scheduler.LRScheduler]] = None,
        gammap_lr_scheduler_kwargs: dict = {},
        gammap_identity_coeff: float = 1.0,
        theta_pretrain_steps: int = 0,
        alternating_train_steps: int = -1, # -1 === infinitely-many steps
        theta_adversarial_data_prop: float = 0.5,
        gammap_complement_proposal_dist: bool = False,
        theta_pretrain_dist: Literal['Uniform', 'Dirichlet'] = 'Dirichlet',
        gammap_squashing_fn: Literal['Sigmoid', 'HardSigmoid'] = 'Sigmoid',
        gammap_identity_penalty_fn: Literal['l1', 'l2', 'Entropy'] = 'l2',
        gammap_rl_strategy: Literal['LogLikelihood', 'ENCO'] = 'LogLikelihood',
        calibrate_classifiers: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.classifiers_name = self.hparams.classifiers_name
        self.classifiers_kwargs = self.hparams.classifiers_kwargs
        self.theta_optimizer_name = self.hparams.theta_optimizer_name
        self.theta_optimizer_kwargs = self.hparams.theta_optimizer_kwargs
        self.theta_lr_scheduler_name = self.hparams.theta_lr_scheduler_name
        self.theta_lr_scheduler_kwargs = self.hparams.theta_lr_scheduler_kwargs
        self.gammap_optimizer_name = self.hparams.gammap_optimizer_name
        self.gammap_optimizer_kwargs = self.hparams.gammap_optimizer_kwargs
        self.gammap_lr_scheduler_name = self.hparams.gammap_lr_scheduler_name
        self.gammap_lr_scheduler_kwargs = self.hparams.gammap_lr_scheduler_kwargs
        self.gammap_identity_coeff = self.hparams.gammap_identity_coeff
        self.theta_pretrain_steps = self.hparams.theta_pretrain_steps
        self.alternating_train_steps = self.hparams.alternating_train_steps
        self.theta_adversarial_data_prop = self.hparams.theta_adversarial_data_prop
        self.gammap_complement_proposal_dist = self.hparams.gammap_complement_proposal_dist
        self.theta_pretrain_dist = self.hparams.theta_pretrain_dist
        self.gammap_squashing_fn = self.hparams.gammap_squashing_fn
        self.gammap_identity_penalty_fn = self.hparams.gammap_identity_penalty_fn
        self.gammap_rl_strategy = self.hparams.gammap_rl_strategy
        self.calibrate_classifiers = self.hparams.calibrate_classifiers
        
        self.validate_hyperparameters()
        self.automatic_optimization = False
        self.classifiers = models.load(self.classifiers_name, noise_conditional=True, **self.classifiers_kwargs)
        if self.calibrate_classifiers:
            self.classifiers = CalibratedModel(self.classifiers)
        self.gammap = nn.Parameter(torch.zeros(*self.classifiers.input_shape, dtype=torch.float32), requires_grad=True)
    
    def configure_theta_optimizer(self):
        optim_constructor = self.theta_optimizer_name if isinstance(self.theta_optimizer_name, optim.Optimizer) else getattr(optim, self.theta_optimizer_name)
        lr_scheduler_constructor = (
            self.theta_lr_scheduler_name if isinstance(self.theta_lr_scheduler_name, (optim.lr_scheduler.LRScheduler, type(None)))
            else getattr(utils.lr_schedulers, self.theta_lr_scheduler_name) if hasattr(utils.lr_schedulers, self.theta_lr_scheduler_name)
            else getattr(optim.lr_scheduler, self.theta_lr_scheduler_name)
        )
        weight_decay = self.theta_optimizer_kwargs['weight_decay'] if 'weight_decay' in self.theta_optimizer_kwargs.keys() else 0.0
        if 'weight_decay' in self.theta_optimizer_kwargs.keys():
            weight_decay = self.theta_optimizer_kwargs['weight_decay']
            del self.theta_optimizer_kwargs['weight_decay']
        else:
            weight_decay = 0.0
        yes_weight_decay, no_weight_decay = [], []
        for name, param in self.classifiers.named_parameters():
            if ('weight' in name) and not('norm' in name):
                yes_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [{'params': yes_weight_decay, 'weight_decay': weight_decay}, {'params': no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim_constructor(param_groups, **self.theta_optimizer_kwargs)
        if lr_scheduler_constructor is not None:
            total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
            if self.alternating_train_steps == -1:
                alternating_train_steps = total_steps - self.theta_pretrain_steps
            else:
                alternating_train_steps = self.alternating_train_steps
            step_count = self.theta_pretrain_steps + alternating_train_steps
            self.theta_lr_scheduler = lr_scheduler_constructor(self.theta_optimizer, total_steps=step_count, **self.theta_lr_scheduler_kwargs)
        else:
            self.theta_lr_scheduler = optim.lr_scheduler.LambdaLR(self.theta_optimizer, lr_lambda=lambda _: 1.0)
        return {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}}
    
    def configure_gammap_optimizer(self):
        optim_constructor = self.gammap_optimizer_name if isinstance(self.gammap_optimizer_name, optim.Optimizer) else getattr(optim, self.gammap_optimizer_name)
        lr_scheduler_constructor = (
            self.gammap_lr_scheduler_name if isinstance(self.gammap_lr_scheduler_name, (optim.lr_scheduler.LRScheduler, type(None)))
            else getattr(utils.lr_schedulers, self.gammap_lr_scheduler_name) if hasattr(utils.lr_schedulers, self.gammap_lr_scheduler_name)
            else getattr(optim.lr_scheduler, self.gammap_lr_scheduler_name)
        )
        if not(optim_constructor in [optim.LBFGS]):
            self.gammap_optimizer_kwargs['weight_decay'] = 0.0
        self.gammap_optimizer = optim_constructor([self.gammap], **self.gammap_optimizer_kwargs)
        if lr_scheduler_constructor is not None:
            total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
            step_count = total_steps - self.theta_pretrain_steps
            self.gammap_lr_scheduler = lr_scheduler_constructor(self.gammap_optimizer, total_steps=step_count, **self.gammap_lr_scheduler_kwargs)
        else:
            self.gammap_lr_scheduler = optim.lr_scheduler.LambdaLR(self.gammap_optimizer, lr_lambda=lambda _: 1.0)
        return {'optimizer': self.gammap_optimizer, 'lr_scheduler': {'scheduler': self.gammap_lr_scheduler, 'interval': 'step'}}
    
    def configure_optimizers(self):
        return [self.configure_theta_optimizer(), self.configure_gammap_optimizer()]
    
    def validate_hyperparameters(self):
        assert self.classifiers_name in models.AVAILABLE_MODELS
        assert isinstance(self.theta_optimizer_name, optim.Optimizer) or hasattr(optim, self.theta_optimizer_name)
        if self.theta_lr_scheduler_name is not None:
            assert isinstance(self.theta_lr_scheduler_name, optim.lr_scheduler.LRScheduler) or hasattr(optim.lr_scheduler, self.theta_lr_scheduler_name) or hasattr(utils.lr_schedulers, self.theta_lr_scheduler_name)
        assert isinstance(self.gammap_optimizer_name, optim.Optimizer) or hasattr(optim, self.gammap_optimizer_name)
        if self.gammap_lr_scheduler_name is not None:
            assert isinstance(self.gammap_lr_scheduler_name, optim.lr_scheduler.LRScheduler) or hasattr(optim.lr_scheduler, self.gammap_lr_scheduler_name) or hasattr(utils.lr_schedulers, self.gammap_lr_scheduler_name)
        assert isinstance(self.gammap_identity_coeff, numbers.Real) and (0 <= self.gammap_identity_coeff < float('inf'))
        assert isinstance(self.theta_pretrain_steps, int) and (self.theta_pretrain_steps >= 0)
        assert isinstance(self.theta_adversarial_data_prop, float) and (0 <= self.theta_adversarial_data_prop <= 1)
        assert self.theta_pretrain_dist in ['Uniform', 'Dirichlet']
        assert self.gammap_squashing_fn in ['Sigmoid', 'HardSigmoid']
        assert self.gammap_identity_penalty_fn in ['l1', 'l2', 'Entropy']
        assert self.gammap_rl_strategy in ['LogLikelihood', 'ENCO']
        assert isinstance(self.calibrate_classifiers, bool)
    
    @torch.no_grad()
    def get_gamma(self):
        if self.gammap_squashing_fn == 'Sigmoid':
            return torch.sigmoid(self.gammap)
        elif self.gammap_squashing_fn == 'HardSigmoid':
            return torch.clamp(self.gammap+0.5, 0, 1)
        else:
            assert False
    
    @torch.no_grad()
    def get_dgamma_dgammap(self, gamma):
        if self.gammap_squashing_fn == 'Sigmoid':
            return gamma*(1-gamma)
        elif self.gammap_squashing_fn == 'HardSigmoid':
            return torch.ones_like(gamma)
        else:
            assert False
    
    @torch.no_grad()
    def get_identity_penalty_fn(self, gamma):
        if self.gammap_identity_penalty_fn == 'l1':
            return gamma.sum()
        elif self.gammap_identity_penalty_fn == 'l2':
            return 0.5*(gamma**2).sum()
        elif self.gammap_identity_penalty_fn == 'Entropy':
            return (gamma*gamma.log() + (1-gamma)*(1-gamma).log()).sum()
        else:
            assert False
    
    @torch.no_grad()
    def get_identity_penalty_grad(self, gamma):
        dgamma_dgammap = self.get_dgamma_dgammap(gamma)
        if self.gammap_identity_penalty_fn == 'l1':
            dpenalty_dgamma = torch.ones_like(gamma)
        elif self.gammap_identity_penalty_fn == 'l2':
            dpenalty_dgamma = gamma
        elif self.gammap_identity_penalty_fn == 'Entropy':
            dpenalty_dgamma = gamma.log() - (1-gamma).log()
        else:
            assert False
        return dpenalty_dgamma*dgamma_dgammap
    
    @torch.no_grad()
    def sample_noise(self, trace, gamma, training_theta=False, training_gammap=False):
        batch_size, *trace_dims = trace.shape
        if training_theta:
            pretrain_batch_size = int(batch_size*self.theta_adversarial_data_prop)
            train_batch_size = batch_size - pretrain_batch_size
            if self.theta_pretrain_dist == 'Uniform':
                pretrain_probs = 0.5*torch.ones_like(trace[:pretrain_batch_size, ...])
            elif self.theta_pretrain_dist == 'Dirichlet':
                p = torch.rand(pretrain_batch_size, 1, 1, device=trace.device, dtype=trace.dtype)
                pretrain_probs = p*torch.ones((1, 1, trace.shape[-1]), device=trace.device, dtype=trace.dtype)
            else:
                assert False
            probs = torch.cat([
                gamma.repeat(train_batch_size, *(len(trace.shape[1:])*[1])),
                pretrain_probs
            ], dim=0)
        elif training_gammap and self.gammap_complement_proposal_dist:
            main_batch_size = np.random.binomial(batch_size, 0.5)
            complement_batch_size = batch_size - main_batch_size
            probs = torch.cat([
                gamma.repeat(main_batch_size, *(len(trace.shape[1:])*[1])),
                (1-gamma).repeat(complement_batch_size, *len(trace.shape[1:])*[1])
            ])
        else:
            probs = gamma.repeat(batch_size, *(len(trace.shape[1:])*[1]))
        noise = (1-probs).bernoulli()
        return noise
    
    @torch.no_grad()
    def get_importance_reweighting(self, gamma, alpha):
        if self.gammap_complement_proposal_dist:
            gamma = torch.clamp(gamma, 1e-6, 1-1e-6)
            return 2. / (1. + ((2*alpha-1)*(gamma.log() - (1-gamma).log())).sum().exp())
        else:
            return torch.ones(alpha.size(0), dtype=gamma.dtype, device=gamma.device)
    
    def get_logits(self, trace, alpha, calibrate=False):
        if calibrate:
            assert self.calibrate_classifiers
            logits = self.classifiers.calibrated_forward(trace*alpha, alpha)
        else:
            logits = self.classifiers(trace*alpha, alpha)
        logits = logits.view(-1, logits.size(-1))
        return logits
    
    @torch.no_grad()
    def get_mutual_information(self, logits):
        mutinfs = (
            torch.full((logits.size(0),), np.log(self.classifiers.output_classes), dtype=logits.dtype, device=logits.device)
            - (nn.functional.softmax(logits, dim=-1)*nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
        )
        return mutinfs
    
    @torch.no_grad()
    def get_log_likelihood(self, mutinfs, importance_reweighting):
        return (mutinfs*importance_reweighting).mean()
    
    @torch.no_grad()
    def get_log_likelihood_grad(self, mutinfs, alpha, importance_reweighting):
        alpha = alpha.squeeze()
        mutinfs = mutinfs*importance_reweighting
        if self.gammap_rl_strategy == 'ENCO':
            alpha = alpha.bool()
            pos_count, neg_count = map(lambda x: x.sum(dim=0), (alpha, ~alpha))
            pos_sum, neg_sum = map(lambda x: torch.where(x, mutinfs.unsqueeze(1), torch.tensor(0., device=mutinfs.device, dtype=mutinfs.dtype)).sum(dim=0), (alpha, ~alpha))
            gradient = neg_sum/(neg_count.clamp(min=1)) - pos_sum/(pos_count.clamp(min=1))
            gradient = gradient.clamp(max=0.) # Ideally, this should always be nonpositive. In practice it might be positive if the classifiers are shitty.
            gradient[torch.logical_or(pos_count==0, neg_count==0)] = 0.
        elif self.gammap_rl_strategy == 'LogLikelihood':
            raise NotImplementedError
        return gradient
    
    def step_theta(self, trace, label, train=True):
        rv = {}
        with torch.set_grad_enabled(train):
            gamma = self.get_gamma()
            noise = self.sample_noise(trace, gamma, training_theta=True)
            logits = self.get_logits(trace, noise, calibrate=not(train) and self.calibrate_classifiers)
            loss = nn.functional.cross_entropy(logits, label)
        if train:
            theta_optimizer, _ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
            theta_optimizer.zero_grad()
            self.manual_backward(loss)
            grad = max(param.grad.abs().max() for param in self.classifiers.parameters() if param.grad is not None)
            rv.update({'grad': grad})
            theta_optimizer.step()
            if theta_lr_scheduler is not None:
                theta_lr_scheduler.step()
        rv.update({'loss': loss})
        rv.update({'rank': get_rank(logits, label).mean()})
        return rv
    
    @torch.no_grad()
    def step_gammap(self, trace, label, train=True):
        rv = None
        def closure():
            nonlocal rv
            if rv is None:
                rv = {}
            gamma = self.get_gamma()
            alpha = self.sample_noise(trace, gamma, training_gammap=True)
            logits = self.get_logits(trace, alpha, calibrate=self.calibrate_classifiers)
            mutinfs = self.get_mutual_information(logits)
            importance_reweighting = self.get_importance_reweighting(gamma, alpha)
            mutinf_loss = self.get_log_likelihood(mutinfs, importance_reweighting)
            identity_penalty = self.get_identity_penalty_fn(gamma)
            total_loss = mutinf_loss + self.gammap_identity_coeff*identity_penalty
            rv.update({
                'mutinf_loss': mutinf_loss,
                'identity_loss': identity_penalty,
                'total_loss': total_loss,
                'rank': get_rank(logits, label).mean()
            })
            if train:
                mutinf_grad = self.get_log_likelihood_grad(mutinfs, alpha, importance_reweighting)
                identity_grad = self.get_identity_penalty_grad(gamma)
                total_grad = mutinf_grad + self.gammap_identity_coeff*identity_grad
                rv.update({
                    'mutinf_rms_grad': rms(mutinf_grad),
                    'identity_rms_grad': rms(self.gammap_identity_coeff*identity_grad)
                })
                self.gammap.grad = total_grad
            return total_loss
        
        if train:
            _, gammap_optimizer = self.optimizers()
            gammap_optimizer.step(closure)
            _, gammap_lr_scheduler = self.lr_schedulers()
            if gammap_lr_scheduler is not None:
                gammap_lr_scheduler.step()
        if rv is None:
            closure()
        return rv
    
    def training_step(self, batch):
        if self.global_step < self.theta_pretrain_steps:
            train_theta = True
            train_gammap = False
        elif (self.alternating_train_steps == -1) or (self.global_step <= self.theta_pretrain_steps + self.alternating_train_steps):
            train_theta = train_gammap = True
        else:
            train_theta = False
            train_gammap = True
        [(theta_trace, theta_label), (gammap_trace, gammap_label)] = batch
        if train_theta:
            rv = self.step_theta(theta_trace, theta_label, train=True)
            for key, val in rv.items():
                self.log(f'train_theta_{key}', val, on_step=True, on_epoch=False)
        if train_gammap:
            rv = self.step_gammap(gammap_trace, gammap_label, train=True)
            for key, val in rv.items():
                self.log(f'train_gammap_{key}', val, on_step=True, on_epoch=False)
    
    @torch.no_grad()
    def validation_step(self, batch):
        trace, label = batch
        rv = self.step_theta(trace, label, train=False)
        for key, val in rv.items():
            self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        rv = self.step_gammap(trace, label, train=False)
        for key, val in rv.items():
            self.log(f'val_gammap_{key}', val, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        gamma = self.get_gamma()
        #recalibrate_batchnorm(self.trainer, self, mode='ALL')
        if self.calibrate_classifiers and ((self.alternating_train_steps == -1) or (self.global_step <= self.theta_pretrain_steps + self.alternating_train_steps)):
            self.classifiers.calibrate_temperature(
                self.trainer.datamodule.val_dataloader(),
                lambda x: self.sample_noise(x, gamma, training_theta=True),
                1
            )
        if self.global_step >= self.theta_pretrain_steps:
            save_dir = os.path.join(self.logger.log_dir, 'gamma_log')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'gamma__step={self.global_step}.npy'), gamma.detach().cpu().numpy().squeeze())