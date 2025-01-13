import numpy as np
import torch
from torch import nn, optim
import lightning as L
from scipy.stats import kendalltau, pearsonr

from common import *
from .utils import *
from ..utils import *
import utils.lr_schedulers
from utils.metrics import get_rank
from utils.gmm_performance_correlation import GMMPerformanceCorrelation

##### TODO: implement online estimate of classifier output temperature scaling
class Module(L.LightningModule):
    def __init__(self,
        classifiers_name: str,
        classifiers_kwargs: dict = {},
        theta_lr_scheduler_name: str = None,
        theta_lr_scheduler_kwargs: dict = {},
        etat_lr_scheduler_name: str = None,
        etat_lr_scheduler_kwargs: dict = {},
        theta_lr: float = 1e-3,
        etat_lr: float = 1e-3,
        etat_beta_1: float = 0.99,
        etat_beta_2: float = 0.99999,
        etat_eps: float = 1e-4,
        theta_weight_decay: float = 0.0,
        etat_weight_decay: float = 0.0,
        budget: float = 50.0,
        timesteps_per_trace: Optional[int] = None,
        class_count: int = 256,
        gradient_estimator: Literal['REINFORCE', 'REBAR'] = 'REBAR',
        rebar_relaxation: Literal['CONCRETE', 'MuProp'] = 'MuProp',
        noise_scale: Optional[float] = None,
        eps: float = 1e-6,
        train_theta: bool = True,
        train_etat: bool = True,
        calibrate_classifiers: bool = False,
        compute_gmm_ktcc: bool = False,
        reference_leakage_assessment: Optional[np.ndarray] = None
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
            output_classes=self.hparams.class_count,
            classifiers_kwargs=self.hparams.classifiers_kwargs,
            calibrate_classifiers=self.hparams.calibrate_classifiers
        )
        self.selection_mechanism = SelectionMechanism(
            self.hparams.timesteps_per_trace,
            C=self.hparams.budget
        )
        if self.hparams.gradient_estimator == 'REBAR':
            self.etat = nn.Parameter(torch.tensor(0., dtype=torch.float32), requires_grad=True)
            self.taut = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32), requires_grad=True)
        if not isinstance(self.hparams.reference_leakage_assessment, dict):
            if isinstance(self.hparams.reference_leakage_assessment, np.ndarray):
                self.hparams.reference_leakage_assessment = {'ref_0': self.hparams.reference_leakage_assessment}
            elif isinstance(self.hparams.reference_leakage_assessment, list):
                self.hparams.reference_leakage_assessment = {f'ref_{idx}': x for idx, x in enumerate(self.hparams.reference_leakage_assessment)}
            elif isinstance(self.hparams.reference_leakage_assessment, type(None)):
                pass
            else:
                assert False
        
    def rand_like(self, x):
        return self.hparams.eps + (1 - 2*self.hparams.eps)*torch.rand_like(x)
        
    def get_eta_and_tau(self): # FIXME: modify this so that we can optimize the eta and tau parameters
        assert self.hparams.gradient_estimator == 'REBAR'
        eta = self.hparams.eps + 2*nn.functional.sigmoid(self.etat)
        tau = self.hparams.eps + nn.functional.softplus(self.taut)
        return eta, tau
    
    def configure_optimizers(self):
        self.etat_optimizer = optim.Adam(
            self.selection_mechanism.parameters(), lr=self.hparams.etat_lr, weight_decay=self.hparams.etat_weight_decay,
            betas=(self.hparams.etat_beta_1, self.hparams.etat_beta_2), eps=self.hparams.etat_eps
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
        if self.trainer.max_epochs != -1:
            self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            assert False
        self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=self.total_steps, **self.hparams.theta_lr_scheduler_kwargs)
        self.etat_lr_scheduler = etat_lr_scheduler_constructor(self.etat_optimizer, total_steps=self.total_steps, **self.hparams.etat_lr_scheduler_kwargs)
        rv = [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etat_optimizer, 'lr_scheduler': {'scheduler': self.etat_lr_scheduler, 'interval': 'step'}}
        ]
        if self.hparams.gradient_estimator == 'REBAR':
            self.rebar_params_optimizer = optim.Adam([self.etat, self.taut], lr=10*self.hparams.etat_lr, betas=(0.9, 0.99999))
            self.rebar_params_lr_scheduler = etat_lr_scheduler_constructor(self.rebar_params_optimizer, total_steps=self.total_steps, **self.hparams.etat_lr_scheduler_kwargs)
            rv.append({'optimizer': self.rebar_params_optimizer, 'lr_scheduler': {'scheduler': self.rebar_params_lr_scheduler, 'interval': 'step'}})
        return rv

    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def get_b_values(self, trace: torch.Tensor, temperature: torch.Tensor):
        assert self.hparams.gradient_estimator == 'REBAR'
        if self.hparams.train_etat:
            log_gamma = self.selection_mechanism.get_log_gamma().unsqueeze(0)
            log_1mgamma = self.selection_mechanism.get_log_1mgamma().unsqueeze(0)
        else: # In the classifier pretraining stage it's convenient to clamp these to a specific value to make pretraining independent of our budget
            log_gamma = np.log(0.5)*torch.ones((1, *trace.shape[1:]), dtype=trace.dtype, device=trace.device)
            log_1mgamma = np.log(0.5)*torch.ones((1, *trace.shape[1:]), dtype=trace.dtype, device=trace.device)
        log_alpha = log_gamma - log_1mgamma
        assert torch.all(torch.isfinite(log_gamma))
        assert torch.all(torch.isfinite(log_1mgamma))
        u = self.rand_like(trace)
        b = torch.where(log_alpha + u.log() - (1-u).log() >= 0, torch.ones_like(u), torch.zeros_like(u))
        uprime = 1 - log_gamma.exp()
        v = self.rand_like(u)
        v = torch.where(b == 1, uprime + v*(1-uprime), v*uprime).clip_(self.hparams.eps, 1-self.hparams.eps)
        if self.hparams.rebar_relaxation == 'CONCRETE':
            to_z = lambda log_alpha, u: (log_alpha + u.log() - (1-u).log())
            rb = nn.functional.sigmoid(to_z(log_alpha, u)/temperature)
            rb_tilde = nn.functional.sigmoid(to_z(log_alpha, v)/temperature)
            rb_tilde_detached = nn.functional.sigmoid(to_z(log_alpha.detach(), v.detach())/temperature)
        elif self.hparams.rebar_relaxation == 'MuProp':
            to_z = lambda log_alpha, u: ((temperature**2 + temperature + 1)/(temperature + 1))*log_alpha + u.log() - (1-u).log()
            rb = nn.functional.sigmoid(to_z(log_alpha, u)/temperature)
            rb_tilde = nn.functional.sigmoid(to_z(log_alpha, v)/temperature)
            rb_tilde_detached = nn.functional.sigmoid(to_z(log_alpha.detach(), v.detach())/temperature) # We only want to 'detach' with respect to log_gamma, not the temperature.
        else:
            raise NotImplementedError
        return b, rb, rb_tilde, rb_tilde_detached
    
    def step_etat(self, batch, train: bool = False):
        if train:
            self.cmi_estimator.classifiers.requires_grad_(False)
            if self.hparams.gradient_estimator == 'REBAR':
                _, etat_optimizer, rebar_params_optimizer = self.optimizers()
                rebar_params_optimizer.zero_grad()
            else:
                _, etat_optimizer = self.optimizers()
            _, etat_lr_scheduler, rebar_params_lr_scheduler = self.lr_schedulers()
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
                self.mutinf_ema = 0.99*self.mutinf_ema + 0.01*mutinf.detach().mean()
            loss = -((mutinf - self.mutinf_ema)*log_prob_mass).mean()
        elif self.hparams.gradient_estimator == 'REBAR':
            eta, tau = self.get_eta_and_tau()
            b, rb, rb_tilde, rb_tilde_detached = self.get_b_values(trace, tau)
            mutinf = self.cmi_estimator.get_mutinf_estimate(trace.repeat(4, 1, 1), torch.cat([b, rb, rb_tilde, rb_tilde_detached], dim=0), labels=label.repeat(4))
            mutinf_b = mutinf[:len(b)].detach()
            mutinf_rb = mutinf[len(b):len(b)+len(rb)]
            mutinf_rb_tilde = mutinf[len(b)+len(rb):len(b)+len(rb)+len(rb_tilde)]
            mutinf_rb_tilde_detached = mutinf[-len(rb_tilde_detached):]
            log_p_b = self.selection_mechanism.log_pmf(b)
            loss = -((mutinf_b - eta*mutinf_rb_tilde_detached)*log_p_b + eta*mutinf_rb - eta*mutinf_rb_tilde).mean()
        else:
            assert False
        rv.update({'loss': loss.detach().mean()})
        if train:
            if self.hparams.gradient_estimator == 'REBAR':
                (etat_grad,) = torch.autograd.grad(loss, self.selection_mechanism.etat, create_graph=True)
                self.selection_mechanism.etat.grad = etat_grad
                rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
                if not hasattr(self, 'etat_grad_ema'):
                    self.etat_grad_ema = etat_grad.detach()
                else:
                    self.etat_grad_ema = 0.999*self.etat_grad_ema + 0.001*etat_grad.detach()
                rebar_params_loss = ((etat_grad - self.etat_grad_ema)**2).mean()
                (self.etat.grad, self.taut.grad) = torch.autograd.grad(rebar_params_loss, [self.etat, self.taut])
                etat_optimizer.step()
                etat_lr_scheduler.step()
                rebar_params_optimizer.step()
                rebar_params_lr_scheduler.step()
                self.selection_mechanism.update_accumulated_gamma()
            else:
                self.manual_backward(loss)
                rv.update({'rms_grad': get_rms_grad(self.selection_mechanism)})
                etat_optimizer.step()
                etat_lr_scheduler.step()
            self.cmi_estimator.classifiers.requires_grad_(True)
        assert all(torch.all(torch.isfinite(param)) for param in self.selection_mechanism.parameters())
        return rv
    
    def step_theta(self, batch, train: bool = False):
        if train:
            theta_optimizer, *_ = self.optimizers()
            theta_lr_scheduler, *_ = self.lr_schedulers()
            theta_optimizer.zero_grad()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        if self.hparams.gradient_estimator == 'REBAR': # we need to train it on the interpolated masks, not just hard masks
            _, tau = self.get_eta_and_tau()
            with torch.no_grad():
                b, rb, _, _ = self.get_b_values(trace.repeat(2, 1, 1), tau)
            alpha = torch.cat([b, rb], dim=0)
        else:
            alpha = self.selection_mechanism.sample(batch_size)
        if train and self.hparams.noise_scale is not None:
            trace = trace + self.hparams.noise_scale*torch.randn_like(trace)
        logits = self.cmi_estimator.get_logits(trace.repeat(4, 1, 1), alpha)
        loss = nn.functional.cross_entropy(logits, label.repeat(4))
        rv.update({'loss': loss.detach(), 'rank': get_rank(logits, label.repeat(4)).mean()})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.cmi_estimator)})
            theta_optimizer.step()
            theta_lr_scheduler.step()
        assert all(torch.all(torch.isfinite(param)) for param in self.cmi_estimator.parameters())
        return rv
    
    def training_step(self, batch):
        if self.hparams.train_theta:
            theta_rv = self.step_theta(batch, train=True)
            for key, val in theta_rv.items():
                self.log(f'train_theta_{key}', val, on_step=False, on_epoch=True)
        if self.hparams.train_etat:
            etat_rv = self.step_etat(batch, train=True)
            for key, val in etat_rv.items():
                self.log(f'train_etat_{key}', val, on_step=False, on_epoch=True)
    
    def validation_step(self, batch):
        if self.hparams.train_theta:
            theta_rv = self.step_theta(batch, train=False)
            for key, val in theta_rv.items():
                self.log(f'val_theta_{key}', val, on_step=False, on_epoch=True)
        if self.hparams.train_etat:
            etat_rv = self.step_etat(batch, train=False)
            for key, val in etat_rv.items():
                self.log(f'val_etat_{key}', val, on_step=False, on_epoch=True)
    
    def calibrate_classifiers(self):
        def get_classifier_input(trace):
            batch_size = trace.size(0)
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
            obfuscated_trace = alpha*trace + (1-alpha)*torch.randn_like(trace)
            return (obfuscated_trace, alpha)
        self.cmi_estimator.classifiers.calibrate_temperature(
            self.trainer.datamodule.val_dataloader(), get_classifier_input
        )
    
    def on_train_epoch_end(self):
        log_gamma = self.selection_mechanism.get_log_gamma().detach().cpu().numpy().squeeze()
        log_gamma_save_dir = os.path.join(self.logger.log_dir, 'log_gamma_over_time')
        os.makedirs(log_gamma_save_dir, exist_ok=True)
        np.save(os.path.join(log_gamma_save_dir, f'log_gamma__step={self.global_step}.npy'), log_gamma)
        if self.hparams.train_etat and self.hparams.calibrate_classifiers:
            self.calibrate_classifiers()
        if self.hparams.reference_leakage_assessment is not None:
            gamma = self.selection_mechanism.get_accumulated_gamma().reshape(-1) #get_gamma().detach().cpu().numpy().reshape(-1)
            for key, leakage_assessment in self.hparams.reference_leakage_assessment.items():
                ktcc = kendalltau(gamma, leakage_assessment.reshape(-1)).statistic
                correlation = pearsonr(gamma, leakage_assessment.reshape(-1)).statistic
                self.log(f'{key}_ktcc', ktcc)
                self.log(f'{key}_corr', correlation)
        if False: #self.current_epoch % (self.total_steps//(100*len(self.trainer.datamodule.train_dataloader()))) == 0:
            gamma = self.selection_mechanism.get_accumulated_gamma().reshape(-1)
            profiling_dataset = self.trainer.datamodule.profiling_dataset
            attack_dataset = self.trainer.datamodule.attack_dataset
            metric = GMMPerformanceCorrelation(gamma.argsort(), device='cuda')
            metric.profile(profiling_dataset)
            rv = metric(attack_dataset)
            self.log('gmmperfcorr', rv)