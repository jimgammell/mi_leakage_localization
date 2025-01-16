from typing import *
import os
from numbers import Number
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from common import *
from trials.utils import *
from utils.baseline_assessments import FirstOrderStatistics
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer

def _plot_leakage_assessments(dest, leakage_assessments, leaking_instruction_timesteps=None, title=None, to_label=None):
    keys = list(leakage_assessments.keys())
    if isinstance(keys[0], Number):
        assert all(isinstance(key, Number) for key in keys)
        keys.sort()
        leakage_assessments = {key: leakage_assessments[key] for key in keys}
    row_count = max(len(x) for x in leakage_assessments.values())
    fig, axes = plt.subplots(row_count, len(leakage_assessments), figsize=(PLOT_WIDTH*len(leakage_assessments), PLOT_WIDTH*row_count))
    if isinstance(leaking_instruction_timesteps, int):
        for ax in axes.flatten():
            ax.axvline(leaking_instruction_timesteps, linestyle=':', color='black', label='leaking instruction')
    elif hasattr(leaking_instruction_timesteps, '__iter__'):
        for x, axes_col in zip(leaking_instruction_timesteps, axes.transpose()):
            for ax in axes_col:
                if (x is None) or (x.dtype == type(None)):
                    continue
                else:
                    for xx in x:
                        ax.axvline(xx, color='black', linestyle='--', linewidth=0.5)
    for col_idx, (setting, _leakage_assessments) in enumerate(leakage_assessments.items()):
        for row_idx, (budget, leakage_assessment) in enumerate(_leakage_assessments.items()):
            ax = axes[row_idx, col_idx]
            ax.plot(leakage_assessment, color='blue', marker='.', markersize=3, linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Est. lkg. of $X_t$' + f' (budget={budget})')
            ax.set_ylim(0, 1)
            if to_label is not None:
                ax.set_title(to_label(setting))
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        batch_size: int = 1000,
        timestep_count: int = 101,
        trial_count: int = 8,
        budgets: Union[float, Sequence[float]] = np.logspace(-1, 3, 5),
        pretrain_classifiers_only: bool = False
    ):
        self.logging_dir = logging_dir
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'theta_lr': 1e-3, 'etat_lr': 1e-3, 'calibrate_classifiers': False, 'ent_penalty': 1e-2}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.batch_size = batch_size
        self.timestep_count = timestep_count
        self.trial_count = trial_count
        self.budgets = budgets
        if not hasattr(self.budgets, '__len__'):
            self.budgets = [self.budgets]
        self.pretrain_classifiers_only = pretrain_classifiers_only
    
    def construct_datasets(self,
        leaky_1o_count: int = 1,
        leaky_2o_count: int = 0,
        data_var: float = 1.0,
        shuffle_locs: int = 1,
        max_no_ops: int = 0,
        lpf_beta: float = 0.9   
    ):
        leaky_count = shuffle_locs*(leaky_1o_count + 2*leaky_2o_count)
        if leaky_count > 0:
            leaky_pts = np.linspace(0, self.timestep_count-1, leaky_count+2)[1:-1].astype(int)
            leaky_1o_pts = leaky_pts[:shuffle_locs*leaky_1o_count] if leaky_1o_count > 0 else None
            leaky_2o_pts = leaky_pts[shuffle_locs*leaky_1o_count:].reshape(2, -1) if leaky_2o_count > 0 else None
        else:
            leaky_pts = leaky_1o_pts = leaky_2o_pts = None
        profiling_dataset = SyntheticAES(
            infinite_dataset=True,
            timesteps_per_trace=self.timestep_count,
            leaking_timestep_count_1o=0,
            leaking_timestep_count_2o=0,
            leaky_1o_pts=leaky_1o_pts,
            leaky_2o_pts=leaky_2o_pts,
            data_var=data_var,
            shuffle_locs=shuffle_locs,
            max_no_ops=max_no_ops,
            lpf_beta=lpf_beta
        )
        attack_dataset = SyntheticAESLike(profiling_dataset, fixed_key=0)
        return profiling_dataset, attack_dataset, leaky_1o_pts, leaky_2o_pts
    
    def construct_trainer(self, profiling_dataset, attack_dataset, budget):
        trainer = LeakageLocalizationTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs={'budget': budget, **self.leakage_localization_kwargs}
        )
        return trainer
    
    def run_experiment(self, logging_dir, kwargs):
        os.makedirs(logging_dir, exist_ok=True)
        if os.path.exists(os.path.join(logging_dir, 'leakage_assessments.npz')):
            rv = np.load(os.path.join(logging_dir, 'leakage_assessments.npz'), allow_pickle=True)
            leakage_assessments = rv['leakage_assessments']
            locs_1o = rv['locs_1o']
            locs_2o = rv['locs_2o']
        else:
            profiling_dataset, attack_dataset, locs_1o, locs_2o = self.construct_datasets(**kwargs)
            trainer = self.construct_trainer(profiling_dataset, attack_dataset, 1.0) # classifier pretraining is independent of budget
            trainer.pretrain_classifiers(os.path.join(logging_dir, 'classifiers_pretrain'), max_steps=self.run_kwargs['max_steps'])
            leakage_assessments = {}
            for budget in self.budgets:
                trainer = self.construct_trainer(profiling_dataset, attack_dataset, budget)
                leakage_assessment = trainer.run(
                    os.path.join(logging_dir, f'll_budget={budget}'),
                    pretrained_classifiers_logging_dir=os.path.join(logging_dir, 'classifiers_pretrain'),
                    **self.run_kwargs
                )
                leakage_assessments[budget] = leakage_assessment
            np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments, locs_1o=locs_1o, locs_2o=locs_2o)
        return leakage_assessments, locs_1o, locs_2o
    
    def plot_leakage_assessments(self, *args, **kwargs):
        if not self.pretrain_classifiers_only:
            _plot_leakage_assessments(*args, **kwargs)
    
    def run_1o_beta_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        leakage_assessments = {}
        for beta in [1 - 0.25**n for n in range(self.trial_count)]:
            subdir = os.path.join(exp_dir, f'beta={beta}')
            leakage_assessments[1-beta], *_ = self.run_experiment(subdir, {'lpf_beta': beta})
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            self.timestep_count//2,
            title=r'Sweep of low-pass filter coefficient: $\beta_{\mathrm{LPF}}$',
            to_label=lambda x: r'$\beta_{\mathrm{LPF}}='+f'{1-x}'+r'$'
        )
    
    def run_1o_data_var_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_data_var_sweep')
        leakage_assessments = {}
        for var in [1.0] + [0.5**(-2*n) for n in range(1, self.trial_count//2)] + [0.5**(2*n) for n in range(1, self.trial_count//2)] + [0.0]:
            subdir = os.path.join(exp_dir, f'var={var}')
            leakage_assessments[var], *_ = self.run_experiment(subdir, {'data_var': var})
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            self.timestep_count//2,
            title=r'Sweep of data-dependent variance: $\sigma_{\mathrm{data}}$',
            to_label=lambda x: r'$\sigma_{\mathrm{data}}='+f'{x}'+r'$'
        )
    
    def run_1o_leaky_pt_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_leaky_pt_sweep')
        leakage_assessments = {}
        locss = []
        for count in [0] + [1 + 2*x for x in range(self.trial_count-1)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'leaky_1o_count': count})
            locss.append(locs)
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            locss,
            title=r'Sweep of leaky instruction count: $n_{\mathrm{lkg}}$',
            to_label=lambda x: r'$n_{\mathrm{lkg}}='+f'{x}'+r'$'
        )
    
    def run_1o_no_op_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_no_op_sweep')
        leakage_assessments = {}
        locss = []
        for count in [0] + [1 + 2*x for x in range(self.trial_count-1)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'max_no_ops': count})
            locss.append(locs)
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            self.timestep_count//2,
            title=r'Sweep of max no-ops: $n_{\mathrm{no-op}}$',
            to_label=lambda x: r'$n_{\mathrm{no-op}}='+f'{x}'+r'$'
        )
    
    def run_1o_shuffle_loc_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_shuffle_sweep')
        leakage_assessments = {}
        locss = []
        for count in [1 + 2*x for x in range(self.trial_count)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'shuffle_locs': count})
            locss.append(locs)
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            locss,
            title=r'Sweep of shuffle location count: $n_{\mathrm{shuff}}$',
            to_label=lambda x: r'$n_{\mathrm{shuff}}='+f'{x}'+r'$'
        )
    
    def run_2o_trial(self):
        exp_dir = os.path.join(self.logging_dir, '2o_trial')
        leakage_assessment, _, locs = self.run_experiment(exp_dir, {'leaky_1o_count': 0, 'leaky_2o_count': 1})
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            {None: leakage_assessment},
            locs,
            title=r'Second-order leakage'
        )
    
    def __call__(self):
        self.run_1o_beta_sweep()
        self.run_1o_data_var_sweep()
        self.run_1o_leaky_pt_count_sweep()
        self.run_1o_no_op_count_sweep()
        self.run_1o_shuffle_loc_sweep()
        #self.run_2o_trial()