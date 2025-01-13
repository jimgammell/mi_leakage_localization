from typing import *
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from common import *
from trials.utils import *
from utils.baseline_assessments import FirstOrderStatistics
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer

def plot_leakage_assessments(dest, leakage_assessments, leaking_instruction_timesteps=None, title=None, to_label=None):
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
    colors = plt.cm.get_cmap('tab10', len(leakage_assessments))
    if isinstance(leaking_instruction_timesteps, int):
        ax.axvline(leaking_instruction_timesteps, linestyle=':', color='black', label='leaking instruction')
    else:
        for idx, x in enumerate(leaking_instruction_timesteps):
            color = colors(idx)
            if (x is None) or isinstance(x, np.ndarray) and (x.dtype == np.array(None).dtype):
                pass
            else:
                linewidths = np.linspace(2., 0.5, len(leaking_instruction_timesteps))
                linestyles = [':', '--', '-.']
                for xx in x:
                    ax.axvline(xx, linewidth=linewidths[idx], linestyle=linestyles[idx%3], color=color)
        line = Line2D([0], [0], color='black', linestyle=':', label='leaking instruction')
    for idx, (setting, leakage_assessment) in enumerate(leakage_assessments.items()):
        color = colors(idx)
        ax.plot(leakage_assessment, color=color, marker='.', markersize=3, linestyle='none', label='' if to_label is None else to_label(setting))
    ax.set_xlabel(r'Timestep $t$')
    ax.set_ylabel(r'Estimated leakage of $X_t$')
    if title is not None:
        ax.set_title(title)
    if to_label is not None:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        batch_size: int = 1000,
        timestep_count: int = 1001,
        trial_count: int = 8
    ):
        self.logging_dir = logging_dir
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'budget': 100, 'theta_lr': 1e-3, 'etat_lr': 1e-2, 'etat_lr_scheduler_name': 'CosineDecayLRSched'}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.batch_size = batch_size
        self.timestep_count = timestep_count
        self.trial_count = trial_count
    
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
            leaky_pts = np.linspace(0, self.timestep_count-1, shuffle_locs*leaky_count+2)[1:-1].astype(int)
        leaky_1o_pts = leaky_pts[:shuffle_locs*leaky_1o_count] if leaky_1o_count > 0 else None
        leaky_2o_pts = leaky_pts[shuffle_locs*leaky_1o_count:].reshape(2, -1) if leaky_2o_count > 0 else None
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
    
    def construct_trainer(self, profiling_dataset, attack_dataset):
        trainer = LeakageLocalizationTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs=self.leakage_localization_kwargs
        )
        return trainer
    
    def run_experiment(self, logging_dir, kwargs):
        os.makedirs(logging_dir, exist_ok=True)
        if os.path.exists(os.path.join(logging_dir, 'leakage_localization', 'leakage_assessment.npz')):
            rv = np.load(os.path.join(logging_dir, 'leakage_localization', 'leakage_assessment.npz'), allow_pickle=True)
            leakage_assessment = rv['leakage_assessment']
            locs_1o = rv['locs_1o']
            locs_2o = rv['locs_2o']
        else:
            profiling_dataset, attack_dataset, locs_1o, locs_2o = self.construct_datasets(**kwargs)
            trainer = self.construct_trainer(profiling_dataset, attack_dataset)
            trainer.pretrain_classifiers(os.path.join(logging_dir, 'classifiers_pretrain'), max_steps=self.run_kwargs['max_steps'])
            leakage_assessment = trainer.run(os.path.join(logging_dir, 'leakage_localization'), pretrained_classifiers_logging_dir=os.path.join(logging_dir, 'classifiers_pretrain'), **self.run_kwargs)
            np.savez(os.path.join(logging_dir, 'leakage_localization', 'leakage_assessment.npz'), leakage_assessment=leakage_assessment, locs_1o=locs_1o, locs_2o=locs_2o)
        return leakage_assessment, locs_1o, locs_2o
    
    def run_1o_beta_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        leakage_assessments = {}
        for beta in [1 - 0.5**n for n in range(self.trial_count)][::-1]:
            subdir = os.path.join(exp_dir, f'beta={beta}')
            leakage_assessments[beta], *_ = self.run_experiment(subdir, {'lpf_beta': beta})
        plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            self.timestep_count//2,
            title=r'Sweep of low-pass filter coefficient: $\beta_{\mathrm{LPF}}$',
            to_label=lambda x: r'$\beta_{\mathrm{LPF}}='+f'{x}'+r'$'
        )
    
    def run_1o_data_var_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_data_var_sweep')
        leakage_assessments = {}
        for var in [0.0] + [10**(-n) for n in range(1, (self.trial_count-2)//2 + 1)] + [1.0] + [10**(n) for n in range(1, (self.trial_count-2)//2 + 1)]:
            subdir = os.path.join(exp_dir, f'var={var}')
            leakage_assessments[var], *_ = self.run_experiment(subdir, {'data_var': var})
        plot_leakage_assessments(
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
        for count in [0, 1] + [1 + 2**n for n in range(1, self.trial_count-1)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'leaky_1o_count': count})
            locss.append(locs)
        plot_leakage_assessments(
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
        for count in [0] + [2**n for n in range(self.trial_count-1)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'max_no_ops': count})
            locss.append(locs)
        plot_leakage_assessments(
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
        for count in [1] + [1 + 2**n for n in range(1, self.trial_count)]:
            subdir = os.path.join(exp_dir, f'count={count}')
            leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'shuffle_locs': count})
            locss.append(locs)
        plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            leakage_assessments,
            locss,
            title=r'Sweep of shuffle location count: $n_{\mathrm{shuff}}$',
            to_label=lambda x: r'$n_{\mathrm{shuff}}='+f'{x}'+r'$'
        )
    
    def run_2o_trial(self):
        exp_dir = os.path.join(self.logging_dir, '2o_trial')
        leakage_assessment, _, locs = self.run_experiment(exp_dir, {'leaky_1o_count': 0, 'leaky_2o_count': 1})
        plot_leakage_assessments(
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