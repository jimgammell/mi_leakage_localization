from typing import *
from copy import copy
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, kendalltau
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader

from common import *
from .utils import *
from datasets.data_module import DataModule
from datasets.dpav4 import DPAv4
from datasets.ascadv1 import ASCADv1
from datasets.aes_hd import AES_HD
from datasets.ed25519_wolfssl import ED25519
from datasets.one_truth_prevails import OneTruthPrevails
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from training_modules import SupervisedTrainer, SupervisedModule, LeakageLocalizationTrainer
from training_modules.supervised_deep_sca.plot_things import plot_hparam_sweep
from training_modules.cooperative_leakage_localization.plot_things import plot_ll_hparam_sweep
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.multi_attack_baseline import MultiAttackTrainer, soft_kendall_tau
from utils.template_attack import TemplateAttack

class Trial:
    def __init__(self,
        dataset_name: Literal['dpav4', 'ascadv1_fixed', 'ascadv1_variable', 'otiait', 'otp', 'aes_hd'],
        trial_config: dict,
        seed_count: int = 1,
        logging_dir: Optional[Union[str, os.PathLike]] = None
    ):
        self.dataset_name = dataset_name
        self.trial_config = trial_config
        self.seed_count = seed_count
        self.logging_dir = logging_dir if logging_dir is not None else dataset_name
        os.makedirs(self.logging_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.logging_dir, 'first_order_stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        self.supervised_model_dir = os.path.join(self.logging_dir, 'supervised_model')
        os.makedirs(self.supervised_model_dir, exist_ok=True)
        self.nn_attr_dir = os.path.join(self.logging_dir, 'nn_attr_assessments')
        os.makedirs(self.nn_attr_dir, exist_ok=True)
        self.ll_classifiers_pretrain_dir = os.path.join(self.logging_dir, 'll_classifiers_pretrain')
        os.makedirs(self.ll_classifiers_pretrain_dir, exist_ok=True)
        self.leakage_localization_dir = os.path.join(self.logging_dir, 'leakage_localization')
        os.makedirs(self.leakage_localization_dir, exist_ok=True)
        self.supervised_hparam_sweep_dir = os.path.join(self.logging_dir, 'supervised_hparam_sweep')
        os.makedirs(self.supervised_hparam_sweep_dir, exist_ok=True)
        self.ll_classifiers_hparam_sweep_dir = os.path.join(self.logging_dir, 'll_classifiers_hparam_sweep')
        os.makedirs(self.ll_classifiers_hparam_sweep_dir, exist_ok=True)
        self.ll_hparam_sweep_dir = os.path.join(self.logging_dir, 'll_hparam_sweep')
        os.makedirs(self.ll_hparam_sweep_dir, exist_ok=True)
        self.ground_truth_dir = os.path.join(self.logging_dir, 'ground_truth_assessments')
        os.makedirs(self.ground_truth_dir, exist_ok=True)
        
        print('Constructing datasets...')
        if self.dataset_name == 'dpav4':
            self.profiling_dataset = DPAv4(root=trial_config['data_dir'], train=True)
            self.attack_dataset = DPAv4(root=trial_config['data_dir'], train=False)
        elif self.dataset_name == 'ascadv1_fixed':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif self.dataset_name == 'ascadv1_variable':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=False)
        elif self.dataset_name == 'aes_hd':
            self.profiling_dataset = AES_HD(root=trial_config['data_dir'], train=True)
            self.attack_dataset = AES_HD(root=trial_config['data_dir'], train=False)
        elif dataset_name == 'otiait':
            self.profiling_dataset = ED25519(root=trial_config['data_dir'], train=True)
            self.attack_dataset = ED25519(root=trial_config['data_dir'], train=False)
        elif dataset_name == 'otp':
            self.profiling_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=True)
            self.attack_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=False)
        else:
            assert False
        print('\tDone.')
        
    def compute_ground_truth_assessments(self):
        for seed in [0]: #range(self.seed_count):
            for attack_type in ['template']: #['mlp', 'template']:
                assessments = {}
                for window_size in [5]: #[1, 3, 5]:
                    name = f'{attack_type}__seed={seed}__window_size={window_size}'
                    assessments[window_size] = {}
                    if 'ascadv1' in self.dataset_name:
                        targets = ['subbytes', 'r_out', 'subbytes__r_out', 'r_in', 'subbytes__r_in', 'r', 'subbytes__r']
                    else:
                        targets = ['subbytes']
                    fig, axes = plt.subplots(len(targets), 3, figsize=(PLOT_WIDTH*3, PLOT_WIDTH*len(targets)))
                    if len(targets) == 1:
                        axes = axes.reshape(1, 3)
                    if not os.path.exists(os.path.join(self.ground_truth_dir, f'{name}.npz')):
                        for target_idx, target in enumerate(targets):
                            self.profiling_dataset.target_values = [target]
                            self.attack_dataset.target_values = [target]
                            trainer = MultiAttackTrainer(self.profiling_dataset, self.attack_dataset, attack_type=attack_type, window_size=window_size, max_parallel_timesteps=100)
                            info = trainer.get_info()
                            assessments[window_size][target] = info
                        np.savez(os.path.join(self.ground_truth_dir, f'{name}.npz'), assessments=assessments[window_size])
                    else:
                        assessments[window_size] = np.load(os.path.join(self.ground_truth_dir, f'{name}.npz'), allow_pickle=True)['assessments'].item()
                    for target_idx, target in enumerate(targets):
                        for key_idx, key in enumerate(['log_p_y_mid_x', 'mutinf']):
                            assessment = assessments[window_size][target][key]
                            axes[target_idx, key_idx].plot(assessment, linestyle='none', marker='.', markersize=1, color='blue')
                            axes[target_idx, key_idx].set_xlabel(r'Timesteps $t$')
                            axes[target_idx, key_idx].set_ylabel(r'Estimated leakage of $X_t$')
                            axes[target_idx, key_idx].set_title('Method: ' + key.replace('_', r'\_'))
                        #rank_mean, rank_std = assessments[window_size][target]['rank_mean'], assessments[window_size][target]['rank_std']
                        #axes[target_idx, 2].fill_between(np.arange(len(assessments[window_size][target]['rank_mean'])), rank_mean-rank_std, rank_mean+rank_std, color='blue', alpha=0.25)
                        #axes[target_idx, 2].plot(rank_mean, linestyle='-', color='blue')
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.ground_truth_dir, f'{name}.png'))
                    plt.close(fig)
    
    def get_ground_truth_assessments(self):
        rv = {}
        for attack_type in ['mlp', 'template']:
            for window_size in [1, 3, 5]:
                assessment_name = f'{attack_type}__window_size={window_size}'
                if 'ascadv1' in self.dataset_name:
                    targets = ['subbytes', 'r_out', 'subbytes__r_out']
                else:
                    targets = ['subbytes']
                mutinfs = defaultdict(list)
                for seed in range(self.seed_count):
                    for target in targets:
                        name = f'{attack_type}__seed={seed}__window_size={window_size}'
                        assessment = np.load(os.path.join(self.ground_truth_dir, f'{name}.npz'), allow_pickle=True)['assessments'].item()[target]
                        mutinfs[target].append(assessment['mutinf'])
                mutinfs = {key: np.stack(val) for key, val in mutinfs.items()}
                rv[assessment_name] = mutinfs
        return rv
    
    def run_template_attacks(self):
        leakage_assessments = self.get_leakage_assessments()
        for name, assessments in leakage_assessments.items():
            if assessments.ndim == 1:
                assessments = assessments.reshape(1, -1)
            for seed, assessment in assessments:
                for poi_count in [1, 5, 9, 13]:
                    points_of_interest = assessments.argsort()[-poi_count:]
                    template_attacker = TemplateAttack(points_of_interest, target_key='label')
                    template_attacker.profile(self.profiling_dataset)
                    template_attacker.attack(self.attack_dataset)
                    
    def compute_random_assessment(self):
        self.random_assessment = {'random': np.random.randn(self.seed_count, self.profiling_dataset.timesteps_per_trace)}
        
    def compute_ascad_first_order_stats(self):
        for target in ['subbytes', 'subbytes__r_in', 'subbytes__r', 'subbytes__r_out', 'r_in', 'r_out', 'r']:
            first_order_stats = FirstOrderStatistics(self.profiling_dataset, targets=target)
            snr = first_order_stats.snr_vals[target].reshape(-1)
            plot_leakage_assessment(snr, os.path.join(self.stats_dir, f'snr_target={target}.png'))
        
    def compute_first_order_stats(self):
        if not os.path.exists(os.path.join(self.stats_dir, 'stats.npy')):
            print('Computing first-order statistical assessments...')
            first_order_stats = FirstOrderStatistics(self.profiling_dataset)
            snr = first_order_stats.snr_vals['label'].reshape(-1)
            sosd = first_order_stats.sosd_vals['label'].reshape(-1)
            cpa = first_order_stats.cpa_vals['label'].reshape(-1)
            np.save(os.path.join(self.stats_dir, 'stats.npy'), np.stack([snr, sosd, cpa]))
            print('\tDone.')
        else:
            rv = np.load(os.path.join(self.stats_dir, 'stats.npy'))
            snr = rv[0, :]
            sosd = rv[1, :]
            cpa = rv[2, :]
            print('Found precomputed first-order statistical assessments.')
        plot_leakage_assessment(snr, os.path.join(self.stats_dir, 'snr.png'))
        plot_leakage_assessment(sosd, os.path.join(self.stats_dir, 'sosd.png'))
        plot_leakage_assessment(cpa, os.path.join(self.stats_dir, 'cpa.png'))
        self.first_order_stats = {
            'snr': np.abs(snr), 'sosd': np.abs(sosd), 'cpa': np.abs(cpa)
        }
    
    def run_supervised_hparam_sweep(self):
        if not os.path.exists(os.path.join(self.supervised_hparam_sweep_dir, 'results.pickle')):
            print('Running supervised hparam sweep...')
            supervised_trainer = SupervisedTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=self.trial_config['supervised_training_kwargs'],
                default_data_module_kwargs={'gaussian_noise_std': 0.5 if self.dataset_name == 'aes_hd' else 0.0}
            )
            supervised_trainer.hparam_tune(logging_dir=self.supervised_hparam_sweep_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found existing supervised hparam sweep.')
        self.optimal_hparams = plot_hparam_sweep(self.supervised_hparam_sweep_dir)
        print(f'Optimal hyperparameters on {self.dataset_name}: {self.optimal_hparams}')
    
    def run_ll_classifiers_hparam_sweep(self):
        if not os.path.exists(os.path.join(self.ll_classifiers_hparam_sweep_dir, 'results.pickle')):
            print('Running LL classifiers hparam sweep...')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            ll_trainer = LeakageLocalizationTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=kwargs,
                default_data_module_kwargs={'gaussian_noise_std': 0.5 if self.dataset_name == 'aes_hd' else 0.0}
            )
            ll_trainer.htune_pretrain_classifiers(logging_dir=self.ll_classifiers_hparam_sweep_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found existing LL classifier pretraining sweep.')
        self.optimal_ll_pretrain_hparams = plot_hparam_sweep(self.ll_classifiers_hparam_sweep_dir)
        print(f'Optimal hyperparameters on {self.dataset_name}: {self.optimal_ll_pretrain_hparams}')
    
    def run_ll_hparam_sweep(self):
        training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
        supervised_dnn = training_module.classifier
        use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
        if not os.path.exists(os.path.join(self.ll_hparam_sweep_dir, 'results.pickle')):
            print('Running LL hparam sweep...')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            kwargs.update(self.trial_config['leakage_localization_kwargs'])
            if use_pretrained_classifiers:
                kwargs.update(self.optimal_ll_pretrain_hparams)
            ll_trainer = LeakageLocalizationTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs)
            ll_trainer.htune_leakage_localization(
                self.ll_hparam_sweep_dir,
                pretrained_classifiers_logging_dir=os.path.join(self.ll_classifiers_pretrain_dir, f'seed=0') if use_pretrained_classifiers else None,
                trial_count=25 if use_pretrained_classifiers else 50,
                max_steps=self.trial_config['max_leakage_localization_steps'],
                supervised_dnn=supervised_dnn,
                references={key: val.mean(axis=0) for key, val in self.get_ground_truth_assessments().items()}
            )
        else:
            print('Found existing LL hparam sweep.')
        self.ll_optimal_hparams = plot_ll_hparam_sweep(self.ll_hparam_sweep_dir)
        print(f'Optimal LL hyperparameters on {self.dataset_name}: {self.ll_optimal_hparams}')
        
    def run_leakage_localization(self):
        training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
        supervised_dnn = training_module.classifier
        use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
        assessments = []
        for seed in range(self.seed_count):
            subdir = os.path.join(self.leakage_localization_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            if not os.path.exists(os.path.join(subdir, 'best_checkpoint.ckpt')):
                print('Running leakage localization...')
                trainer = LeakageLocalizationTrainer(
                    self.profiling_dataset, self.attack_dataset,
                    default_training_module_kwargs=self.trial_config['default_kwargs']
                )
                leakage_localization_kwargs = copy(self.trial_config['default_kwargs'])
                leakage_localization_kwargs.update(self.trial_config['leakage_localization_kwargs'])
                leakage_localization_kwargs.update({'supervised_dnn': supervised_dnn})
                leakage_localization_kwargs.update(self.ll_optimal_hparams)
                leakage_assessment = trainer.run(
                    logging_dir=subdir,
                    pretrained_classifiers_logging_dir=os.path.join(self.ll_classifiers_pretrain_dir, f'seed={seed}') if use_pretrained_classifiers else None,
                    max_steps=self.trial_config['max_leakage_localization_steps'],
                    override_kwargs=leakage_localization_kwargs,
                    anim_gammas=False
                )
            else:
                assert os.path.exists(os.path.join(subdir, 'leakage_assessment.npy'))
                leakage_assessment = np.load(os.path.join(subdir, 'leakage_assessment.npy'))
            assessments.append(leakage_assessment)
        self.leakage_localization_assessments = {
            'leakage_localization': np.stack(assessments)
        }
        print('\tDone.')
    
    def train_supervised_model(self):
        for subdir in ['ll_eval', *[f'seed={seed}' for seed in range(self.seed_count)]]:
            os.makedirs(os.path.join(self.supervised_model_dir, subdir), exist_ok=True)
            if not os.path.exists(os.path.join(self.supervised_model_dir, subdir, 'final_checkpoint.ckpt')):
                print('Training supervised model...')
                training_module_kwargs = copy(self.trial_config['supervised_training_kwargs'])
                training_module_kwargs.update(self.optimal_hparams)
                supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=training_module_kwargs)
                supervised_trainer.run(logging_dir=os.path.join(self.supervised_model_dir, subdir), max_steps=self.trial_config['max_classifiers_pretrain_steps'])
                print('\tDone.')
            else:
                print('Found pretrained supervised model.')
    
    def plot_supervised_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        min_ranks, min_losses = [], []
        for seed in range(self.seed_count):
            color = colormap(seed)
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            assert os.path.exists(os.path.join(subdir, 'final_checkpoint.ckpt'))
            training_curves = load_training_curves(subdir)
            axes[0].plot(*training_curves['train_rank'], linestyle='--', color=color, **PLOT_KWARGS)
            axes[0].plot(*training_curves['val_rank'], linestyle='-', color=color, **PLOT_KWARGS)
            axes[1].plot(*training_curves['train_loss'], linestyle='--', color=color)
            axes[1].plot(*training_curves['val_loss'], linestyle='-', color=color)
            optimal_idx = np.argmin(training_curves['val_rank'][-1])
            min_ranks.append(training_curves['val_rank'][-1][optimal_idx])
            min_losses.append(training_curves['val_loss'][-1][optimal_idx])
        axes[0].set_xlabel('Training step')
        axes[1].set_xlabel('Training step')
        axes[0].set_ylabel('Rank')
        axes[1].set_ylabel('Loss')
        class_count = self.profiling_dataset.class_count
        axes[0].axhline((class_count+1)/2, color='black', linestyle=':')
        axes[1].axhline(np.log(class_count), color='black', linestyle=':')
        legend_elems = [
            Line2D([0], [0], color='black', linestyle='--', label='train'),
            Line2D([0], [0], color='black', linestyle='-', label='val'),
            Line2D([0], [0], color='black', linestyle=':', label='random guessing')
        ]
        axes[0].legend(handles=legend_elems, loc='lower left')
        axes[1].legend(handles=legend_elems, loc='lower left')
        axes[1].set_yscale('symlog')
        dset_name = self.dataset_name.replace(r'_', r'\_')
        fig.suptitle(f'Dataset: {dset_name}')
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, 'training_curves.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
        print(f'Supervised ranks: {np.mean(min_ranks)} +/- {np.std(min_ranks)}')
        print(f'Supervised losses: {np.mean(min_losses)} +/- {np.std(min_losses)}')
    
    def compute_supervised_ranks_over_time(self, wouters_zaid_model=None):
        data_module = DataModule(self.profiling_dataset, self.attack_dataset)
        attack_dataloader = data_module.test_dataloader()
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        for seed in range(self.seed_count):
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
            assert to_name('') is not None
            assert os.path.exists(subdir)
            if not os.path.exists(os.path.join(subdir, to_name('rank_over_time.npy'))):
                evaluator = AESMultiTraceEvaluator(attack_dataloader, subdir if wouters_zaid_model is None else wouters_zaid_model, seed=seed, dataset_name=self.dataset_name)
                rank_over_time = evaluator()
                np.save(os.path.join(subdir, to_name('rank_over_time.npy')), rank_over_time)
            else:
                rank_over_time = np.load(os.path.join(subdir, to_name('rank_over_time.npy')))
            color = colormap(seed)
            ax.plot(np.arange(1, len(rank_over_time)+1), rank_over_time, color=color)
        ax.axhline(0.5*(self.profiling_dataset.class_count+1), linestyle=':', color='black', label='random guessing')
        ax.legend(loc='upper right')
        ax.set_xscale('log')
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank')
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, to_name('rank_over_time.png')), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def create_paper_rot_plot(self):
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        colormap = plt.cm.get_cmap('tab10', self.seed_count)
        for seed in range(self.seed_count):
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            rank_over_time = np.load(os.path.join(subdir, 'rank_over_time.npy'))
            color = colormap(seed)
            ax.plot(np.arange(1, len(rank_over_time)+1), rank_over_time, color=color)
            if os.path.exists(os.path.join(subdir, 'wouters_rank_over_time.npy')):
                wouters_rank_over_time = np.load(os.path.join(subdir, 'wouters_rank_over_time.npy'))
                ax.plot(np.arange(1, len(wouters_rank_over_time)+1), wouters_rank_over_time, color=color, linestyle='--')
            else:
                wouters_rank_over_time = None
            if os.path.exists(os.path.join(subdir, 'zaid_rank_over_time.npy')):
                zaid_rank_over_time = np.load(os.path.join(subdir, 'zaid_rank_over_time.npy'))
                ax.plot(np.arange(1, len(zaid_rank_over_time)+1), zaid_rank_over_time, color=color, linestyle='-.')
            else:
                zaid_rank_over_time = None
        ax.axhline(0.5*(self.profiling_dataset.class_count+1), linestyle=':', color='black', label='random guessing')
        legend_handles = [
            Line2D([0], [0], color='black', linestyle=':', label='random guessing'),
            Line2D([0], [0], color='gray', linestyle='-', label='ours')
        ]
        if wouters_rank_over_time is not None:
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='--', label='Wouters et al.'))
        if zaid_rank_over_time is not None:
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='-.', label='Zaid et al.'))
        ax.legend(loc='upper right', handles=legend_handles)
        ax.set_xscale('log')
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank')
        ax.set_title('Dataset:' + self.dataset_name.replace('_', r'\_'))
        fig.tight_layout()
        fig.savefig(os.path.join(self.supervised_model_dir, 'paper_rank_over_time.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
        
    def compute_neural_net_attributions(self, wouters_zaid_model=None):
        data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
        profiling_dataloader = data_module.train_dataloader()
        gradviss, saliencies, occlusions, inputxgrads, lrps = [], [], [], [], []
        for seed in range(self.seed_count):
            subdir = os.path.join(self.nn_attr_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            nn_attributor = NeuralNetAttribution(profiling_dataloader, os.path.join(self.supervised_model_dir, f'seed={seed}') if wouters_zaid_model is None else wouters_zaid_model, seed=seed)
            to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
            assert to_name('') is not None
            if wouters_zaid_model is None and not os.path.exists(os.path.join(subdir, to_name('lrp.npy'))):
                print('Computing LRP...')
                lrp = nn_attributor.compute_lrp().reshape(-1)
                np.save(os.path.join(subdir, to_name('lrp.npy')), lrp)
                print('\tDone.')
            elif wouters_zaid_model is None:
                lrp = np.load(os.path.join(subdir, to_name('lrp.npy')))
                print('Found precomputed LRP.')
            if not os.path.exists(os.path.join(subdir, to_name('gradvis.npy'))):
                print('Computing GradVis...')
                gradvis = nn_attributor.compute_gradvis().reshape(-1)
                np.save(os.path.join(subdir, to_name('gradvis.npy')), gradvis)
                print('\tDone.')
            else:
                gradvis = np.load(os.path.join(subdir, to_name('gradvis.npy')))
                print('Found precomputed GradVis.')
            if not os.path.exists(os.path.join(subdir, to_name('saliency.npy'))):
                print('Computing saliency...')
                saliency = nn_attributor.compute_saliency().reshape(-1)
                np.save(os.path.join(subdir, to_name('saliency.npy')), saliency)
                print('\tDone.')
            else:
                saliency = np.load(os.path.join(subdir, to_name('saliency.npy')))
                print('Found precomputed saliency.')
            r"""if not os.path.exists(os.path.join(subdir, to_name('occlusion.npy'))):
                print('Computing occlusion...')
                occlusion = nn_attributor.compute_occlusion().reshape(-1)
                np.save(os.path.join(subdir, to_name('occlusion.npy')), occlusion)
                print('\tDone.')
            else:
                occlusion = np.load(os.path.join(subdir, to_name('occlusion.npy')))
                print('Found precomputed occlusion.')"""
            if not os.path.exists(os.path.join(subdir, to_name('inputxgrad.npy'))):
                print('Computing inputxgrad...')
                inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
                np.save(os.path.join(subdir, to_name('inputxgrad.npy')), inputxgrad)
                print('\tDone.')
            else:
                inputxgrad = np.load(os.path.join(subdir, to_name('inputxgrad.npy')))
                print('Found precomputed inputxgrad.')
            plot_leakage_assessment(gradvis, os.path.join(subdir, to_name('gradvis.png')))
            plot_leakage_assessment(saliency, os.path.join(subdir, to_name('saliency.png')))
            #plot_leakage_assessment(occlusion, os.path.join(subdir, to_name('occlusion.png')))
            plot_leakage_assessment(inputxgrad, os.path.join(subdir, to_name('inputxgrad.png')))
            if wouters_zaid_model is None:
                plot_leakage_assessment(lrp, os.path.join(subdir, to_name('lrp.png')))
                lrps.append(lrp)
            gradviss.append(gradvis)
            saliencies.append(saliency)
            #occlusions.append(occlusion)
            inputxgrads.append(inputxgrad)
        setattr(self, to_name('nn_attr_assessments'), {
            to_name('gradvis'): np.stack(gradviss), to_name('saliency'): np.stack(saliencies), to_name('inputxgrad'): np.stack(inputxgrads),# to_name('occlusion'): np.stack(occlusions)
            **({to_name('lrp'): np.stack(lrps)} if wouters_zaid_model is None else {})
        })
    
    def pretrain_leakage_localization_classifiers(self):
        for seed in range(self.seed_count):
            subdir = os.path.join(self.ll_classifiers_pretrain_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            if not os.path.exists(os.path.join(subdir, 'final_checkpoint.ckpt')):
                print('Pretraining leakage localization classifiers...')
                trainer = LeakageLocalizationTrainer(
                    self.profiling_dataset, self.attack_dataset,
                    default_training_module_kwargs=self.trial_config['default_kwargs']
                )
                classifiers_pretrain_kwargs = copy(self.trial_config['default_kwargs'])
                classifiers_pretrain_kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
                classifiers_pretrain_kwargs.update(self.optimal_ll_pretrain_hparams)
                trainer.pretrain_classifiers(
                    logging_dir=subdir,
                    max_steps=self.trial_config['max_classifiers_pretrain_steps'],
                    override_kwargs=classifiers_pretrain_kwargs
                )
                print('\tDone.')
            else:
                print('Found pretrained leakage localization classifiers.')
    
    def get_leakage_assessments(self):
        leakage_assessments = {}
        leakage_assessments.update(self.random_assessment)
        if hasattr(self, 'first_order_stats'):
            leakage_assessments.update(self.first_order_stats)
        if hasattr(self, 'nn_attr_assessments'):
            leakage_assessments.update(self.nn_attr_assessments)
        if hasattr(self, 'zaid_nn_attr_assessments'):
            leakage_assessments.update(self.zaid_nn_attr_assessments)
        if hasattr(self, 'wouters_nn_attr_assessments'):
            leakage_assessments.update(self.wouters_nn_attr_assessments)
        if hasattr(self, 'leakage_localization_assessments'):
            leakage_assessments.update(self.leakage_localization_assessments)
        return leakage_assessments
    
    def eval_leakage_assessments(self): # should print out valid code for a Latex booktabs table
        leakage_assessments = self.get_leakage_assessments()
        ground_truth_assessments = self.get_ground_truth_assessments()
        kendalltau_evaluations = {}
        for leakage_assessment_name, leakage_assessment in leakage_assessments.items():
            kendalltau_evaluations[leakage_assessment_name] = {}
            print(leakage_assessment_name)
            for ground_truth_assessment_name, ground_truth_assessment in ground_truth_assessments.items():
                print(f'\t{ground_truth_assessment_name}')
                ground_truth_assessment = np.mean(np.stack(list(ground_truth_assessment.values())), axis=0)
                kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name] = []
                for seed in range(ground_truth_assessment.shape[0]):
                    if leakage_assessment.ndim == 1:
                        _leakage_assessment = leakage_assessment
                    else:
                        _leakage_assessment = leakage_assessment[seed, :]
                    window_size = int(ground_truth_assessment_name.split('=')[-1])
                    _leakage_assessment = torch.tensor(_leakage_assessment).unfold(0, window_size, 1).numpy().mean(axis=-1)
                    #mutinf, rank_mean, rank_std = ground_truth_assessment[seed, 0, :], ground_truth_assessment[seed, 1, :], ground_truth_assessment[seed, 2, :]
                    mutinf = ground_truth_assessment[seed, :]
                    kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].append(soft_kendall_tau(_leakage_assessment, mutinf))
                kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name] = np.stack(kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name])
                print(f'\t\tKendall tau: {kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].mean()} +/- {kendalltau_evaluations[leakage_assessment_name][ground_truth_assessment_name].std()}')
        return kendalltau_evaluations
    
    def create_main_paper_leakage_assessment_plots(self):
        leakage_assessments = self.get_leakage_assessments()['leakage_localization'].reshape(self.seed_count, -1)
        ground_truth_assessments = self.get_ground_truth_assessments()['template__window_size=5']
        fig, axes = plt.subplots(2, 1, figsize=(PLOT_WIDTH, 2*PLOT_WIDTH))
        cmap = cm.get_cmap('tab10', len(ground_truth_assessments))
        colors = [cmap(i) for i in range(len(ground_truth_assessments))]
        for (key, _ground_truth_assessment), color in zip(ground_truth_assessments.items(), colors):
            axes[0].fill_between(range(_ground_truth_assessment.shape[-1]), np.min(_ground_truth_assessment, axis=0), np.max(_ground_truth_assessment, axis=0), color=color, alpha=0.25, **PLOT_KWARGS)
            axes[0].plot(np.median(_ground_truth_assessment, axis=0), color=color, marker='.', markersize=1, linestyle='none', label=key.replace('_', r'\_'), **PLOT_KWARGS)
        axes[1].fill_between(range(leakage_assessments.shape[-1]), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
        axes[1].plot(np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].set_xlabel(r'Timestep $t$')
        axes[0].set_ylabel(r'Estimated leakage of $X_t$')
        axes[0].set_title('Ground truth-like assessment')
        axes[1].set_xlabel(r'Timestep $t$')
        axes[1].set_ylabel(r'Estimated leakage of $X_t$')
        axes[1].set_title('Adversarial leakage localization (ours)')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_leakage_assessments.pdf'), **SAVEFIG_KWARGS)
    
    def plot_leakage_assessments(self):
        leakage_assessments = self.get_leakage_assessments()
        row_count = 2
        col_count = int(np.ceil(len(leakage_assessments)/row_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(col_count*PLOT_WIDTH, row_count*PLOT_WIDTH))
        for ax, (la_name, la) in zip(axes.flatten(), leakage_assessments.items()):
            if la.ndim == 1:
                ax.plot(la, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            elif la.ndim == 2:
                median = np.median(la, axis=0)
                min = np.min(la, axis=0)
                max = np.max(la, axis=0)
                #ax.errorbar(range(len(median)), median, yerr=[median-min, max-median], fmt='none', ecolor='red', label='min--max')
                ax.plot(median, color='blue', marker='.', linestyle='none', markersize=1, label='median', **PLOT_KWARGS)
                ax.legend()
            else:
                assert False
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakage of $X_t$')
            la_name = la_name.replace('_', r'\_')
            ax.set_title(f'Technique: {la_name}')
        for ax in axes.flatten()[len(leakage_assessments):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'leakage_assessments.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def __call__(self):
        self.compute_random_assessment()
        if 'ascad' in self.dataset_name:
            self.compute_ascad_first_order_stats()
        if ('compute_ground_truth_assessments' in self.trial_config) and self.trial_config['compute_ground_truth_assessments']:
            self.compute_ground_truth_assessments()
        if ('compute_first_order_stats' in self.trial_config) and self.trial_config['compute_first_order_stats']:
            self.compute_first_order_stats()
        if ('run_supervised_hparam_sweep' in self.trial_config) and self.trial_config['run_supervised_hparam_sweep']:
            self.run_supervised_hparam_sweep()
        if ('train_supervised_model' in self.trial_config) and self.trial_config['train_supervised_model']:
            self.train_supervised_model()
            self.plot_supervised_training_curves()
        if ('compute_nn_attributions' in self.trial_config) and self.trial_config['compute_nn_attributions']:
            self.compute_neural_net_attributions()
            if self.dataset_name == 'dpav4':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__DPAv4')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__DPAv4')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__DPAv4')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__DPAv4')
                self.create_paper_rot_plot()
            elif self.dataset_name == 'ascadv1_fixed':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__ASCADv1f')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__ASCADv1f')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__ASCADv1f')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__ASCADv1f')
                self.create_paper_rot_plot()
            elif self.dataset_name == 'ascadv1_variable':
                self.compute_supervised_ranks_over_time()
                self.create_paper_rot_plot()
            elif self.dataset_name == 'aes_hd':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__AES_HD')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__AES_HD')
                self.compute_supervised_ranks_over_time()
                self.compute_supervised_ranks_over_time(wouters_zaid_model='ZaidNet__AES_HD')
                self.compute_supervised_ranks_over_time(wouters_zaid_model='WoutersNet__AES_HD')
                self.create_paper_rot_plot()
        if self.dataset_name not in ['otiait', 'otp', 'dpav4']:
            if ('run_ll_classifiers_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_classifiers_hparam_sweep']:
                self.run_ll_classifiers_hparam_sweep()
            if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers']:
                self.pretrain_leakage_localization_classifiers()
        if ('run_ll_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_hparam_sweep']:
            self.run_ll_hparam_sweep()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_leakage_localization()
        self.eval_leakage_assessments()
        self.plot_leakage_assessments()
        self.create_main_paper_leakage_assessment_plots()