from typing import *
from copy import copy
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib import gridspec
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import *
from .utils import *
from datasets.data_module import DataModule
from datasets.dpav4 import DPAv4
from datasets.ascadv1 import ASCADv1
from datasets.aes_hd import AES_HD
from datasets.ed25519_wolfssl import ED25519
from datasets.one_truth_prevails import OneTruthPrevails
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from training_modules import SupervisedTrainer, SupervisedModule, LeakageLocalizationTrainer, LeakageLocalizationModule
from training_modules.supervised_deep_sca.plot_things import plot_hparam_sweep
from training_modules.cooperative_leakage_localization.plot_things import plot_ll_hparam_sweep
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.multi_attack_baseline import MultiAttackTrainer, soft_kendall_tau
from utils.template_attack import TemplateAttack
from utils.dnn_performance_auc import compute_dnn_performance_auc
from utils.baseline_assessments.occpoi import OccPOI

OCCL_VALS = [1, 5, 17, 65, 257]
GAMMAO_VALS = np.arange(0.05, 1.0, 0.05)

def get_assessment_name(key):
    lut = {
        'random': 'Random',
        'ground_truth': 'Ground Truth',
        'snr': 'SNR',
        'sosd': 'SOSD',
        'cpa': 'CPA',
        'gradvis': 'GradVis',
        'lrp': 'LRP',
        'occlusion': '1-Occlusion',
        'saliency': 'Saliency',
        'inputxgrad': r'Input $*$ Grad',
        'zaid_gradvis': 'GradVis (ZaidNet)',
        'zaid_occlusion': '1-Occlusion (ZaidNet)',
        'zaid_saliency': 'Saliency (ZaidNet)',
        'zaid_inputxgrad': r'Input $*$ Grad (ZaidNet)',
        'wouters_gradvis': 'GradVis (WoutersNet)',
        'wouters_occlusion': '1-Occlusion (WoutersNet)',
        'wouters_saliency': 'Saliency (WoutersNet)',
        'wouters_inputxgrad': r'Input $*$ Grad (WoutersNet)',
        'leakage_localization': 'Adversarial Leakage Localization (Ours)',
        'occpoi': 'OccPOI',
        'zaid_occpoi': 'OccPOI (ZaidNet)',
        'wouters_occpoi': 'OccPOI (WoutersNet)',
        'second_order_occlusion': '2nd-order Occlusion'
    }
    for n in OCCL_VALS:
        lut[f'occl_{n}'] = f'{n}-Occlusion'
        lut[f'wouters_occl_{n}'] = f'{n}-Occlusion (WoutersNet)'
        lut[f'zaid_occl_{n}'] = f'{n}-Occlusion (ZaidNet)'
    return lut[key]

def get_sensitive_variable_label_and_color(key, dataset_name):
    cmap = cm.get_cmap('tab10', 7)
    colors = [cmap(i) for i in range(7)]
    if 'ascad' in dataset_name:
        labels = {
            'subbytes': r'$\operatorname{Sbox}(k_3 \oplus w_3)$',
            'r_out': r'$r_{\mathrm{out}}$',
            'subbytes__r_out': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$',
            'r_in': r'$r_{\mathrm{in}}$',
            'subbytes__r_in': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{in}}$',
            'r': r'$r$',
            'subbytes__r': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r$'
        }
        label = labels[key]
        color = colors[list(labels.keys()).index(key)]
    elif dataset_name == 'dpav4':
        label = r'$\operatorname{Sbox}(k_0 \oplus w_0) \oplus m_0$'
        color = colors[0]
    elif dataset_name == 'aes_hd':
        label = r'$\operatorname{Sbox}^{-1}(k_{11}^* \oplus c_{11}) \oplus c_7$'
        color = colors[0]
    elif dataset_name == 'otiait':
        label = 'Ephemeral key nibble'
        color = colors[0]
    elif dataset_name == 'otp':
        label = 'Dummy load?'
        color = colors[0]
    else:
        assert False
    return (label, color)

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
        self.dnn_auc_dir = os.path.join(self.logging_dir, 'dnn_auc')
        os.makedirs(self.dnn_auc_dir, exist_ok=True)
        
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
        
    r"""def compute_ground_truth_assessments(self):
        for seed in range(self.seed_count):
            for attack_type in ['template']: #['mlp', 'template']:
                assessments = {}
                for window_size in [5]: #[1, 3, 5]:
                    name = f'{attack_type}__seed={seed}__window_size={window_size}'
                    assessments[window_size] = {}
                    if 'ascadv1' in self.dataset_name:
                        targets = ['subbytes', 'r_out', 'subbytes__r_out', 'r_in', 'subbytes__r_in', 'r', 'subbytes__r', 'k__p__r_in']
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
                    plt.close(fig)"""
    
    def compute_ground_truth_assessments(self):
        if 'ascadv1' in self.dataset_name:
            targets = [ # based on Egger (2021) findings
                'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
            ]
        else: # these are unprotected so leakage is dominated by the SubBytes variable itself
            targets = ['label']
        if not os.path.exists(os.path.join(self.ground_truth_dir, 'assessments.npz')):
            if self.dataset_name == 'dpav4':
                dataset = DPAv4(root=self.trial_config['data_dir'], train=False, ground_truth=True)
            else:
                dataset = self.attack_dataset
            snr = FirstOrderStatistics(dataset, targets).snr_vals
            np.savez(os.path.join(self.ground_truth_dir, 'assessments.npz'), **snr)
        else:
            snr = np.load(os.path.join(self.ground_truth_dir, 'assessments.npz'), allow_pickle=True)
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        if 'ascadv1' in self.dataset_name:
            ax.plot(snr['r_in'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$r_{\mathrm{in}}$')
            ax.plot(snr['r'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$r_3$')
            ax.plot(snr['r_out'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$r_{\mathrm{out}}$')
            ax.plot(snr['plaintext__key__r_in'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$k_3 \oplus w_3 \oplus r_{\mathrm{in}}$')
            ax.plot(snr['subbytes__r'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_3$')
            ax.plot(snr['subbytes__r_out'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$')
            ax.plot(snr['s_prev__subbytes__r_out'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$S_{\mathrm{prev}} \oplus \operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$')
            ax.plot(snr['security_load'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$\operatorname{Sbox}(S_{\mathrm{prev}} \oplus r_{\mathrm{in}}) \oplus r_{\mathrm{out}}$')
        elif self.dataset_name == 'dpav4':
            ax.plot(snr['label'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$\operatorname{Sbox}(k_0 \oplus w_0) \oplus m_0$')
        elif self.dataset_name == 'aes_hd':
            ax.plot(snr['label'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'$\operatorname{Sbox}(k^*_{11} \oplus c_{11}) \oplus c_7$')
        elif self.dataset_name == 'otp':
            ax.plot(snr['label'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'Dummy load?')
        elif self.dataset_name == 'otiait':
            ax.plot(snr['label'].squeeze(), linestyle='-', linewidth=0.1, marker='.', markersize=1, label=r'Ephemeral key nibble')
        else:
            assert False
        ax.set_xlabel(r'Timestep $t$')
        ax.set_ylabel(r'Estimated leakage of $X_t$')
        ax.set_title('`Omniscient\' SNR')
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.ground_truth_dir, f'omniscient_snr__dataset={self.dataset_name}.png'), **SAVEFIG_KWARGS)
        plt.close(fig)

    def get_ground_truth_assessments(self):
        r"""rv = {}
        for attack_type in ['template']: #['mlp', 'template']:
            for window_size in [5]: #[1, 3, 5]:
                assessment_name = f'{attack_type}__window_size={window_size}'
                if 'ascadv1' in self.dataset_name:
                    targets = ['r_out', 'subbytes__r_out', 'r_in', 'subbytes__r_in', 'r', 'subbytes__r', 'subbytes']
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
        return rv"""
        assessments = np.load(os.path.join(self.ground_truth_dir, 'assessments.npz'), allow_pickle=True)
        return assessments
    
    def eval_leakage_assessments(self): # should print out valid code for a Latex booktabs table
        leakage_assessments = self.get_leakage_assessments()
        ground_truth_assessments = self.get_ground_truth_assessments()
        ground_truth_assessment = np.stack(list(ground_truth_assessments.values())).mean(axis=0)
        spearmanr_evaluations = defaultdict(list)
        for leakage_assessment_name, leakage_assessment in leakage_assessments.items():
            for leakage_assessment_sample in leakage_assessment.reshape(-1, leakage_assessment.shape[-1]):
                if leakage_assessment_sample.var() > 0:
                    corr = spearmanr(leakage_assessment_sample.squeeze(), ground_truth_assessment.squeeze()).statistic
                else:
                    corr = 0.
                spearmanr_evaluations[leakage_assessment_name].append(corr)
        spearmanr_evaluations = {key: np.stack(val) for key, val in spearmanr_evaluations.items()}
        for key, val in spearmanr_evaluations.items():
            print(f'method={key}: spearmanr={val.mean()}+/-{val.std()}')
        return spearmanr_evaluations
    
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
        if not os.path.exists(os.path.join(self.stats_dir, 'stats.npz')):
            stats = {}
            for target in [ # based on Egger (2021) findings
                    'subbytes', 'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r',
                    'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
                ]:
                first_order_stats = FirstOrderStatistics(self.profiling_dataset, targets=target)
                snr = first_order_stats.snr_vals[target].reshape(-1)
                stats[target] = snr
            np.savez(os.path.join(self.stats_dir, 'stats.npz'), **stats)
        else:
            stats = np.load(os.path.join(self.stats_dir, 'stats.npz'), allow_pickle=True)
        for target, snr in stats.items():
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
    
    def compute_dnn_auc_vals_on_baselines(self):
        leakage_assessments = self.get_leakage_assessments()
        forward_auc_vals, reverse_auc_vals = defaultdict(list), defaultdict(list)
        for seed in range(self.seed_count):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, f'seed={(seed+1)%self.seed_count}', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            for assessment_name, _leakage_assessments in leakage_assessments.items():
                if _leakage_assessments.ndim > 1:
                    leakage_assessment = _leakage_assessments[seed, :]
                else:
                    assert _leakage_assessments.ndim == 1
                    leakage_assessment = _leakage_assessments
                auc_vals = compute_dnn_performance_auc(
                    DataLoader(self.attack_dataset, batch_size=len(self.attack_dataset), num_workers=max(1, os.cpu_count()//4)),
                    supervised_dnn, leakage_assessment, device='cuda', cluster_count=10
                )
                forward_auc_vals[assessment_name].append(auc_vals['forward_dnn_auc'])
                reverse_auc_vals[assessment_name].append(auc_vals['reverse_dnn_auc'])
        forward_auc_vals, reverse_auc_vals = {key: np.stack(val) for key, val in forward_auc_vals.items()}, {key: np.stack(val) for key, val in reverse_auc_vals.items()}
        for key in leakage_assessments.keys():
            print(f'Assessment: {key}')
            print(f'\tForward AUC: {forward_auc_vals[key].mean()} +/- {forward_auc_vals[key].std()}')
            print(f'\tReverse AUC: {reverse_auc_vals[key].mean()} +/- {reverse_auc_vals[key].std()}')
    
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
    
    def run_single_ablation_study(self, search_settings: List[dict], fixed_settings: dict, study_name: str, use_pretrained_classifiers: bool):
        training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
        supervised_dnn = training_module.classifier
        if not os.path.exists(os.path.join(self.leakage_localization_dir, study_name, f'result.npy')):
            assessments = []
            for search_setting in search_settings:
                subdir = os.path.join(self.leakage_localization_dir, study_name, '__'.join((f'{key}={val}' for key, val in search_setting.items())))
                os.makedirs(subdir, exist_ok=True)
                print(f'Running ablation study: {study_name}')
                trainer = LeakageLocalizationTrainer(
                    self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=self.trial_config['default_kwargs']
                )
                kwargs = copy(self.trial_config['default_kwargs'])
                kwargs.update(self.trial_config['leakage_localization_kwargs'])
                kwargs.update({'supervised_dnn': supervised_dnn})
                kwargs.update(self.ll_optimal_hparams)
                kwargs.update(fixed_settings)
                kwargs.update(search_setting)
                assessment = trainer.run(
                    logging_dir=subdir,
                    pretrained_classifiers_logging_dir=os.path.join(self.ll_classifiers_pretrain_dir, f'seed={0}') if use_pretrained_classifiers else None,
                    max_steps=self.trial_config['max_leakage_localization_steps'],
                    override_kwargs=kwargs,
                    anim_gammas=False
                )
                assessments.append(assessment.squeeze())
            assessments = np.stack(assessments)
            np.save(os.path.join(self.leakage_localization_dir, study_name, 'result.npy'), assessments)
        else:
            assessments = np.load(os.path.join(self.leakage_localization_dir, study_name, 'result.npy'))
        ground_truth_assessments = self.get_ground_truth_assessments()
        ground_truth_assessment = np.stack(list(ground_truth_assessments.values())).mean(axis=0).squeeze()
        osnrs = np.stack([
            spearmanr(assessment, ground_truth_assessment).statistic for assessment in assessments
        ])
        return assessments, osnrs
    
    def run_ll_ablation_study(self):
        use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
        basic_assessments, basic_osnrs = self.run_single_ablation_study(
            search_settings=[{'starting_prob': gammao} for gammao in np.arange(0.1, 1.0, 0.1)],
            fixed_settings={},
            study_name='baseline',
            use_pretrained_classifiers=use_pretrained_classifiers
        )
        maximax_assessments, maximax_osnrs = self.run_single_ablation_study(
            search_settings=[{'starting_prob': gammao} for gammao in np.arange(0.1, 1.0, 0.1)],
            fixed_settings={'adversarial_mode': False},
            study_name='maximax',
            use_pretrained_classifiers=use_pretrained_classifiers
        )
        norm_assessments, norm_osnrs = self.run_single_ablation_study(
            search_settings=[{'norm_penalty': penalty} for penalty in np.logspace(-2, 2, 9)[::-1]],
            fixed_settings={'no_budget': True},
            study_name='norm_penalty',
            use_pretrained_classifiers=use_pretrained_classifiers
        )
        fixed_classifier_assessments, fixed_classifier_osnrs = self.run_single_ablation_study(
            search_settings=[{'starting_prob': gammao} for gammao in np.arange(0.1, 1.0, 0.1)],
            fixed_settings={'train_theta': False},
            study_name='fixed_classifier',
            use_pretrained_classifiers=True
        )
        pretrained_classifier_assessments, pretrained_classifier_osnrs = self.run_single_ablation_study(
            search_settings=[{'starting_prob': gammao} for gammao in np.arange(0.1, 1.0, 0.1)],
            fixed_settings={'standard_classifier_dir': os.path.join(self.supervised_model_dir, f'seed={0}'), 'train_theta': False},
            study_name='pretrained_classifier',
            use_pretrained_classifiers=False
        )
        concrete_assessments, concrete_osnrs = self.run_single_ablation_study(
            search_settings=[{'starting_prob': gammao} for gammao in np.arange(0.1, 1.0, 0.1)],
            fixed_settings={'gradient_estimator': 'CONCRETE', 'fixed_concrete_temperature': 1.0},
            study_name='concrete_temp=1.0',
            use_pretrained_classifiers=use_pretrained_classifiers
        )
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        tax = ax.twiny()
        tax.plot(np.arange(0.1, 1.0, 0.1), basic_osnrs, color='blue', marker='.', label='No ablation', **PLOT_KWARGS)
        tax.plot(np.arange(0.1, 1.0, 0.1), concrete_osnrs, color='red', marker='.', label=r'REBAR $\to$ CONCRETE($\lambda=1$)', **PLOT_KWARGS)
        line, = ax.plot(np.logspace(-2, 2, 9)[::-1], norm_osnrs, color='orange', marker='.', label=r'Fixed $\boldsymbol{\gamma}$ budget $\to$ Penalize $\mathbb{E}\left[\lVert \boldsymbol{\mathcal{A}}_{\boldsymbol{\gamma}} \rVert_1 + \lVert \boldsymbol{\mathcal{A}}_{\boldsymbol{\gamma}} \rVert_2\right]$', **PLOT_KWARGS)
        tax.plot(np.arange(0.1, 1.0, 0.1), maximax_osnrs, color='green', marker='.', label=r'Adversarial $\to$ ``Cooperative"', **PLOT_KWARGS)
        tax.plot(np.arange(0.1, 1.0, 0.1), fixed_classifier_osnrs, color='purple', marker='.', label=r'Alternating SGD $\to$ fully train $\boldsymbol{\theta}$, then $\boldsymbol{\overline{\eta}}$', **PLOT_KWARGS)
        tax.plot(np.arange(0.1, 1.0, 0.1), pretrained_classifier_osnrs, color='brown', marker='.', label=r'Alternating SGD $\to$ ``interpret" standard classifier', **PLOT_KWARGS)
        traces = [
            ('No ablation', basic_osnrs, 'blue'),
            ('CONCRETE', concrete_osnrs, 'red'),
            ('Norm penalty', norm_osnrs, 'orange'),
            ('Maximax', maximax_osnrs, 'green'),
            (r'Train $\boldsymbol{\theta}$, then $\overline{\boldsymbol{\eta}}$', fixed_classifier_osnrs, 'purple'),
            (None, pretrained_classifier_osnrs, 'brown')
        ]
        best_label, best_trace, best_color = max(traces, key=lambda t: np.max(t[1]))
        best_max = np.max(best_trace)
        tax.axhline(best_max, color=best_color, linestyle=':', linewidth=1.5)
        tax.set_xlabel(r'Budget: $\overline{\gamma}$')
        ax.set_xlabel(r'Norm penalty coefficient: $\lambda$')
        ax.set_xscale('log')
        ax.set_ylabel(r'oSNR value$\uparrow$')
        lines1, labels1 = tax.get_legend_handles_labels()
        leg = tax.legend(lines1+[line], labels1+[line.get_label()], loc='lower right', fontsize='x-small')
        fig.tight_layout()
        fig.savefig(os.path.join(self.leakage_localization_dir, 'ablation_results.pdf'), **SAVEFIG_KWARGS)
        fig.savefig(os.path.join(self.leakage_localization_dir, 'ablation_results.png'), **SAVEFIG_KWARGS)
    
    def run_ll_gammao_sweep(self):
        if not os.path.exists(os.path.join(self.leakage_localization_dir, 'gammao_sweep.npy')):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, 'll_eval', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            use_pretrained_classifiers = self.dataset_name not in ['otp', 'otiait', 'dpav4']
            assessments = np.full((self.seed_count, len(GAMMAO_VALS), self.profiling_dataset.data_shape[-1]), np.nan, dtype=np.float32)
            for seed in range(self.seed_count):
                for gammao_idx, gammao in enumerate(GAMMAO_VALS):
                    subdir = os.path.join(self.leakage_localization_dir, f'gammao={gammao}_seed={seed}')
                    os.makedirs(subdir, exist_ok=True)
                    if not os.path.exists(os.path.join(subdir, 'best_checkpoint.ckpt')):
                        print(f'Running LL hparam sweep with gammao={gammao}, seed={seed}')
                        trainer = LeakageLocalizationTrainer(
                            self.profiling_dataset, self.attack_dataset,
                            default_training_module_kwargs=self.trial_config['default_kwargs']
                        )
                        leakage_localization_kwargs = copy(self.trial_config['default_kwargs'])
                        leakage_localization_kwargs.update(self.trial_config['leakage_localization_kwargs'])
                        leakage_localization_kwargs.update({'supervised_dnn': supervised_dnn})
                        leakage_localization_kwargs.update(self.ll_optimal_hparams)
                        leakage_localization_kwargs.update({'starting_prob': gammao})
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
                    assessments[seed, gammao_idx, :] = leakage_assessment.squeeze()
            np.save(os.path.join(self.leakage_localization_dir, 'gammao_sweep.npy'), assessments)
        else:
            assessments = np.load(os.path.join(self.leakage_localization_dir, 'gammao_sweep.npy'))
        assert np.all(np.isfinite(assessments))
        ground_truth_assessments = self.get_ground_truth_assessments()
        ground_truth_assessment = np.stack(list(ground_truth_assessments.values())).mean(axis=0)
        spearmanr_evaluations = np.full((self.seed_count, len(GAMMAO_VALS)), np.nan, dtype=np.float32)
        for seed in range(self.seed_count):
            for gammao_idx, _ in enumerate(GAMMAO_VALS):
                corr = spearmanr(assessments[seed, gammao_idx, :], ground_truth_assessment.squeeze()).statistic
                spearmanr_evaluations[seed, gammao_idx] = corr
        assert np.all(np.isfinite(spearmanr_evaluations))
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        ax.fill_between(GAMMAO_VALS, spearmanr_evaluations.min(axis=0), spearmanr_evaluations.max(axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
        ax.plot(GAMMAO_VALS, np.median(spearmanr_evaluations, axis=0), color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
        ax.set_xlabel(r'Budget: $\overline{\gamma}$')
        ax.set_ylabel(r'oSNR of ALL with budget $\overline{\gamma}$')
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'gammao_performance_sweep.pdf'), **SAVEFIG_KWARGS)
        fig.savefig(os.path.join(self.logging_dir, 'gammao_performance_sweep.png'), **SAVEFIG_KWARGS)

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
    
    def run_timing_trials(self):
        if os.path.exists(os.path.join(self.logging_dir, 'timing_info.npz')):
            timing_info = np.load(os.path.join(self.logging_dir, 'timing_info.npz'), allow_pickle=True)
        else:
            timing_info = {}
            
            data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
            profiling_dataloader = data_module.train_dataloader()
            model_dir = os.path.join(self.supervised_model_dir, f'seed={0}')
            nn_attributor = NeuralNetAttribution(profiling_dataloader, model_dir, seed=0)
            nn_attr_timing = nn_attributor.time_things()
            timing_info.update(nn_attr_timing)

            supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=self.trial_config['supervised_training_kwargs'])
            supervised_runtime = np.stack([
                supervised_trainer.time_run(test_steps=100)
                for _ in range(5)
            ])
            sup_per_trial_runtime = supervised_runtime.sum(axis=1)*self.trial_config['max_classifiers_pretrain_steps']/100
            timing_info['supervised_training'] = 1e-3*sup_per_trial_runtime/60 # min

            kwargs = self.trial_config['default_kwargs']
            kwargs.update(self.trial_config['leakage_localization_kwargs'])
            ll_trainer = LeakageLocalizationTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs)
            ll_runtime = np.stack([
                ll_trainer.time_run(test_steps=100)
                for _ in range(5)
            ])
            all_per_trial_runtime = ll_runtime.sum(axis=1)*(
                self.trial_config['max_leakage_localization_steps']
                + (self.trial_config['max_classifiers_pretrain_steps'] if self.dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'aes_hd'] else 0)
            )/100
            timing_info['all'] = 1e-3*all_per_trial_runtime/60

            np.savez(os.path.join(self.logging_dir, 'timing_info.npz'), **timing_info)
        print('Timing info:')
        for key, val in timing_info.items():
            print(f'{key}: {val.mean()} +/- {val.std()} minutes')
    
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
        
    def occlusion_window_sweep(self):
        window_sizes = np.arange(1, 51, 2)
        if not os.path.exists(os.path.join(self.nn_attr_dir, 'occl_window_sweep.npy')):
            print('Running occlusion window size sweep')
            data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
            profiling_dataloader = data_module.train_dataloader()
            results = np.full((len(window_sizes), self.seed_count, self.profiling_dataset.data_shape[-1]), np.nan, dtype=np.float32)
            progress_bar = tqdm(total=self.seed_count*len(window_sizes))
            for seed in range(self.seed_count):
                model_dir = os.path.join(self.supervised_model_dir, f'seed={seed}')
                nn_attributor = NeuralNetAttribution(profiling_dataloader, model_dir, seed=seed)
                for window_idx, window_size in enumerate(window_sizes):
                    leakage_assessment = nn_attributor.compute_n_occlusion(window_size)
                    results[window_idx, seed, :] = leakage_assessment
                    progress_bar.update(1)
            assert np.all(np.isfinite(results))
            np.save(os.path.join(self.nn_attr_dir, 'occl_window_sweep.npy'), results)
        else:
            print('Found precomputed occlusion window size sweep')
            results = np.load(os.path.join(self.nn_attr_dir, 'occl_window_sweep.npy'))
        ground_truth_assessments = self.get_ground_truth_assessments()
        ground_truth_assessment = np.stack(list(ground_truth_assessments.values())).mean(axis=0)
        spearmanr_evaluations = np.full(results.shape[:-1], np.nan, dtype=np.float32)
        for window_idx in range(results.shape[0]):
            for seed_idx in range(results.shape[1]):
                corr = spearmanr(results[window_idx, seed_idx, :].squeeze(), ground_truth_assessment.squeeze()).statistic
                spearmanr_evaluations[window_idx, seed_idx] = corr
        assert np.all(np.isfinite(spearmanr_evaluations))
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        ax.fill_between(window_sizes, spearmanr_evaluations.min(axis=1), spearmanr_evaluations.max(axis=1), color='blue', alpha=0.25, **PLOT_KWARGS)
        ax.plot(window_sizes, np.median(spearmanr_evaluations, axis=1), color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
        ax.set_xlabel(r'Window size: $m$')
        ax.set_ylabel(r'oSNR for $m$-occlusion')
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'occl_window_size_performance_sweep.pdf'), **SAVEFIG_KWARGS)
        assert hasattr(self, 'nn_attr_assessments')
        best_window_idx = np.argmax(spearmanr_evaluations.mean(axis=1))
        self.best_occlusion_window_size = window_sizes[best_window_idx]
        print(f'Best occlusion window size: {self.best_occlusion_window_size}')
        self.nn_attr_assessments['m_occlusion'] = results[best_window_idx, :, :]
        if not os.path.exists(os.path.join(self.nn_attr_dir, '2o_m_occl.npy')):
            print(f'Computing 2nd-order {self.best_occlusion_window_size}-occlusion...')
            data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
            profiling_dataloader = data_module.train_dataloader()
            results = []
            for seed in range(self.seed_count):
                model_dir = os.path.join(self.supervised_model_dir, f'seed={seed}')
                nn_attributor = NeuralNetAttribution(profiling_dataloader, model_dir, seed=seed)
                leakage_assessment = nn_attributor.compute_second_order_occlusion(window_size=10)#self.best_occlusion_window_size)
                results.append(leakage_assessment.squeeze())
            occl2o = np.stack(results)
            np.save(os.path.join(self.nn_attr_dir, '2o_m_occl.npy'), occl2o)
        else:
            occl2o = np.load(os.path.join(self.nn_attr_dir, '2o_m_occl.npy'))
        plot_leakage_assessment(occl2o.mean(axis=0), os.path.join(self.nn_attr_dir, 'sec_order_m_occl.png'))
        self.nn_attr_assessments['second_order_m_occl'] = occl2o

    def compute_neural_net_attributions(self, wouters_zaid_model=None):
        data_module = DataModule(self.profiling_dataset, self.attack_dataset, val_prop=0.0)
        profiling_dataloader = data_module.train_dataloader()
        attack_dataloader = data_module.test_dataloader()
        to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
        gradviss, saliencies, inputxgrads, lrps, occpois, occl2os, ext_occpois = [], [], [], [], [], [], []
        for occl_n in [1]:
            setattr(self, to_name(f'occl_{occl_n}'), [])
        for seed in range(self.seed_count):
            subdir = os.path.join(self.nn_attr_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            model_dir = os.path.join(self.supervised_model_dir, f'seed={seed}') if wouters_zaid_model is None else wouters_zaid_model
            nn_attributor = NeuralNetAttribution(profiling_dataloader, model_dir, seed=seed)
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
            if not os.path.exists(os.path.join(subdir, to_name('inputxgrad.npy'))):
                print('Computing inputxgrad...')
                inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
                np.save(os.path.join(subdir, to_name('inputxgrad.npy')), inputxgrad)
                print('\tDone.')
            else:
                inputxgrad = np.load(os.path.join(subdir, to_name('inputxgrad.npy')))
                print('Found precomputed inputxgrad.')
            if not os.path.exists(os.path.join(subdir, to_name('second_order_occl.npy'))):
                print('Computing second-order occlusion...')
                occl2o = nn_attributor.compute_second_order_occlusion().reshape(-1)
                np.save(os.path.join(subdir, to_name('second_order_occl.npy')), occl2o)
                print('\tDone.')
            else:
                occl2o = np.load(os.path.join(subdir, to_name('second_order_occl.npy')))
                print('Found precomputed second-order occlusion.')
            if not os.path.exists(os.path.join(subdir, to_name('occpoi.npy'))):
                print('Computing OccPOI...')
                occpoi = OccPOI(attack_dataloader=attack_dataloader, model=model_dir, seed=seed, dataset_name=self.dataset_name)()
                np.save(os.path.join(subdir, to_name('occpoi.npy')), occpoi)
            else:
                occpoi = np.load(os.path.join(subdir, to_name('occpoi.npy')))
                print('Found precomputed OccPOI.')
            if not(os.path.exists(os.path.join(subdir, to_name('extended_occpoi.npy')))):
                print('Computing extended OccPOI...')
                ext_occpoi = OccPOI(attack_dataloader=attack_dataloader, model=model_dir, seed=seed, dataset_name=self.dataset_name)(extended=True)
                np.save(os.path.join(subdir, to_name('extended_occpoi.npy')), ext_occpoi)
            else:
                ext_occpoi = np.load(os.path.join(subdir, to_name('extended_occpoi.npy')))
            for occl_n in [1]:
                if not os.path.exists(os.path.join(subdir, to_name(f'{occl_n}_occl.npy'))):
                    print(f'Computing {occl_n}-occlusion...')
                    n_occl = nn_attributor.compute_n_occlusion(occl_n).reshape(-1)
                    np.save(os.path.join(subdir, to_name(f'{occl_n}_occl.npy')), n_occl)
                else:
                    n_occl = np.load(os.path.join(subdir, to_name(f'{occl_n}_occl.npy')))
                    print(f'Found precomputed {occl_n}-occlusion')
                plot_leakage_assessment(n_occl, os.path.join(subdir, to_name(f'{occl_n}_occl.png')))
                vals = getattr(self, to_name(f'occl_{occl_n}'))
                vals.append(n_occl)
            plot_leakage_assessment(gradvis, os.path.join(subdir, to_name('gradvis.png')))
            plot_leakage_assessment(saliency, os.path.join(subdir, to_name('saliency.png')))
            plot_leakage_assessment(inputxgrad, os.path.join(subdir, to_name('inputxgrad.png')))
            plot_leakage_assessment(occl2o, os.path.join(subdir, to_name('second_order_occl.png')))
            plot_leakage_assessment(ext_occpoi, os.path.join(subdir, to_name('ext_occpoi.png')))
            ext_occpois.append(ext_occpoi)
            occpois.append(occpoi)
            plot_leakage_assessment(occpoi, os.path.join(subdir, to_name('occpoi.png')))
            if wouters_zaid_model is None:
                plot_leakage_assessment(lrp, os.path.join(subdir, to_name('lrp.png')))
                lrps.append(lrp)
            gradviss.append(gradvis)
            saliencies.append(saliency)
            inputxgrads.append(inputxgrad)
            occl2os.append(occl2o)
        setattr(self, to_name('nn_attr_assessments'), {
            to_name('gradvis'): np.stack(gradviss), to_name('saliency'): np.stack(saliencies), to_name('inputxgrad'): np.stack(inputxgrads),
            to_name('second_order_occlusion'): np.stack(occl2os),
            to_name('ext_occpoi'): np.stack(ext_occpois),
            to_name('occpoi'): np.stack(occpois),
            **({to_name('lrp'): np.stack(lrps)} if wouters_zaid_model is None else {})
        })
        if os.path.exists(os.path.join(self.logging_dir, 'occpoi_reported_result.npy')):
            occpoi_indices = np.load(os.path.join(self.logging_dir, 'occpoi_reported_result.npy'))
            leakage_assessment = np.zeros(self.profiling_dataset.data_shape, dtype=np.float32).squeeze()
            leakage_assessment[..., occpoi_indices] = 1
            self.nn_attr_assessments['occpoi_reported'] = leakage_assessment
        val = getattr(self, to_name('nn_attr_assessments'))
        for occl_n in [1]:
            _val = getattr(self, to_name(f'occl_{occl_n}'))
            val[to_name(f'occl_{occl_n}')] = np.stack(_val)
    
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
    
    def create_main_paper_dnn_auc_plots(self):
        forward_results = defaultdict(list)
        reverse_results = defaultdict(list)
        datamodule = DataModule(self.profiling_dataset, self.attack_dataset, eval_batch_size=len(self.attack_dataset))
        attack_dataloader = datamodule.test_dataloader()
        progress_bar = tqdm(total=self.seed_count*len(self.get_leakage_assessments()))
        for seed in range(self.seed_count):
            training_module = SupervisedModule.load_from_checkpoint(os.path.join(self.supervised_model_dir, f'seed={(seed+1)%5}', 'best_checkpoint.ckpt'))
            supervised_dnn = training_module.classifier
            for leakage_assessment_name, leakage_assessment in self.get_leakage_assessments().items():
                forward_result_name = f'{leakage_assessment_name}_forward_seed={seed}.npy'
                reverse_result_name = f'{leakage_assessment_name}_reverse_seed={seed}.npy'
                if not(os.path.exists(os.path.join(self.dnn_auc_dir, forward_result_name)) and os.path.exists(os.path.join(self.dnn_auc_dir, reverse_result_name))):
                    if leakage_assessment.ndim > 1:
                        assert leakage_assessment.shape[-1] >= self.seed_count
                        _leakage_assessment = leakage_assessment[seed, :]
                    else:
                        _leakage_assessment = leakage_assessment
                    out = compute_dnn_performance_auc(attack_dataloader, supervised_dnn, _leakage_assessment, 'cuda', average=False, cluster_count=None)
                    np.save(os.path.join(self.dnn_auc_dir, forward_result_name), out['forward_dnn_auc'])
                    np.save(os.path.join(self.dnn_auc_dir, reverse_result_name), out['reverse_dnn_auc'])
                forward_result = np.load(os.path.join(self.dnn_auc_dir, forward_result_name))
                reverse_result = np.load(os.path.join(self.dnn_auc_dir, reverse_result_name))
                forward_results[leakage_assessment_name].append(forward_result)
                reverse_results[leakage_assessment_name].append(reverse_result)
                progress_bar.update(1)
        forward_results = {key: np.stack(val) for key, val in forward_results.items()}
        reverse_results = {key: np.stack(val) for key, val in reverse_results.items()}
        
        print('Forward ablation test:')
        for key, val in forward_results.items():
            print(f'\t{key}: {np.mean(val)} +/- {np.std(np.mean(val, axis=-1))}')
        print('Reverse ablation test:')
        for key, val in reverse_results.items():
            print(f'\t{key}: {np.mean(val)} +/- {np.std(np.mean(val, axis=-1))}')

        if self.dataset_name == 'ascadv1_variable': # Main paper plot
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
            forward_random_baseline = forward_results['random'].reshape(self.seed_count, -1)
            forward_stat_baseline = forward_results['sosd'].reshape(self.seed_count, -1)
            forward_nn_baseline = forward_results['m_occlusion'].reshape(self.seed_count, -1)
            forward_ll = forward_results['leakage_localization'].reshape(self.seed_count, -1)
            xx = np.arange(forward_random_baseline.shape[-1])
            ax.fill_between(xx, forward_random_baseline.min(axis=0), forward_random_baseline.max(axis=0), color='red', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, forward_stat_baseline.min(axis=0), forward_stat_baseline.max(axis=0), color='green', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, forward_nn_baseline.min(axis=0), forward_nn_baseline.max(axis=0), color='purple', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, forward_ll.min(axis=0), forward_ll.max(axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            ax.plot(xx, np.median(forward_random_baseline, axis=0), color='red', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(forward_stat_baseline, axis=0), color='green', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(forward_nn_baseline, axis=0), color='purple', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(forward_ll, axis=0), color='blue', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.set_xlabel(r'Number of \textit{un-occluded} features')
            ax.set_ylabel('Mean rank on test dataset')
            ax.set_title('Forward DNN occlusion test')
            ax.set_xscale('symlog', linthresh=10)
            fig.tight_layout()
            fig.savefig(os.path.join(self.logging_dir, 'main_paper_forward_dnn_occlusion.png'), **SAVEFIG_KWARGS)
            fig.savefig(os.path.join(self.logging_dir, 'main_paper_forward_dnn_occlusion.pdf'), **SAVEFIG_KWARGS)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
            reverse_random_baseline = reverse_results['random'].reshape(self.seed_count, -1)[:, ::-1]
            reverse_stat_baseline = reverse_results['sosd'].reshape(self.seed_count, -1)[:, ::-1]
            reverse_nn_baseline = reverse_results['m_occlusion'].reshape(self.seed_count, -1)[:, ::-1]
            reverse_ll = reverse_results['leakage_localization'].reshape(self.seed_count, -1)[:, ::-1]
            xx = np.arange(reverse_random_baseline.shape[-1])
            ax.fill_between(xx, reverse_random_baseline.min(axis=0), reverse_random_baseline.max(axis=0), color='red', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, reverse_stat_baseline.min(axis=0), reverse_stat_baseline.max(axis=0), color='green', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, reverse_nn_baseline.min(axis=0), reverse_nn_baseline.max(axis=0), color='purple', alpha=0.25, **PLOT_KWARGS)
            ax.fill_between(xx, reverse_ll.min(axis=0), reverse_ll.max(axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            ax.plot(xx, np.median(reverse_random_baseline, axis=0), color='red', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(reverse_stat_baseline, axis=0), color='green', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(reverse_nn_baseline, axis=0), color='purple', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.plot(xx, np.median(reverse_ll, axis=0), color='blue', marker='.', markersize=1, linestyle='none', **PLOT_KWARGS)
            ax.set_xlabel(r'Number of \textit{occluded} features')
            ax.set_ylabel('Mean rank on test dataset')
            ax.set_title('Reverse DNN occlusion test')
            ax.set_xscale('symlog', linthresh=10)
            fig.tight_layout()
            fig.savefig(os.path.join(self.logging_dir, 'main_paper_reverse_dnn_occlusion.png'), **SAVEFIG_KWARGS)
            fig.savefig(os.path.join(self.logging_dir, 'main_paper_reverse_dnn_occlusion.pdf'), **SAVEFIG_KWARGS)
            plt.close(fig)


        
        # appendix plot
        r"""col_count = 6
        row_count = 2*int(np.ceil(len(forward_results)/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(0.75*col_count*PLOT_WIDTH, 0.75*row_count*PLOT_WIDTH))
        forward_axes = axes[:row_count//2, :].flatten()
        reverse_axes = axes[row_count//2:, :].flatten()
        for idx, assessment_name in enumerate(forward_results.keys()):
            forward_ax = forward_axes[idx]
            reverse_ax = reverse_axes[idx]
            forward_result = forward_results[assessment_name].reshape(self.seed_count, -1)
            reverse_result = reverse_results[assessment_name].reshape(self.seed_count, -1)
            forward_ax.fill_between(range(forward_result.shape[-1]), np.min(forward_result, axis=0), np.max(forward_result, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            forward_ax.plot(np.median(forward_result, axis=0), color='blue', **PLOT_KWARGS)
            reverse_ax.fill_between(range(reverse_result.shape[-1]), np.min(reverse_result, axis=0), np.max(reverse_result, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            reverse_ax.plot(np.median(reverse_result, axis=0), color='blue', **PLOT_KWARGS)
            forward_ax.set_xlabel('Number of un-occluded inputs')
            forward_ax.set_ylabel('Mean rank on test dataset')
            reverse_ax.set_xlabel('Number of un-occluded inputs')
            reverse_ax.set_ylabel('Mean rank on test dataset')
            forward_ax.set_title(f'Forward DNN occlusion test:\n{get_assessment_name(assessment_name)}')
            reverse_ax.set_title(f'Reverse DNN occlusion test:\n{get_assessment_name(assessment_name)}')
        for ax in forward_axes[len(forward_results):]:
            ax.axis('off')
        for ax in reverse_axes[len(reverse_results):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.dnn_auc_dir, 'full_dnn_occlusion.png'), **SAVEFIG_KWARGS)"""
    
    def create_main_paper_leakage_assessment_plots(self):
        leakage_assessments = self.get_leakage_assessments()['leakage_localization'].reshape(self.seed_count, -1)
        random_assessment = self.get_leakage_assessments()['random'].reshape(self.seed_count, -1)
        if self.dataset_name != 'ascadv1_variable':
            return
        stat_baseline_assessment = self.get_leakage_assessments()['sosd'].reshape(1, -1)
        nn_attr_baseline_assessment = self.get_leakage_assessments()['m_occlusion'].reshape(self.seed_count, -1)
        stat_baseline_name = 'SoSD'
        nn_attr_baseline_name = '7-Occlusion'
        r"""if self.dataset_name == 'ascadv1_fixed':
            stat_baseline_assessment = self.get_leakage_assessments()['cpa'].reshape(1, -1)
            nn_attr_baseline_assessment = self.get_leakage_assessments()['occlusion'].reshape(self.seed_count, -1)
            stat_baseline_name = 'CPA'
            nn_attr_baseline_name = '1-occlusion'
        else:
            stat_baseline_assessment = self.get_leakage_assessments()['sosd'].reshape(1, -1)
            nn_attr_baseline_assessment = self.get_leakage_assessments()['gradvis'].reshape(self.seed_count, -1)
            stat_baseline_name = 'SoSD'
            nn_attr_baseline_name = 'GradVis'"""
        ground_truth_assessments = self.get_ground_truth_assessments()
        ground_truth_assessment = np.stack(list(ground_truth_assessments.values())).mean(axis=0).squeeze()
        sorted_indices = ground_truth_assessment.argsort()
        fig, axes = plt.subplots(2, 2, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        for assessment, color, ax, title in zip(
            [random_assessment, stat_baseline_assessment, nn_attr_baseline_assessment, leakage_assessments],
            ['red', 'green', 'purple', 'blue'], axes.flatten(), ['Random', 'SoSD', '7-Occlusion', 'ALL (Ours)']
        ):
            assessment = np.abs(assessment[:, sorted_indices])
            if assessment.shape[0] > 1:
                ax.fill_between(ground_truth_assessment[sorted_indices], assessment.min(axis=0), assessment.max(axis=0), color=color, alpha=0.25, **PLOT_KWARGS)
            ax.plot(ground_truth_assessment[sorted_indices], np.median(assessment, axis=0), color=color, linestyle='none', marker='.', markersize=1, **PLOT_KWARGS)
            ax.set_xscale('log')
            if title in ['SoSD', '7-Occlusion']:
                ax.set_yscale('log')
            ax.set_title(title)
        fig.text(0.5, 0.04, 'Leakage of $X_t$ according to `Omniscient\' SNR', ha='center')
        fig.text(0.04, 0.5, 'Estimated leakage of $X_t$', va='center', rotation='vertical')
        fig.tight_layout(rect=[0.05, 0.05, 1, 1])
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_leakage_assessments.png'), **SAVEFIG_KWARGS)
        fig.savefig(os.path.join(self.logging_dir, 'main_paper_leakage_assessments.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def create_appendix_leakage_assessment_plots(self):
        leakage_assessments = self.get_leakage_assessments()
        ground_truth_assessment = self.get_ground_truth_assessments()['template__window_size=5']
        ground_truth_assessment = np.mean(np.stack(list(ground_truth_assessment.values())), axis=0)[0, :]
        col_count = 6
        row_count = int(2*np.ceil((len(leakage_assessments))/col_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(0.75*col_count*PLOT_WIDTH, 0.75*row_count*PLOT_WIDTH))
        comparison_axes = axes[row_count//2:, :]
        assessment_axes = axes[:row_count//2, :]
        for (assessment_name, assessment), comparison_ax, assessment_ax in zip(leakage_assessments.items(), comparison_axes.flatten(), assessment_axes.flatten()):
            assessment = np.abs(assessment.reshape(-1, assessment.shape[-1]))
            if assessment_name != 'ground_truth':
                averaged_assessment = np.stack([
                    torch.tensor(_assessment).unfold(0, 5, 1).numpy().mean(axis=-1)
                    for _assessment in assessment
                ])
            else:
                assessment = assessment.reshape(1, -1)
                averaged_assessment = assessment
            comparison_ax.errorbar(
                ground_truth_assessment, np.median(averaged_assessment, axis=0),
                yerr=(np.median(averaged_assessment, axis=0)-np.min(averaged_assessment, axis=0), np.max(averaged_assessment, axis=0)-np.median(averaged_assessment, axis=0)),
                color='blue', fmt='.', markersize=3, elinewidth=0.5, capsize=2, linestyle='none', **PLOT_KWARGS
            )
            assessment_ax.fill_between(np.arange(assessment.shape[-1]), np.min(assessment, axis=0), np.max(assessment, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
            assessment_ax.plot(np.median(assessment, axis=0), color='blue', linewidth=0.25, **PLOT_KWARGS)
            comparison_ax.set_xscale('log')
            comparison_ax.set_yscale('log')
            assessment_ax.set_yscale('log')
            comparison_ax.set_xlabel(r'Ground truth-like leakage of $X_t$')
            comparison_ax.set_ylabel(r'Estimated leakage of $X_t$')
            assessment_ax.set_xlabel(r'Timestep $t$')
            assessment_ax.set_ylabel(r'Estimated leakage of $X_t$')
            assessment_ax.set_title(f'{get_assessment_name(assessment_name)}')
            comparison_ax.set_title(f'{get_assessment_name(assessment_name)}')
        for ax in comparison_axes.flatten()[len(leakage_assessments):]:
            ax.axis('off')
        for ax in assessment_axes.flatten()[len(leakage_assessments):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'appendix_leakage_assessment_plots.png'), **SAVEFIG_KWARGS)
    
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
        #self.run_timing_trials()
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
            self.occlusion_window_sweep()
        if ('run_ll_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_hparam_sweep']:
            self.run_ll_hparam_sweep()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_leakage_localization()
            self.run_ll_gammao_sweep()
        if True:
            if ('run_ll_classifiers_hparam_sweep' in self.trial_config) and self.trial_config['run_ll_classifiers_hparam_sweep']:
                self.run_ll_classifiers_hparam_sweep()
            if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers']:
                self.pretrain_leakage_localization_classifiers()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_ll_ablation_study()
        self.create_main_paper_dnn_auc_plots()
        self.eval_leakage_assessments()
        self.plot_leakage_assessments()
        self.create_main_paper_leakage_assessment_plots()
        self.create_appendix_leakage_assessment_plots()
        #self.compute_dnn_auc_vals_on_baselines()