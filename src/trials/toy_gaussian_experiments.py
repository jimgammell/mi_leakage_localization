from typing import *
import os
from collections import defaultdict
from copy import copy
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from common import *
from datasets.simple_gaussian import SimpleGaussianDataset
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
from training_modules.supervised_deep_sca import SupervisedTrainer
from utils.baseline_assessments import NeuralNetAttribution, FirstOrderStatistics

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        seed_count: int = 1,
        trial_count: int = 11,
        run_baselines: bool = True
    ):
        self.logging_dir = logging_dir
        self.seed_count = seed_count
        self.trial_count = trial_count
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.supervised_kwargs = {'classifier_name': 'mlp-1d', 'classifier_kwargs': {'use_dropout': False, 'layer_count': 1}, 'lr': 1e-3}
        self.leakage_localization_kwargs = {
            'classifiers_name': 'mlp-1d', 'classifiers_kwargs': {'use_dropout': False, 'layer_count': 1}, 'theta_lr': 1e-3, 'etat_lr': 1e-3,
            'adversarial_mode': False, 'ent_penalty': 0.0, 'starting_prob': 0.5,
        }
        self.run_baselines = run_baselines
    
    def run_experiments(self, logging_dir, dataset_kwargss, run_baselines: bool = True):
        leakage_assessments = {}
        for trial_name, dataset_kwargs in dataset_kwargss:
            leakage_assessments[trial_name] = {}
            profiling_dataset = SimpleGaussianDataset(**dataset_kwargs)
            attack_dataset = SimpleGaussianDataset(**dataset_kwargs)
            if run_baselines and self.run_baselines:
                first_order_stats = FirstOrderStatistics(profiling_dataset)
                leakage_assessments[trial_name]['snr'] = first_order_stats.snr_vals['label'].reshape(-1)
                leakage_assessments[trial_name]['sosd'] = first_order_stats.sosd_vals['label'].reshape(-1)
                leakage_assessments[trial_name]['cpa'] = first_order_stats.cpa_vals['label'].reshape(-1)
                sup_trainer = SupervisedTrainer(
                    profiling_dataset, attack_dataset,
                    default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
                    default_training_module_kwargs=self.supervised_kwargs
                )
                sup_trainer.run(
                    os.path.join(logging_dir, trial_name, 'supervised'), max_steps=self.run_kwargs['max_steps']
                )
                neural_net_attributor = NeuralNetAttribution(
                    DataLoader(profiling_dataset, batch_size=len(profiling_dataset)), model=os.path.join(logging_dir, trial_name, 'supervised')
                )
                leakage_assessments[trial_name]['gradvis'] = neural_net_attributor.compute_gradvis()
                leakage_assessments[trial_name]['saliency'] = neural_net_attributor.compute_saliency()
                leakage_assessments[trial_name]['lrp'] = neural_net_attributor.compute_lrp()
                leakage_assessments[trial_name]['occlusion'] = neural_net_attributor.compute_occlusion()
                leakage_assessments[trial_name]['inputxgrad'] = neural_net_attributor.compute_inputxgrad()
            ll_trainer = LeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
                default_training_module_kwargs={**self.leakage_localization_kwargs}
            )
            ll_trainer.pretrain_classifiers(os.path.join(logging_dir, trial_name, 'pretrain_classifiers'), max_steps=self.run_kwargs['max_steps']//10)
            ll_trainer = LeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                default_data_module_kwargs={'train_batch_size': len(profiling_dataset)//10},
                default_training_module_kwargs={**self.leakage_localization_kwargs}
            )
            ll_leakage_assessment = ll_trainer.run(
                os.path.join(logging_dir, trial_name, 'leakage_localization'),
                pretrained_classifiers_logging_dir=os.path.join(logging_dir, trial_name, 'pretrain_classifiers'),
                max_steps=9*(self.run_kwargs['max_steps']//10),
                anim_gammas=self.run_kwargs['anim_gammas']
            )
            leakage_assessments[trial_name]['leakage_localization'] = ll_leakage_assessment
        return leakage_assessments
    
    def tune_1o_count_sweep(self):
        dataset_kwargss = [('none', {'no_hard_feature': True, 'easy_feature_count': 101})]
        for starting_prob in [0.001, 0.01, 0.1, 0.5]:
            for ent_penalty in [0.0, 1e-4, 1e-2, 1e0]:
                out = self.run_experiments(
                    os.path.join(self.logging_dir, '1o_count_tune', f'starting_prob={starting_prob}__ent_penalty={ent_penalty}'),
                    dataset_kwargss=dataset_kwargss, run_baselines=False
                )
                assessment = out['none']['leakage_localization'].reshape(-1)
                assessment -= np.min(assessment)
                assessment /= np.max(assessment)
                print(f'prob={starting_prob}, ent_penalty={ent_penalty}, diff={np.min(assessment[1:]) - assessment[0]}')
    
    def run_1o_count_sweep(self):
        dataset_kwargss = [
            (f'count={x}', {'no_hard_feature': True, 'easy_feature_count': x}) for x in [10*x + 1 for x in range(self.trial_count)]
        ][::-1]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
        
    def plot_1o_count_sweep(self):
        fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, 1*PLOT_WIDTH))
        for count in range(1, self.trial_count+1):
            nn_attr_leakage_assessments = {key: [] for key in ['gradvis', 'saliency', 'lrp', 'occlusion', 'inputxgrad']}
            ll_leakage_assessment = []
            for seed in range(self.seed_count):
                leakage_assessment = np.load(os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()
                for key in nn_attr_leakage_assessments.keys():
                    _leakage_assessment = leakage_assessment[f'count={count}'][key].reshape(-1)
                    nn_attr_leakage_assessments[key].append(_leakage_assessment)
                ll_leakage_assessment.append(leakage_assessment[f'count={count}']['leakage_localization'].reshape(-1))
            nn_attr_leakage_assessments = {key: np.stack(val) for key, val in nn_attr_leakage_assessments.items()}
            ll_leakage_assessment = np.stack(ll_leakage_assessment)
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for idx, ((name, assessment), color) in enumerate(zip(nn_attr_leakage_assessments.items(), colors)):
                assessment -= np.min(assessment, axis=-1, keepdims=True)
                assessment /= np.max(assessment, axis=-1, keepdims=True)
                nonleaky_pts = assessment[:, 0]
                leaky_pts = assessment[:, 1:]
                diffs = (leaky_pts - nonleaky_pts.reshape(-1, 1)).reshape(-1)
                axes[1].plot(len(diffs)*[count], diffs, color=color, marker='.', markersize=1, linestyle='none')
                #axes[1].errorbar([count], [np.median(diffs)], [[np.median(diffs)-np.min(diffs)], [np.max(diffs)-np.median(diffs)]], color=color, fmt='.', )
            ll_leakage_assessment -= np.min(ll_leakage_assessment, axis=-1, keepdims=True)
            ll_leakage_assessment /= np.max(ll_leakage_assessment, axis=-1, keepdims=True)
            nonleaky_pts = ll_leakage_assessment[:, 0]
            leaky_pts = ll_leakage_assessment[:, 1:]
            diffs = (leaky_pts - nonleaky_pts.reshape(-1, 1)).reshape(-1)
            axes[2].plot(len(diffs)*[count], diffs, color='blue', marker='.', linestyle='none')
            #axes[2].errorbar([count], [np.median(diffs)], [[np.median(diffs)-np.min(diffs)], [np.max(diffs)-np.median(diffs)]], color='blue', fmt='.')
        for ax in axes:
            ax.set_xlabel('Leaky point count')
            ax.set_ylabel('Estimated leakage')
            ax.set_ylim(-1, 1)
        axes[0].set_title('First-order statistics')
        axes[1].set_title('Neural net attribution')
        axes[2].set_title('Leakage localization')
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, '1o_count_sweep', 'sweep.png'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def run_1o_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'no_hard_feature': True, 'random_feature_count': 0, 'easy_feature_count': 2, 'easy_feature_snrs': [1.0, x]})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_var_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(self.logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss, budgets=budgets)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def run_xor_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'easy_feature_snrs': x})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ][::-1]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, 'xor_var_sweep', f'seed={seed}')
            if not os.path.exists(os.path.join(logging_dir, 'leakage_assessments.npz')):
                leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss, budgets=budgets)
                np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def __call__(self):
        #self.tune_1o_count_sweep()
        self.run_1o_count_sweep()
        #self.plot_1o_count_sweep()
        self.run_1o_var_sweep(1.0)
        self.run_xor_var_sweep(1.0)