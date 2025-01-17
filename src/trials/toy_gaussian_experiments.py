from typing import *
import os
from copy import copy
import numpy as np
from torch.utils.data import DataLoader

from datasets.simple_gaussian import SimpleGaussianDataset
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
from training_modules.supervised_deep_sca import SupervisedTrainer
from utils.baseline_assessments import NeuralNetAttribution, FirstOrderStatistics

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        seed_count: int = 1,
        trial_count: int = 21,
        run_baselines: bool = True
    ):
        self.logging_dir = logging_dir
        self.seed_count = seed_count
        self.trial_count = trial_count
        self.run_kwargs = {'max_steps': 1000, 'anim_gammas': False}
        self.supervised_kwargs = {'classifier_name': 'mlp-1d', 'classifier_kwargs': {'use_dropout': False, 'layer_count': 1}, 'lr': 1e-3}
        self.leakage_localization_kwargs = {
            'classifiers_name': 'mlp-1d', 'classifiers_kwargs': {'use_dropout': False, 'layer_count': 1}, 'theta_lr': 1e-3, 'etat_lr': 1e-3, 'adversarial_mode': False, 'ent_penalty': 1e-2
        }
        self.run_baselines = run_baselines
    
    def run_experiments(self, logging_dir, dataset_kwargss, budgets, run_baselines: bool = True):
        assert len(budgets) == len(dataset_kwargss)
        leakage_assessments = {}
        for (trial_name, dataset_kwargs), budget in zip(dataset_kwargss, budgets):
            leakage_assessments[trial_name] = {}
            profiling_dataset = SimpleGaussianDataset(**dataset_kwargs)
            attack_dataset = SimpleGaussianDataset(**dataset_kwargs)
            if self.run_baselines:
                first_order_stats = FirstOrderStatistics(profiling_dataset)
                leakage_assessments['snr'] = first_order_stats.snr_vals['label'].reshape(-1)
                leakage_assessments['sosd'] = first_order_stats.sosd_vals['label'].reshape(-1)
                leakage_assessments['cpa'] = first_order_stats.cpa_vals['label'].reshape(-1)
                sup_trainer = SupervisedTrainer(
                    profiling_dataset, attack_dataset,
                    default_data_module_kwargs={'train_batch_size': len(profiling_dataset)},
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
                default_data_module_kwargs={'train_batch_size': len(profiling_dataset)},
                default_training_module_kwargs={**self.leakage_localization_kwargs, 'budget': budget}
            )
            ll_trainer.pretrain_classifiers(os.path.join(logging_dir, trial_name, 'pretrain_classifiers'), max_steps=self.run_kwargs['max_steps'])
            ll_trainer = LeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                default_data_module_kwargs={'train_batch_size': len(profiling_dataset)},
                default_training_module_kwargs={**self.leakage_localization_kwargs, 'budget': budget}
            )
            ll_leakage_assessment = ll_trainer.run(
                os.path.join(logging_dir, trial_name, 'leakage_localization'),
                pretrained_classifiers_logging_dir=os.path.join(logging_dir, trial_name, 'pretrain_classifiers'),
                **self.run_kwargs
            )
            leakage_assessments[trial_name]['leakage_localization'] = ll_leakage_assessment
        return leakage_assessments
    
    def run_1o_count_sweep(self, budgets: Union[float, Sequence[float]] = 1.0, hparam_test: bool = False):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'count={x}', {'no_hard_feature': True, 'easy_feature_count': x}) for x in range(1, self.trial_count+1)
        ][::-1]
        if hparam_test:
            dataset_kwargss = [('test', {'no_hard_feature': True, 'easy_feature_count': 128})]
            budgets = [budgets[-1]]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_count_sweep', f'seed={seed}')
            leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss, budgets=budgets)
            np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
        if hparam_test:
            return leakage_assessments['test']['leakage_localization']
    
    def run_1o_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0, hparam_test: bool = False):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'no_hard_feature': True, 'random_feature_count': 0, 'easy_feature_count': 2, 'easy_feature_snrs': [1.0, x]})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ]
        if hparam_test:
            dataset_kwargss = [('test', {'no_hard_feature': True, 'random_feature_count': 0, 'easy_feature_count': 2, 'easy_feature_snrs': [1.0, 0.1]})]
            budgets = [budgets[self.trial_count//2]]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, '1o_var_sweep', f'seed={seed}')
            leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss, budgets=budgets)
            np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def run_xor_var_sweep(self, budgets: Union[float, Sequence[float]] = 1.0, hparam_test: bool = False):
        if not hasattr(budgets, '__len__'):
            budgets = self.trial_count*[budgets]
        dataset_kwargss = [
            (f'var={x}', {'easy_feature_snrs': x})
            for x in [0.5**n for n in range(1, self.trial_count//2+1)][::-1] + [1.0] + [2.0**n for n in range(1, self.trial_count//2+1)]
        ][::-1]
        if hparam_test:
            dataset_kwargss = [('test', {'easy_feature_snrs': 0.5})]
            budgets = [budgets[self.trial_count//2]]
        for seed in range(self.seed_count):
            logging_dir = os.path.join(self.logging_dir, 'xor_var_sweep', f'seed={seed}')
            leakage_assessments = self.run_experiments(logging_dir, dataset_kwargss, budgets=budgets)
            np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), leakage_assessments=leakage_assessments)
    
    def __call__(self):
        r"""base_dir = r'/home/jgammell/Desktop/mi_leakage_localization/outputs/toy_gaussian/hparam_sweep'
        subdirs = os.listdir(base_dir)
        for subdir in subdirs:
            if not os.path.exists(os.path.join(base_dir, subdir, '1o_count_sweep', 'leakage_assessments.npz')):
                continue
            leakage_assessment = np.load(os.path.join(base_dir, subdir, '1o_count_sweep', 'leakage_assessments.npz'), allow_pickle=True)['leakage_assessments'].item()['test']['leakage_localization']
            min_ratio = leakage_assessment[0] / np.min(leakage_assessment[1:])
            mean_ratio = leakage_assessment[0] / np.mean(leakage_assessment[1:])
            print(f'subdir={subdir}, min_ratio={min_ratio}, mean_ratio={mean_ratio}')"""
        self.run_1o_count_sweep(1.0)
        self.run_1o_var_sweep(1.0)
        self.run_xor_var_sweep(1.0)
    
    def hparam_sweep(self):
        orig_logging_dir = self.logging_dir
        run_baselines = self.run_baselines
        self.run_baselines = False
        performance = []
        for trial_idx in range(50):
            budget = 10**np.random.uniform(-2, 2)
            ent_penalty = 10**np.random.uniform(-2, 2)
            self.logging_dir = os.path.join(orig_logging_dir, 'hparam_sweep', f'budget={budget}__ent_penalty={ent_penalty}')
            self.leakage_localization_kwargs['ent_penalty'] = ent_penalty
            os.makedirs(self.logging_dir, exist_ok=True)
            leakage_assessment = self.run_1o_count_sweep(budget, hparam_test=True)
            performance.append((budget, ent_penalty, leakage_assessment))
            #self.run_1o_var_sweep(budget, hparam_test=True)
            #self.run_xor_var_sweep(budget, hparam_test=True)
        performance.sort(key=lambda x: x[0])
        performance.sort(key=lambda x: x[1])
        for (budget, ent_penalty, leakage_assessment) in performance:
            print(f'budget={budget}, ent_penalty={ent_penalty}, mean_ratio={leakage_assessment[0]/np.mean(leakage_assessment[1:])}, min_ratio={leakage_assessment[0]/np.min(leakage_assessment[1:])}')
        self.logging_dir = orig_logging_dir
        self.run_baselines = run_baselines