import os
from copy import copy, deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from .utils import *
from .localization_assessment import evaluate_template_attack_exploitability, dnn_ablation, evaluate_gmm_exploitability
from training_modules.supervised_classification import SupervisedClassificationModule
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer as ALLTrainer
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from .trial import Trial

LEAKAGE_ASSESSMENT_TECHNIQUES = ['random', 'cpa', 'snr', 'sosd', 'ablation', 'gradvis', 'input_x_grad', 'all']

class PortabilityTrial(Trial):
    def __init__(self,
        base_dir,
        profiling_datasets,
        attack_datasets,
        data_modules,
        epoch_count=100,
        seed_count=1,
        template_attack_poi_count=20,
        default_supervised_classifier_kwargs={},
        default_all_style_classifier_kwargs={}
    ):
        self.base_dir = base_dir
        self.profiling_datasets = profiling_datasets
        self.attack_datasets = attack_datasets
        self.data_modules = data_modules
        self.epoch_count = epoch_count
        self.seed_count = seed_count
        self.template_attack_poi_count = template_attack_poi_count
        self.default_supervised_classifier_kwargs = default_supervised_classifier_kwargs
        self.default_all_style_classifier_kwargs = default_all_style_classifier_kwargs
        self.profile_count = len(self.profiling_datasets)
        self.attack_count = len(self.attack_datasets)
        self.features = self.profiling_datasets[0].timesteps_per_trace
    
        os.makedirs(self.base_dir, exist_ok=True)
    
    def compute_random_baseline(self):
        if self.load_leakage_assessment('random') is None:
            print('Computing random baseline')
            leakage_assessments = np.random.randn(self.profile_count, self.seed_count, self.features)
            self.save_leakage_assessment('random', leakage_assessments)
    
    def compute_first_order_baselines(self):
        for name, f in zip(['cpa', 'snr', 'sosd'], [calculate_cpa, calculate_snr, calculate_sosd]):
            if self.load_leakage_assessment(name) is None:
                print(f'Computing {name} baseline')
                leakage_assessment = np.full((self.profile_count, 1, self.features), np.nan, dtype=np.float32)
                for idx, profiling_dataset in enumerate(self.profiling_datasets):
                    transform = profiling_dataset.transform
                    profiling_dataset.return_metadata = True
                    profiling_dataset.transform = None
                    _leakage_assessment = f(profiling_dataset, profiling_dataset, 'label')[('label', None)].squeeze()
                    leakage_assessment[idx, 0, :] = _leakage_assessment
                    profiling_dataset.return_metadata = False
                    profiling_dataset.transform = transform
                self.save_leakage_assessment(name, leakage_assessment)
    
    def plot_leakage_assessment(self, name):
        leakage_assessment = self.load_leakage_assessment(name)
        if leakage_assessment is None:
            return
        fig, axes = plt.subplots(1, leakage_assessment.shape[0], figsize=(4*leakage_assessment.shape[0], 4))
        for idx, (ax, _leakage_assessment) in enumerate(zip(axes, leakage_assessment)):
            self._plot_leakage_assessment(_leakage_assessment, ax, color='blue')
            ax.set_xlabel('Timestep $t$')
            ax.set_ylabel('Estimated leakage of $X_t$')
            ax.set_title(f'Device {idx}')
        fig.tight_layout()
        fig.savefig(os.path.join(self.base_dir, f'{name}_leakage_assessment.png'))
    
    def eval_leakage_assessments(self, template_attack=True):
        leakage_assessments = self.load_leakage_assessments()
        for technique_name, leakage_assessment in leakage_assessments.items():
            if not os.path.exists(os.path.join(self.base_dir, f'{technique_name}_eval.npz')):
                technique_eval = {}
            else:
                technique_eval = np.load(os.path.join(self.base_dir, f'{technique_name}_eval.npz'))
                technique_eval = dict(technique_eval)
            if template_attack and not('ta_exploitability' in technique_eval.keys()):
                print(f'Calculating template attack exploitability for {technique_name} technique')
                progress_bar = tqdm(total=len(leakage_assessment)*len(self.profiling_datasets))
                result = []
                for _leakage_assessment in leakage_assessment:
                    ta_exploitabilities = []
                    for profiling_dataset, attack_dataset in zip(self.profiling_datasets, self.attack_datasets):
                        ta_exploitability = np.stack([
                            evaluate_template_attack_exploitability(profiling_dataset, attack_dataset, _leakage_assessment, poi_count=self.template_attack_poi_count)
                            for __leakage_assessment in _leakage_assessment
                        ])
                        ta_exploitabilities.append(ta_exploitability)
                        progress_bar.update(1)
                    result.append(np.stack(ta_exploitabilities))
                technique_eval['ta_exploitability'] = np.stack(result)
            print(technique_eval['ta_exploitability'].shape)
            print()
            np.savez(os.path.join(self.base_dir, f'{technique_name}_eval.npz'))
    
    def plot_everything(self):
        for technique_name in LEAKAGE_ASSESSMENT_TECHNIQUES:
            self.plot_leakage_assessment(technique_name)