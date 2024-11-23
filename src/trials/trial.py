import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from .utils import *
from .localization_assessment import evaluate_template_attack_exploitability
from training_modules.supervised_classification import SupervisedClassificationModule
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer as ALLTrainer
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd

LEAKAGE_ASSESSMENT_TECHNIQUES = ['random', 'cpa', 'snr', 'sosd']

class Trial:
    def __init__(self,
        base_dir,
        profiling_dataset,
        attack_dataset,
        seed_count=5,
        template_attack_poi_count=20
    ):
        self.base_dir = base_dir
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.seed_count = seed_count
        self.template_attack_poi_count = template_attack_poi_count
        
        os.makedirs(self.base_dir, exist_ok=True)
    
    def __call__(self):
        self.compute_random_baseline()
        self.compute_first_order_baselines()
        self.eval_leakage_assessments()
        self.plot_everything()
    
    def save_leakage_assessment(self, name, leakage_assessment):
        np.save(os.path.join(self.base_dir, f'{name}_leakage_assessment.npy'), leakage_assessment)
    
    def load_leakage_assessment(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_leakage_assessment.npy')):
            return None
        leakage_assessment = np.load(os.path.join(self.base_dir, f'{name}_leakage_assessment.npy'))
        return leakage_assessment
    
    def compute_random_baseline(self):
        if self.load_leakage_assessment('random') is None:
            random_leakage_assessment = np.random.randn(self.seed_count, self.profiling_dataset.timesteps_per_trace)
            self.save_leakage_assessment('random', random_leakage_assessment)
    
    def compute_first_order_baselines(self):
        self.profiling_dataset.return_metadata = self.attack_dataset.return_metadata = True
        if self.load_leakage_assessment('cpa') is None:
            cpa_leakage_assessment = calculate_cpa(self.profiling_dataset, self.profiling_dataset, 'label')[('label', None)].squeeze()
            self.save_leakage_assessment('cpa', cpa_leakage_assessment[np.newaxis, :])
        if self.load_leakage_assessment('snr') is None:
            snr_leakage_assessment = calculate_snr(self.profiling_dataset, self.profiling_dataset, 'label')[('label', None)].squeeze()
            self.save_leakage_assessment('snr', snr_leakage_assessment[np.newaxis, :])
        if self.load_leakage_assessment('sosd') is None:
            sosd_leakage_assessment = calculate_sosd(self.profiling_dataset, self.profiling_dataset, 'label')[('label', None)].squeeze()
            self.save_leakage_assessment('sosd', sosd_leakage_assessment[np.newaxis, :])
        self.profiling_dataset.return_metadata = self.attack_dataset.return_metadata = False
    
    def _plot_leakage_assessment(self, leakage_assessment, ax, **plot_kwargs):
        ax.plot(leakage_assessment[0, ...].squeeze(), marker='.', markersize=1, linestyle='none', **plot_kwargs)
    
    def plot_leakage_assessment(self, name):
        leakage_assessment = self.load_leakage_assessment(name)
        if leakage_assessment is None:
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        self._plot_leakage_assessment(leakage_assessment, ax, color='blue')
        fig.savefig(os.path.join(self.base_dir, f'{name}_leakage_assessment.png'))
    
    def _plot_ta_exploitability(self, ta_exploitability, ax, **plot_kwargs):
        ta_exploitability = ta_exploitability.reshape((-1, ta_exploitability.shape[-1]))
        traces_seen = np.arange(1, ta_exploitability.shape[-1]+1)
        ax.plot(traces_seen, np.median(ta_exploitability, axis=0), linestyle='-', **plot_kwargs)
        ax.fill_between(traces_seen, np.percentile(ta_exploitability, 25, axis=0), np.percentile(ta_exploitability, 75, axis=0), alpha=0.25, **plot_kwargs)
        
    def plot_ta_exploitability(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_eval.npz')):
            return
        technique_eval = np.load(os.path.join(self.base_dir, f'{name}_eval.npz'))
        if not 'ta_exploitability' in technique_eval.keys():
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        self._plot_ta_exploitability(technique_eval['ta_exploitability'], ax, color='blue')
        fig.savefig(os.path.join(self.base_dir, f'{name}_ta_exploitability.png'))
    
    def load_leakage_assessments(self):
        leakage_assessments = {}
        for name in LEAKAGE_ASSESSMENT_TECHNIQUES:
            leakage_assessment = self.load_leakage_assessment(name)
            if leakage_assessment is not None:
                leakage_assessments[name] = leakage_assessment
        return leakage_assessments
    
    def eval_leakage_assessments(self, template_attack=True):
        leakage_assessments = self.load_leakage_assessments()
        for technique_name, leakage_assessment in leakage_assessments.items():
            if not os.path.exists(os.path.join(self.base_dir, f'{technique_name}_eval.npz')):
                technique_eval = {}
            else:
                technique_eval = np.load(os.path.join(self.base_dir, f'{technique_name}_eval.npz'))
            if template_attack and not('ta_exploitability' in technique_eval.keys()):
                technique_eval['ta_exploitability'] = np.stack([
                    evaluate_template_attack_exploitability(self.profiling_dataset, self.attack_dataset, _leakage_assessment, poi_count=self.template_attack_poi_count)
                    for _leakage_assessment in leakage_assessment
                ])
            np.savez(os.path.join(self.base_dir, f'{technique_name}_eval.npz'), **technique_eval)
    
    def plot_everything(self):
        for technique_name in LEAKAGE_ASSESSMENT_TECHNIQUES:
            self.plot_leakage_assessment(technique_name)
            self.plot_ta_exploitability(technique_name)