from typing import *
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset

from common import *
from .gmm_sequence import *
from datasets.simple_gaussian import SimpleGaussianDataset
from datasets.data_module import DataModule
from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer

class Trial:
    def __init__(self,
        base_dir: Union[str, os.PathLike]
    ):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def numerical_experiments(self):
        output_dir = os.path.join(self.base_dir, 'numerical_experiments')
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, 'conditional_mi_sweep.pickle')):
            with open(os.path.join(output_dir, 'conditional_mi_sweep.pickle'), 'rb') as f:
                results = pickle.load(f)
        else:
            results = sweep_conditional_mi()
            with open(os.path.join(output_dir, 'conditional_mi_sweep.pickle'), 'wb') as f:
                pickle.dump(results, f)
        fig = plot_conditional_mi(results)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'conditional_mi_sweep.pdf'))
    
    def sigma_sweep(self):
        output_dir = os.path.join(self.base_dir, 'sigma_sweep')
        os.makedirs(output_dir, exist_ok=True)
        sigma_vals = [10**x for x in np.linspace(-1, 3, 20)]
        for sigma_val in sigma_vals:
            trial_dir = os.path.join(output_dir, f'sigma={sigma_val}')
            os.makedirs(trial_dir, exist_ok=True)
            profiling_dataset = SimpleGaussianDataset(point_counts=[0, 1], infinite_dataset=True, sigma=sigma_val)
            attack_dataset = SimpleGaussianDataset(point_counts=[0, 1], infinite_dataset=False, sigma=sigma_val)
            trainer = AdversarialLeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                max_epochs=500,
                default_training_module_kwargs=dict(
                    classifiers_name='multilayer-perceptron',
                    classifiers_kwargs={'output_classes': 2},
                    timesteps_per_trace=profiling_dataset.timesteps_per_trace,
                    gammap_lr = 1e-2
                )
            )
            trainer.train_gamma(trial_dir, anim_gammas=False)
    
    def leaky_point_count_sweep(self):
        output_dir = os.path.join(self.base_dir, 'leaky_point_sweep')
        os.makedirs(output_dir, exist_ok=True)
        leaky_point_counts = [16]
        for leaky_point_count in leaky_point_counts:
            trial_dir = os.path.join(output_dir, f'leaky_point_count={leaky_point_count}')
            os.makedirs(trial_dir, exist_ok=True)
            profiling_dataset = SimpleGaussianDataset(point_counts=[1, leaky_point_count], infinite_dataset=True, sigma=1.0)
            attack_dataset = SimpleGaussianDataset(point_counts=[1, leaky_point_count], infinite_dataset=False, sigma=1.0)
            trainer = AdversarialLeakageLocalizationTrainer(
                profiling_dataset, attack_dataset,
                max_epochs=500,
                default_training_module_kwargs=dict(
                    classifiers_name='multilayer-perceptron',
                    classifiers_kwargs={'output_classes': 2},
                    timesteps_per_trace=profiling_dataset.timesteps_per_trace,
                    entropy_weight = 1.,
                    gumbel_tau = 1.,
                    occluded_point_count=1
                )
            )
            trainer.train_gamma(trial_dir, anim_gammas=False)
    
    def dataset_size_sweep(self):
        pass
    
    def leakage_order_sweep(self):
        pass