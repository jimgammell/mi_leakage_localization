import os
from copy import copy, deepcopy
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from .utils import *
from .localization_assessment import evaluate_template_attack_exploitability, dnn_ablation
from training_modules.supervised_classification import SupervisedClassificationModule
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer as ALLTrainer
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd

LEAKAGE_ASSESSMENT_TECHNIQUES = ['random', 'cpa', 'snr', 'sosd', 'ablation', 'gradvis', 'input_x_grad']

class Trial:
    def __init__(self,
        base_dir,
        profiling_dataset,
        attack_dataset,
        data_module,
        epoch_count=100,
        seed_count=5,
        template_attack_poi_count=20,
        default_supervised_classifier_kwargs={},
        default_all_style_classifier_kwargs={}
    ):
        self.base_dir = base_dir
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.data_module = data_module
        self.epoch_count = epoch_count
        self.seed_count = seed_count
        self.template_attack_poi_count = template_attack_poi_count
        self.default_supervised_classifier_kwargs = default_supervised_classifier_kwargs
        self.default_all_style_classifier_kwargs = default_all_style_classifier_kwargs
        
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
        transform = self.profiling_dataset.transform
        self.profiling_dataset.transform = None
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
        self.profiling_dataset.transform = transform
    
    def _plot_leakage_assessment(self, leakage_assessment, ax, **plot_kwargs):
        medians = np.median(leakage_assessment, axis=0)
        mins = np.min(leakage_assessment, axis=0)
        maxes = np.max(leakage_assessment, axis=0)
        t = np.arange(leakage_assessment.shape[-1])
        ax.errorbar(
            t, medians, yerr=[medians-mins, maxes-medians], fmt='o', markersize=1, linewidth=0.25, **plot_kwargs
        )
    
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
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    def plot_ta_exploitability(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_eval.npz')):
            return
        technique_eval = np.load(os.path.join(self.base_dir, f'{name}_eval.npz'))
        if not 'ta_exploitability' in technique_eval.keys():
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        self._plot_ta_exploitability(technique_eval['ta_exploitability'], ax, color='blue')
        fig.savefig(os.path.join(self.base_dir, f'{name}_ta_exploitability.png'))
        plt.close(fig)
    
    def plot_dnn_ablation(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_eval.npz')):
            return
        technique_eval = np.load(os.path.join(self.base_dir, f'{name}_eval.npz'))
        if not 'dnn_ablation' in technique_eval.keys():
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        x = 10*np.arange(technique_eval['dnn_ablation'].shape[1])
        ax.fill_between(x, np.min(technique_eval['dnn_ablation'], axis=0), np.max(technique_eval['dnn_ablation'], axis=0), color='blue', alpha=0.25)
        ax.plot(x, np.median(technique_eval['dnn_ablation'], axis=0), color='blue')
        ax.set_xlabel('Number of ablated points')
        ax.set_ylabel('Mean rank')
        fig.savefig(os.path.join(self.base_dir, f'{name}_dnn_ablation.png'))
        plt.close(fig)
    
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
                technique_eval = dict(technique_eval)
            if template_attack and not('ta_exploitability' in technique_eval.keys()):
                technique_eval['ta_exploitability'] = np.stack([
                    evaluate_template_attack_exploitability(self.profiling_dataset, self.attack_dataset, _leakage_assessment, poi_count=self.template_attack_poi_count)
                    for _leakage_assessment in leakage_assessment
                ])
            if not('dnn_ablation' in technique_eval.keys()):
                evals = []
                for seed in range(self.seed_count):
                    classifier = self.load_optimal_supervised_classifier(seed).model
                    evals.append(dnn_ablation(classifier, self.data_module.test_dataloader(), leakage_assessment[seed, ...]))
                technique_eval['dnn_ablation'] = np.stack(evals)
            np.savez(os.path.join(self.base_dir, f'{technique_name}_eval.npz'), **technique_eval)
    
    def train_supervised_classifier(self, logging_dir, override_kwargs={}):
        if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
        else:
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module_kwargs = copy(self.default_supervised_classifier_kwargs)
            training_module_kwargs.update(override_kwargs)
            training_module = SupervisedClassificationModule(**training_module_kwargs)
            checkpoint = ModelCheckpoint(
                filename='best',
                monitor='val-rank',
                save_top_k=1,
                mode='min'
            )
            trainer = Trainer(
                max_epochs=self.epoch_count,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        return training_curves
    
    def train_all_style_classifier(self, logging_dir, override_kwargs={}):
        if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
        else:
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module_kwargs = copy(self.default_all_style_classifier_kwargs)
            training_module_kwargs.update(override_kwargs)
            training_module = ALLTrainer(
                normalize_erasure_probs_for_classifier=True,
                split_training_steps=len(self.data_module.train_dataloader())*self.epoch_count,
                **training_module_kwargs
            )
            checkpoint = ModelCheckpoint(
                filename='best',
                monitor='val-rank',
                save_top_k=1,
                mode='min'
            )
            trainer = Trainer(
                max_epochs=self.epoch_count,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            training_module = ALLTrainer.load_from_checkpoint(
                os.path.join(logging_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt'),
                **training_module_kwargs
            )
            classifier_state = deepcopy(training_module.classifier.state_dict())
            torch.save(os.path.join(logging_dir, 'classifier_state.pth'))
        classifier_state = torch.load(os.path.join(logging_dir, 'classifier_state.pth'), map_location='cpu')
        return training_curves, classifier_state
    
    def supervised_lr_sweep(self, learning_rates):
        min_val_ranks = []
        sweep_base_dir = os.path.join(self.base_dir, 'lr_sweep')
        os.makedirs(sweep_base_dir, exist_ok=True)
        for learning_rate in learning_rates:
            logging_dir = os.path.join(sweep_base_dir, f'learning_rate={learning_rate}')
            training_curves = self.train_supervised_classifier(logging_dir, override_kwargs={'optimizer_kwargs': {'lr': learning_rate}})
            plot_training_curves(training_curves, logging_dir, keys=[['train-loss', 'val-loss'], ['train-rank', 'val-rank']])
            min_val_rank = training_curves['val-rank'][-1].min()
            min_val_ranks.append(min_val_rank)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(learning_rates, min_val_ranks, color='blue')
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Minimum validation rank achieved')
        ax.set_title('Supervised classification learning rate sweep')
        fig.savefig(os.path.join(sweep_base_dir, 'lr_sweep.png'))
        optimal_learning_rate = learning_rates[np.argmin(min_val_ranks)]
        self.optimal_learning_rate = optimal_learning_rate
    
    def train_optimal_supervised_classifier(self):
        assert hasattr(self, 'optimal_learning_rate')
        base_dir = os.path.join(self.base_dir, 'standard_classifier')
        for seed in range(self.seed_count):
            set_seed(seed)
            self.train_supervised_classifier(os.path.join(base_dir, f'seed={seed}'), override_kwargs={'optimizer_kwargs': {'lr': self.optimal_learning_rate}})
    
    def train_optimal_all_classifier(self):
        assert hasattr(self, 'optimal_learning_rate')
        base_dir = os.path.join(self.base_dir, 'all_classifier')
        for seed in range(self.seed_count):
            set_seed(seed)
            self.train_all_style_classifier(os.path.join(base_dir, f'seed={seed}'), override_kwargs={'classifier_optimizer_kwargs': {'lr': self.optimal_learning_rate}})
    
    def load_optimal_supervised_classifier(self, seed):
        assert hasattr(self, 'optimal_learning_rate')
        classifier_dir = os.path.join(self.base_dir, 'standard_classifier', f'seed={seed}')
        checkpoint_path = os.path.join(classifier_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt')
        training_module_kwargs = copy(self.default_supervised_classifier_kwargs)
        training_module_kwargs.update({'optimizer_kwargs': {'lr': self.optimal_learning_rate}})
        training_module = SupervisedClassificationModule.load_from_checkpoint(
            checkpoint_path,
            **training_module_kwargs
        )
        return training_module
    
    def load_optimal_all_classifier(self, seed):
        assert hasattr(self, 'optimal_learning_rate')
        classifier_dir = os.path.join(self.base_dir, 'all_classifier', f'seed={seed}')
        checkpoint_path = os.path.join(classifier_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt')
        training_module_kwargs = copy(self.default_all_style_classifier_kwargs)
        training_module_kwargs.update({'classifier_optimizer_kwargs': {'lr': self.optimal_learning_rate}})
        training_module = ALLTrainer.load_from_checkpoint(
            checkpoint_path,
            **training_module_kwargs
        )
        return training_module
    
    def compute_neural_net_explainability_baselines(self):
        for name, fn in zip(['ablation', 'gradvis', 'input_x_grad'], [compute_feature_ablation_map, compute_gradvis, compute_input_x_gradient]):
            if self.load_leakage_assessment(name) is None:
                leakage_assessments = []
                for seed in range(self.seed_count):
                    training_module = self.load_optimal_supervised_classifier(seed)
                    leakage_assessment = fn(training_module, self.profiling_dataset).squeeze()
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                self.save_leakage_assessment(name, leakage_assessments)
    
    def plot_everything(self):
        for technique_name in LEAKAGE_ASSESSMENT_TECHNIQUES:
            self.plot_leakage_assessment(technique_name)
            self.plot_ta_exploitability(technique_name)
            self.plot_dnn_ablation(technique_name)