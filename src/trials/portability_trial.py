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
        obf_epoch_count=100,
        seed_count=1,
        template_attack_poi_count=20,
        train_indices=[0, 1, 2, 3],
        default_supervised_classifier_kwargs={},
        default_all_style_classifier_kwargs={}
    ):
        self.base_dir = base_dir
        self.profiling_datasets = profiling_datasets
        self.attack_datasets = attack_datasets
        self.data_modules = data_modules
        self.epoch_count = epoch_count
        self.obf_epoch_count = obf_epoch_count
        self.seed_count = seed_count
        self.template_attack_poi_count = template_attack_poi_count
        self.train_indices = train_indices
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
                for idx in self.train_indices:
                    profiling_dataset = self.profiling_datasets[idx]
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
            if not('dnn_ablation' in technique_eval.keys()) or (len(technique_eval['dnn_ablation'].shape) < 4):
                print(f'Calculating DNN ablation results for {technique_name} technique')
                result = np.full((len(self.train_indices), len(self.data_modules), self.seed_count, self.profiling_datasets[0].timesteps_per_trace//10+1), np.nan, dtype=np.float32)
                for train_idx in range(len(self.train_indices)):
                    for dev_idx, data_module in enumerate(self.data_modules):
                        for seed in range(self.seed_count):
                            classifier = self.load_optimal_supervised_classifier(dev_idx, seed).model
                            _leakage_assessment = leakage_assessment[dev_idx, seed]
                            result[train_idx, dev_idx, seed, :] = dnn_ablation(classifier, data_module.test_dataloader(), _leakage_assessment)
                technique_eval['dnn_ablation'] = result
            np.savez(os.path.join(self.base_dir, f'{technique_name}_eval.npz'), **technique_eval)
    
    def plot_ta_exploitability(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_eval.npz')):
            return
        technique_eval = np.load(os.path.join(self.base_dir, f'{name}_eval.npz'))
        if not 'ta_exploitability' in technique_eval.keys():
            return
        results = technique_eval['ta_exploitability']
        dataset_count = results.shape[0]
        assert dataset_count == results.shape[1]
        fig, axes = plt.subplots(len(self.train_indices), dataset_count, figsize=(4*dataset_count, 4*len(self.train_indices)))
        if len(self.train_indices) == 1:
            axes = axes.reshape((1, -1))
        for row_idx in range(len(self.train_indices)):
            for col_idx in range(dataset_count):
                ax = axes[row_idx, col_idx]
                result = results[row_idx, col_idx]
                self._plot_ta_exploitability(result, ax, color='blue')
                ax.set_xlabel('Traces seen')
                ax.set_ylabel('Guessing entropy')
                ax.set_title(f'Train {row_idx}, test {col_idx}')
        fig.tight_layout()
        fig.savefig(os.path.join(self.base_dir, f'{name}_ta_exploitability.png'))

    def _plot_dnn_ablation(self, dnn_ablation, ax, color='blue', label=None):
        x = 10*np.arange(dnn_ablation.shape[-1])
        ax.fill_between(x, np.min(dnn_ablation, axis=0), np.max(dnn_ablation, axis=0), alpha=0.25, color=color, rasterized=True)
        ax.plot(x, np.median(dnn_ablation, axis=0), color=color, label=label, rasterized=True)
    
    def plot_dnn_ablation(self, name):
        if not os.path.exists(os.path.join(self.base_dir, f'{name}_eval.npz')):
            return
        technique_eval = np.load(os.path.join(self.base_dir, f'{name}_eval.npz'))
        if not 'ta_exploitability' in technique_eval.keys():
            return
        results = technique_eval['dnn_ablation']
        print(results.shape)
        dataset_count = results.shape[0]
        fig, axes = plt.subplots(len(self.train_indices), dataset_count, figsize=(4, 4))
        if len(self.train_indices) == 1:
            axes = axes.reshape((1, -1))
        for row_idx in range(len(self.train_indices)):
            for col_idx in range(dataset_count):
                ax = axes[row_idx, col_idx]
                result = results[row_idx, col_idx]
                print(result.shape)
                assert False
                self._plot_dnn_ablation(result, ax)
                ax.set_xlabel('Number of ablated points')
                ax.set_ylabel('Guessing entropy')
                ax.set_title(f'Train {row_idx}, test {col_idx}')
        fig.tight_layout()
        fig.savefig(os.path.join(self.base_dir, f'{name}_dnn_ablation.png'))
    
    def plot_everything(self):
        for technique_name in LEAKAGE_ASSESSMENT_TECHNIQUES:
            self.plot_leakage_assessment(technique_name)
            self.plot_ta_exploitability(technique_name)
            self.plot_dnn_ablation(technique_name)
    
    def train_supervised_classifier(self, data_module, logging_dir, override_kwargs={}):
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
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        return training_curves
    
    def train_all_classifier(self, data_module, logging_dir, override_kwargs={}):
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
                split_training_steps=len(data_module.train_dataloader())*self.epoch_count,
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
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
            training_module = ALLTrainer.load_from_checkpoint(
                os.path.join(logging_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt'),
                **training_module_kwargs
            )
            classifier_state = deepcopy(training_module.classifier.state_dict())
            torch.save(classifier_state, os.path.join(logging_dir, 'classifier_state.pth'))
        classifier_state = torch.load(os.path.join(logging_dir, 'classifier_state.pth'), map_location='cpu', weights_only=True)
        return training_curves, classifier_state
    
    def run_all_algorithm(self, idx, logging_dir, seed=0, override_kwargs={}):
        assert hasattr(self, 'optimal_learning_rates')
        optimal_learning_rate = self.optimal_learning_rates[idx]
        data_module = self.data_modules[idx]
        if os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
            training_module = ALLTrainer.load_from_checkpoint(
                os.path.join(logging_dir, 'final_checkpoint.ckpt'),
                **self.default_all_style_classifier_kwargs
            )
            erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
        else:
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            module_kwargs = self.default_all_style_classifier_kwargs
            module_kwargs.update({'classifier_optimizer_kwargs': {'lr': self.optimal_learning_rates[idx]}})
            module_kwargs.update(override_kwargs)
            training_module = ALLTrainer(
                **module_kwargs
            )
            training_module.classifier.load_state_dict(torch.load(os.path.join(self.base_dir, 'all_classifier', f'device_{idx}', f'seed={seed}', 'classifier_state.pth'), weights_only=True))
            training_module.classifier = torch.compile(training_module.classifier)
            trainer = Trainer(
                max_epochs=self.obf_epoch_count,
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.fit(training_module, datamodule=data_module)
            erasure_probs = nn.functional.sigmoid(training_module.unsquashed_obfuscation_weights).detach().cpu().numpy().squeeze()
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        return training_curves, erasure_probs
    
    def supervised_lr_sweep(self, learning_rates):
        self.optimal_learning_rates = []
        for idx in range(len(self.data_modules)):
            data_module = self.data_modules[idx]
            min_val_ranks = []
            sweep_base_dir = os.path.join(self.base_dir, 'lr_sweep', f'device_{idx}')
            os.makedirs(sweep_base_dir, exist_ok=True)
            for learning_rate in learning_rates:
                logging_dir = os.path.join(sweep_base_dir, f'learning_rate={learning_rate}')
                training_curves = self.train_supervised_classifier(data_module, logging_dir, override_kwargs={'optimizer_kwargs': {'lr': learning_rate}})
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
            self.optimal_learning_rates.append(optimal_learning_rate)
            
    def lambda_sweep(self, lambda_vals):
        print('Doing lambda sweep')
        self.optimal_lambdas = []
        for idx in self.train_indices:
            data_module = self.data_modules[idx]
            perf_corr_vals = []
            sweep_base_dir = os.path.join(self.base_dir, 'lambda_sweep', f'dev_{idx}')
            os.makedirs(sweep_base_dir, exist_ok=True)
            if not os.path.exists(os.path.join(sweep_base_dir, 'perf_corr_vals.pickle')):
                for lambda_val in lambda_vals:
                    logging_dir = os.path.join(sweep_base_dir, f'lambda={lambda_val}')
                    training_curves, erasure_probs = self.run_all_algorithm(idx, logging_dir, override_kwargs={'obfuscator_l2_norm_penalty': lambda_val})
                    plot_training_curves(
                        training_curves, logging_dir,
                        keys=[['classifier-train-loss_epoch', 'classifier-val-loss'], ['train-rank', 'val-rank'], ['obfuscator-train-loss_epoch', 'obfuscator-val-loss'], ['min-obf-weight', 'max-obf-weight', 'mean-obf-weight']]
                    )
                    fig, ax = plt.subplots(figsize=(4, 4))
                    self._plot_leakage_assessment(erasure_probs[np.newaxis, :], ax)
                    ax.set_yscale('log')
                    fig.savefig(os.path.join(logging_dir, 'erasure_probs.png'))
                    plt.close(fig)
                    perf_corr, _ = evaluate_gmm_exploitability(data_module.train_dataset, data_module.val_dataset, erasure_probs, poi_count=self.template_attack_poi_count, fast=True)
                    perf_corr_vals.append(perf_corr)
                with open(os.path.join(sweep_base_dir, 'perf_corr_vals.pickle'), 'wb') as f:
                    pickle.dump((lambda_vals, perf_corr_vals), f)
            else:
                with open(os.path.join(sweep_base_dir, 'perf_corr_vals.pickle'), 'rb') as f:
                    (lambda_vals, perf_corr_vals) = pickle.load(f)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(lambda_vals, perf_corr_vals, color='blue', marker='.', linestyle='--')
            ax.set_xscale('log')
            ax.set_xlabel('Erasure probs norm penalty: $\lambda$')
            ax.set_ylabel('GMM SKTCC')
            ax.set_title('Adversarial leakage localization $\lambda$ sweep')
            fig.tight_layout()
            fig.savefig(os.path.join(sweep_base_dir, 'lambda_sweep.pdf'))
            plt.close(fig)
            optimal_lambda = lambda_vals[np.argmax(perf_corr_vals)]
            self.optimal_lambdas.append(optimal_lambda)
    
    def train_optimal_supervised_classifier(self):
        print('Training optimal supervised classifier')
        assert hasattr(self, 'optimal_learning_rates')
        base_dir = os.path.join(self.base_dir, 'standard_classifier')
        for idx in range(len(self.data_modules)):
            data_module = self.data_modules[idx]
            for seed in range(self.seed_count):
                set_seed(seed)
                logging_dir = os.path.join(base_dir, f'device_{idx}', f'seed={seed}')
                training_curves = self.train_supervised_classifier(data_module, logging_dir, override_kwargs={'optimizer_kwargs': {'lr': self.optimal_learning_rates[idx]}})
                plot_training_curves(training_curves, logging_dir, keys=[['train-loss', 'val-loss'], ['train-rank', 'val-rank']])
    
    def train_optimal_all_classifier(self):
        print('Training optimal ALL classifier')
        assert hasattr(self, 'optimal_learning_rates')
        base_dir = os.path.join(self.base_dir, 'all_classifier')
        for idx in self.train_indices:
            data_module = self.data_modules[idx]
            for seed in range(self.seed_count):
                set_seed(seed)
                logging_dir = os.path.join(base_dir, f'device_{idx}', f'seed={seed}')
                training_curves, _ = self.train_all_classifier(data_module, logging_dir, override_kwargs={'classifier_optimizer_kwargs': {'lr': 0.5*self.optimal_learning_rates[idx]}})
                plot_training_curves(training_curves, logging_dir, keys=[['classifier-train-loss_epoch', 'classifier-val-loss'], ['train-rank', 'val-rank']])
    
    def run_optimal_all(self):
        print('Running optimal ALL')
        if self.load_leakage_assessment('all') is None:
            print('Running ALL with optimal hyperparameters')
            assert hasattr(self, 'optimal_learning_rates') and hasattr(self, 'optimal_lambdas')
            base_dir = os.path.join(self.base_dir, 'all_leakage_assessments')
            leakage_assessments = []
            for idx in self.train_indices:
                dev_dir = os.path.join(base_dir, f'dev_{idx}')
                _leakage_assessments = []
                for seed in range(self.seed_count):
                    set_seed(seed)
                    logging_dir = os.path.join(dev_dir, f'seed={seed}')
                    training_curves, erasure_probs = self.run_all_algorithm(idx, logging_dir, seed, override_kwargs={'obfuscator_l2_norm_penalty': self.optimal_lambdas[idx]})
                    plot_training_curves(
                        training_curves, logging_dir,
                        keys=[['classifier-train-loss_epoch', 'classifier-val-loss'], ['train-rank', 'val-rank'],
                              ['obfuscator-train-loss_epoch', 'obfuscator-val-loss'], ['min-obf-weight', 'max-obf-weight', 'mean-obf-weight']]
                    )
                    _leakage_assessments.append(self.optimal_lambdas[idx]*erasure_probs)
                leakage_assessments.append(np.stack(_leakage_assessments))
            self.save_leakage_assessment('all', np.stack(leakage_assessments))
    
    def load_optimal_supervised_classifier(self, dev_idx, seed):
        assert hasattr(self, 'optimal_learning_rates')
        classifier_dir = os.path.join(self.base_dir, 'standard_classifier', f'device_{dev_idx}', f'seed={seed}')
        checkpoint_path = os.path.join(classifier_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt')
        training_module_kwargs = copy(self.default_supervised_classifier_kwargs)
        training_module_kwargs.update({'optimizer_kwargs': {'lr': self.optimal_learning_rates[dev_idx]}})
        training_module = SupervisedClassificationModule.load_from_checkpoint(
            checkpoint_path,
            **training_module_kwargs
        )
        return training_module
    
    def load_optimal_all_classifier(self, dev_idx, seed):
        assert hasattr(self, 'optimal_learning_rates')
        classifier_dir = os.path.join(self.base_dir, 'all_classifier', f'device_{dev_idx}', f'seed={seed}')
        checkpoint_path = os.path.join(classifier_dir, 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt')
        training_module_kwargs = copy(self.default_all_style_classifier_kwargs)
        training_module_kwargs.update({'classifier_optimizer_kwargs': {'lr': 0.5*self.optimal_learning_rates[dev_idx]}})
        training_module = ALLTrainer.load_from_checkpoint(
            checkpoint_path,
            **training_module_kwargs
        )
        return training_module