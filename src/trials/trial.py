from typing import *
import os
from scipy.interpolate import UnivariateSpline
import torch
from torch.utils.data import Dataset
from lightning.pytorch.tuner import Tuner

from common import *
from .utils import *
from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer

def smooth_sweep(x, y, kernel_size=3):
    smoothed_y = np.convolve(y, np.ones(kernel_size)/kernel_size, mode='valid')
    x = x[kernel_size//2:-(kernel_size//2)]
    return (x, smoothed_y)

class Trial:
    def __init__(self,
        base_dir: Union[str, os.PathLike],
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        supervised_classifier_kwargs: dict,
        all_kwargs: dict
    ):
        self.base_dir = base_dir
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.supervised_classifier_kwargs = supervised_classifier_kwargs
        self.all_kwargs = all_kwargs
        
        os.makedirs(self.base_dir, exist_ok=True)
        
    def all_lambda_sweep(self, lambda_vals, starting_module_path=None):
        lambda_vals = np.array(lambda_vals, dtype=np.float32)
        base_dir = os.path.join(self.base_dir, 'all_lambda_sweep')
        trainer = AdversarialLeakageLocalizationTrainer(
            profiling_dataset=self.profiling_dataset,
            attack_dataset=self.attack_dataset,
            **self.all_kwargs
        )
        os.makedirs(base_dir, exist_ok=True)
        for idx, lambda_val in enumerate(lambda_vals):
            logging_dir = os.path.join(base_dir, f'lambda={lambda_val}')
            trainer.train_gamma(logging_dir, starting_module_path=starting_module_path, override_kwargs={'gammap_identity_coeff': lambda_val})
        
    def all_theta_smith_lr_sweep(self):
        logging_dir = os.path.join(self.base_dir, 'all_theta_smith_lr_sweep')
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
        trainer = AdversarialLeakageLocalizationTrainer(
            profiling_dataset=self.profiling_dataset,
            attack_dataset=self.attack_dataset,
            **self.all_kwargs
        )
        trainer.smith_lr_sweep(logging_dir)
        
    def all_theta_lr_sweep(self, learning_rates, base_dir_name='all_theta_lr_sweep', smooth_kernel_size=5):
        learning_rates = np.array(learning_rates, dtype=np.float32)
        es_val_rank, es_val_loss, final_val_rank, final_val_loss = map(lambda _: np.full((len(learning_rates),), np.nan, dtype=np.float32), range(4))
        base_dir = os.path.join(self.base_dir, base_dir_name)
        if not os.path.exists(os.path.join(base_dir, 'sweep_results.pickle')):
            trainer = AdversarialLeakageLocalizationTrainer(
                profiling_dataset=self.profiling_dataset,
                attack_dataset=self.attack_dataset,
                **self.all_kwargs
            )
            os.makedirs(base_dir, exist_ok=True)
            for idx, learning_rate in enumerate(learning_rates):
                logging_dir = os.path.join(base_dir, f'lr={learning_rate}')
                if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
                    trainer.pretrain_classifiers(logging_dir, override_kwargs={'theta_optimizer_kwargs': {'lr': learning_rate}})
                with open(os.path.join(logging_dir, 'training_curves.pickle'), 'rb') as f:
                    training_curves = pickle.load(f)
                assert training_curves is not None
                es_idx = np.argmin(training_curves['val_theta_loss'][-1])
                es_val_rank[idx] = training_curves['val_theta_rank'][-1][es_idx]
                es_val_loss[idx] = training_curves['val_theta_loss'][-1][es_idx]
                final_val_rank[idx] = training_curves['val_theta_rank'][-1][-1]
                final_val_loss[idx] = training_curves['val_theta_loss'][-1][-1]
            assert all(np.all(np.isfinite(x)) for x in [es_val_rank, es_val_loss, final_val_rank, final_val_loss])
            optimal_idx = np.argmin(smooth_sweep(learning_rates, es_val_loss, kernel_size=smooth_kernel_size)[1])+(smooth_kernel_size//2)
            with open(os.path.join(base_dir, 'sweep_results.pickle'), 'wb') as f:
                pickle.dump({
                    'learning_rates': learning_rates,
                    'es_val_rank': es_val_rank,
                    'final_val_rank': final_val_rank,
                    'es_val_loss': es_val_loss,
                    'final_val_loss': final_val_loss,
                    'optimal_lr_idx': optimal_idx,
                    'optimal_lr': learning_rates[optimal_idx]
                }, f)
        with open(os.path.join(base_dir, 'sweep_results.pickle'), 'rb') as f:
            sweep_results = pickle.load(f)
            learning_rates = sweep_results['learning_rates']
            es_val_rank = sweep_results['es_val_rank']
            es_val_loss = sweep_results['es_val_loss']
            final_val_rank = sweep_results['final_val_rank']
            final_val_loss = sweep_results['final_val_loss']
            optimal_idx = sweep_results['optimal_lr_idx']
        
        fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
        axes[0].plot(learning_rates, es_val_loss, marker='o', linestyle='none', color='blue')
        axes[0].plot(*smooth_sweep(learning_rates, es_val_loss, kernel_size=smooth_kernel_size), linestyle='-', color='blue')
        axes[0].plot(learning_rates, final_val_loss, marker='s', linestyle='none', color='blue')
        axes[0].plot(*smooth_sweep(learning_rates, final_val_loss, kernel_size=smooth_kernel_size), linestyle='--', color='blue')
        axes[0].axvline(learning_rates[optimal_idx], color='red', linestyle=':')
        axes[1].plot(learning_rates, es_val_rank, marker='o', linestyle='none', color='blue')
        axes[1].plot(*smooth_sweep(learning_rates, es_val_rank, kernel_size=smooth_kernel_size), linestyle='-', color='blue')
        axes[1].plot(learning_rates, final_val_rank, marker='s', linestyle='none', color='blue')
        axes[1].plot(*smooth_sweep(learning_rates, final_val_rank, kernel_size=smooth_kernel_size), linestyle='--', color='blue')
        axes[1].axvline(learning_rates[optimal_idx], color='red', linestyle=':')
        axes[0].set_xlabel('Learning rate')
        axes[0].set_ylabel('Validation loss')
        axes[1].set_xlabel('Learning rate')
        axes[1].set_ylabel('Validation correct key rank')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[1].set_xscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(base_dir, 'lr_sweep.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
        
        return os.path.join(base_dir, f'lr={learning_rates[optimal_idx]}', 'lightning_output', 'version_0', 'checkpoints', 'best.ckpt')