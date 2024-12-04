from typing import *
import os
from scipy.interpolate import UnivariateSpline
import torch
from torch.utils.data import Dataset

from common import *
from .utils import *
from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer

def smooth_sweep(x, y):
    spline = UnivariateSpline(x, y, s=0.1)
    return (x, spline(x))

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
        
    def all_theta_lr_sweep(self, learning_rates):
        learning_rates = np.array(learning_rates, dtype=np.float32)
        es_val_rank, es_val_loss, final_val_rank, final_val_loss = map(lambda _: np.full((len(learning_rates),), np.nan, dtype=np.float32), range(4))
        base_dir = os.path.join(self.base_dir, 'all_theta_lr_sweep')
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
        optimal_idx = np.argmin(smooth_sweep(learning_rates, es_val_loss)[1])
        
        fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
        axes[0].plot(learning_rates, es_val_loss, marker='o', linestyle='none', color='blue')
        axes[0].plot(*smooth_sweep(learning_rates, es_val_loss), linestyle='-', color='blue')
        axes[0].plot(learning_rates, final_val_loss, marker='s', linestyle='none', color='blue')
        axes[0].plot(*smooth_sweep(learning_rates, final_val_loss), linestyle='--', color='blue')
        axes[0].axvline(learning_rates[optimal_idx], color='red', linestyle=':')
        axes[1].plot(learning_rates, es_val_rank, marker='o', linestyle='none', color='blue')
        axes[1].plot(*smooth_sweep(learning_rates, es_val_rank), linestyle='-', color='blue')
        axes[1].plot(learning_rates, final_val_rank, marker='s', linestyle='none', color='blue')
        axes[1].plot(*smooth_sweep(learning_rates, final_val_rank), linestyle='--', color='blue')
        axes[1].plot(learning_rates[optimal_idx], color='red', linestyle=':')
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