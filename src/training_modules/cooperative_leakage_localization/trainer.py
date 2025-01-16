from copy import copy
from collections import defaultdict
from torch.utils.data import Dataset
from lightning import LightningModule, Trainer as LightningTrainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from trials.utils import *
from datasets.data_module import DataModule
from .module import Module
from .plot_things import *

class Trainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {},
        reference_leakage_assessment: Optional[np.ndarray] = None
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        self.reference_leakage_assessment = reference_leakage_assessment
        
        self.data_module = DataModule(
            self.profiling_dataset,
            self.attack_dataset,
            **self.default_data_module_kwargs
        )
    
    def pretrain_classifiers(self,
        logging_dir: Union[str, os.PathLike],
        max_steps: int = 1000,
        override_kwargs: dict = {}
    ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            kwargs['budget'] *= 10
            training_module = Module(
                train_etat=False,
                timesteps_per_trace=self.profiling_dataset.timesteps_per_trace,
                class_count=self.profiling_dataset.class_count,
                **kwargs
            )
            checkpoint = ModelCheckpoint(
                monitor='val_theta_rank',
                mode='min',
                save_top_k=1,
                dirpath=logging_dir,
                filename='best_checkpoint'
            )
            trainer = LightningTrainer(
                max_steps=training_module.to_global_steps(max_steps),
                val_check_interval=1.,
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
        plot_training_curves(logging_dir, anim_gammas=False)
    
    def htune_pretrain_classifiers(self,
        logging_dir: Union[str, os.PathLike],
        trial_count: int = 25,
        max_steps: int = 1000,
        override_kwargs: dict = {}
    ):
        lr_vals = sum([[m*10**n for m in range(1, 10)] for n in range(-6, -2)], start=[])
        beta1_vals = [0.0, 0.5, 0.9, 0.99]
        weight_decay_vals = [0.0, 1e-4, 1e-2]
        lr_schedulers = [None, 'CosineDecayLRSched']
        results = defaultdict(list)
        for trial_idx in range(trial_count):
            experiment_dir = os.path.join(logging_dir, f'trial_{trial_idx}')
            os.makedirs(experiment_dir, exist_ok=True)
            hparams = {
                'theta_lr': np.random.choice(lr_vals),
                'theta_beta_1': np.random.choice(beta1_vals),
                'theta_weight_decay': np.random.choice(weight_decay_vals),
                'theta_lr_scheduler_name': np.random.choice(lr_schedulers)
            }
            override_kwargs.update(hparams)
            self.pretrain_classifiers(
                logging_dir=experiment_dir,
                max_steps=max_steps, 
                override_kwargs=override_kwargs
            )
            with open(os.path.join(experiment_dir, 'hparams.pickle'), 'wb') as f:
                pickle.dump(hparams, f)
            training_curves = get_training_curves(experiment_dir)
            for key, val in hparams.items():
                results[key].append(val)
            optimal_idx = np.argmin(training_curves['val_theta_rank'][-1])
            results['min_rank'].append(training_curves['val_theta_rank'][-1][optimal_idx])
            results['final_rank'].append(training_curves['val_theta_rank'][-1][-1])
            results['min_loss'].append(training_curves['val_theta_loss'][-1][optimal_idx])
            results['final_loss'].append(training_curves['val_theta_loss'][-1][-1])
        with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)
    
    def run(self,
        logging_dir: Union[str, os.PathLike],
        pretrained_classifiers_logging_dir: Optional[Union[str, os.PathLike]] = None,
        max_steps: int = 1000,
        anim_gammas: bool = True,
        override_kwargs: dict = {},
        reference: Optional[np.ndarray] = None
    ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            training_module = Module(
                timesteps_per_trace=self.profiling_dataset.timesteps_per_trace,
                class_count=self.profiling_dataset.class_count,
                reference_leakage_assessment=self.reference_leakage_assessment,
                **kwargs
            )
            if pretrained_classifiers_logging_dir is not None:
                assert os.path.exists(pretrained_classifiers_logging_dir)
                pretrained_module = Module.load_from_checkpoint(os.path.join(pretrained_classifiers_logging_dir, 'best_checkpoint.ckpt'))
                training_module.cmi_estimator.classifiers.load_state_dict(pretrained_module.cmi_estimator.classifiers.state_dict())
            trainer = LightningTrainer(
                max_steps=training_module.to_global_steps(max_steps),
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.fit(training_module, datamodule=self.data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
            leakage_assessment = training_module.selection_mechanism.get_accumulated_gamma().reshape(-1)
            plot_leakage_assessment(leakage_assessment, os.path.join(logging_dir, 'leakage_assessment.png'))
        training_curves = load_training_curves(logging_dir)
        plot_training_curves(logging_dir, anim_gammas=anim_gammas, reference=reference)
        return leakage_assessment