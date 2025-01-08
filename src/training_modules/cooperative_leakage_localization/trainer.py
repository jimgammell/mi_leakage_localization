from copy import copy
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
        gradient_estimation_strategy: Literal['REINFORCE'] = 'REINFORCE',
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {},
        reference_leakage_assessment: Optional[np.ndarray] = None
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.gradient_estimation_strategy = gradient_estimation_strategy
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        self.reference_leakage_assessment = reference_leakage_assessment
        
        self.data_module = DataModule(
            self.profiling_dataset,
            self.attack_dataset,
            train_batch_size=512,
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
            training_module = Module(
                train_etat=False,
                timesteps_per_trace=self.profiling_dataset.timesteps_per_trace,
                **kwargs
            )
            trainer = LightningTrainer(
                max_steps=max_steps,
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output')
            )
            trainer.fit(training_module, datamodule=self.data_module)
            if training_module.hparams.calibrate_classifiers:
                training_module.calibrate_classifiers()
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        plot_training_curves(logging_dir, anim_gammas=False)
    
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
                reference_leakage_assessment=self.reference_leakage_assessment,
                **kwargs
            )
            if pretrained_classifiers_logging_dir is not None:
                assert os.path.exists(pretrained_classifiers_logging_dir)
                pretrained_module = Module.load_from_checkpoint(os.path.join(pretrained_classifiers_logging_dir, 'final_checkpoint.ckpt'))
                training_module.cmi_estimator.classifiers.load_state_dict(pretrained_module.cmi_estimator.classifiers.state_dict())
            trainer = LightningTrainer(
                max_steps=max_steps,
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
            plot_leakage_assessment(training_module.selection_mechanism.get_accumulated_gamma().reshape(-1), os.path.join(logging_dir, 'leakage_assessment.png'))
        training_curves = load_training_curves(logging_dir)
        plot_training_curves(logging_dir, anim_gammas=anim_gammas, reference=reference)
    
    def hparam_tune(self,
        logging_dir: Union[str, os.PathLike],
        pretrained_classifiers_logging_dir: Optional[Union[str, os.PathLike]] = None,
        max_steps: int = 1000,
        anim_gammas: bool = False,
        override_kwargs: dict = {}
    ):
        while True:
            budget = 10**np.random.uniform(1, 3)
            etat_lr = 10**np.random.uniform(-5, -3)
            theta_lr = 10**np.random.uniform(-5, -3)
            experiment_dir = os.path.join(logging_dir, f'budget={budget}__etat_lr={etat_lr}__theta_lr={theta_lr}')
            override_kwargs['budget'] = budget
            override_kwargs['etat_lr'] = etat_lr
            override_kwargs['theta_lr'] = theta_lr
            self.run(
                logging_dir=experiment_dir,
                pretrained_classifiers_logging_dir=pretrained_classifiers_logging_dir,
                max_steps=max_steps,
                anim_gammas=anim_gammas,
                override_kwargs=override_kwargs
            )
            training_curves = load_training_curves(experiment_dir)
            best_gmm_corr = np.max(training_curves['gmmperfcorr'][-1])
            print(f'GMM perf corr: {best_gmm_corr} in directory: {experiment_dir}')