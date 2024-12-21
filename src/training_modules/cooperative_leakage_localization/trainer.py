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
        default_training_module_kwargs: dict = {}
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.gradient_estimation_strategy = gradient_estimation_strategy
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        
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
                classifiers_pretrain_phase=True,
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
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        plot_training_curves(logging_dir, anim_gammas=False)
    
    def run(self,
        logging_dir: Union[str, os.PathLike],
        pretrained_classifiers_logging_dir: Optional[Union[str, os.PathLike]] = None,
        max_steps: int = 1000,
        anim_gammas: bool = True,
        override_kwargs: dict = {}
    ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            training_module = Module(
                timesteps_per_trace=self.profiling_dataset.timesteps_per_trace,
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
        plot_training_curves(logging_dir, anim_gammas=anim_gammas)