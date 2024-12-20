from torch.utils.data import Dataset
from lightning import LightningModule, Trainer
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
        max_steps: int = 1000,
        gradient_estimation_strategy: Literal['REINFORCE'] = 'REINFORCE',
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {}
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.max_steps = max_steps
        self.gradient_estimation_strategy = gradient_estimation_strategy
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        
        self.data_module = DataModule(
            self.profiling_dataset,
            self.attack_dataset,
            **self.default_data_module_kwargs
        )
    
    def run(self,
        logging_dir: Union[str, os.PathLike],
        override_kwargs: dict = {},
        anim_gammas: bool = True
    ):
        assert override_kwargs == {}
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            training_module = Module(**self.default_training_module_kwargs)
            trainer = Trainer(
                max_steps=self.max_steps,
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