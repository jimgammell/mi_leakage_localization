from typing import *
import os
import numpy as np

from datasets.aes_pt_v2 import AES_PTv2, VALID_DEVICES as _VALID_DEVICES
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
VALID_DEVICES = [x for x in _VALID_DEVICES if x != 'Pinata']

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike],
        countermeasure: Literal['MS1', 'MS2', 'Unprotected'] = 'Unprotected',
        trial_config: dict = {},
        seed_count: int = 1,
        batch_size: int = 1000,
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        pretrain_classifiers_only: bool = False
    ):
        self.logging_dir = logging_dir
        self.countermeasure = countermeasure
        self.trial_config = trial_config
        self.seed_count = seed_count
        if self.seed_count != 1:
            raise NotImplementedError
        self.batch_size = batch_size
        self.run_kwargs = {'max_steps': 100000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'budget': 100, 'theta_lr': 1e-3, 'etat_lr': 1e-3}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.pretrain_classifiers_only = pretrain_classifiers_only
    
    def get_leave_3_out_datasets(self):
        out = {}
        for device in VALID_DEVICES:
            profiling_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=True, devices=device
            )
            attack_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=False, devices=[d for d in VALID_DEVICES if d != device]
            )
            out[device] = (profiling_dataset, attack_dataset)
        return out
    
    def get_leave_1_out_datasets(self):
        out = {}
        for device in VALID_DEVICES:
            profiling_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=True, devices=[d for d in VALID_DEVICES if d != device]
            )
            attack_dataset = AES_PTv2(
                root=self.trial_config['data_dir'], train=False, devices=device
            )
            out[device] = (profiling_dataset, attack_dataset)
        return out
    
    def get_leave_0_out_dataset(self):
        profiling_dataset = AES_PTv2(
            root=self.trial_config['data_dir'], train=True, devices=VALID_DEVICES
        )
        attack_dataset = AES_PTv2(
            root=self.trial_config['data_dir'], train=False, devices=VALID_DEVICES
        )
        return (profiling_dataset, attack_dataset)
    
    def get_datasets(self):
        return {
            **{key: val for key, val in self.get_leave_3_out_datasets().items()},
            **{f'-{key}': val for key, val in self.get_leave_1_out_datasets().items()},
            'all': self.get_leave_0_out_dataset()
        }
        
    def get_trainer(self, profiling_dataset, attack_dataset):
        trainer = LeakageLocalizationTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs=self.leakage_localization_kwargs
        )
        return trainer
    
    def run_leakage_assessments(self):
        leakage_assessments = {}
        for dataset_name, (profiling_dataset, attack_dataset) in self.get_datasets().items():
            logging_dir = os.path.join(self.logging_dir, f'dataset_name={dataset_name}')
            os.makedirs(logging_dir, exist_ok=True)
            trainer = self.get_trainer(profiling_dataset, attack_dataset)
            trainer.pretrain_classifiers(
                os.path.join(logging_dir, 'classifiers_pretrain'),
                max_steps=self.run_kwargs['max_steps']//10
            )
            if not self.pretrain_classifiers_only:
                leakage_assessment = trainer.run(
                    os.path.join(logging_dir, 'leakage_localization'),
                    pretrained_classifiers_logging_dir=os.path.join(logging_dir, 'classifiers_pretrain'),
                    **self.run_kwargs
                )
            else:
                os.makedirs(os.path.join(logging_dir, 'leakage_localization'), exist_ok=True)
                leakage_assessment = None
            np.savez(os.path.join(logging_dir, 'leakage_localization', 'leakage_assessment.npy'), leakage_assessment=leakage_assessment)
    
    def __call__(self):
        self.run_leakage_assessments()