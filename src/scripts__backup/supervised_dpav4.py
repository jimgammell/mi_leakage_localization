import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from _common import *
from datasets.dpav4 import DPAv4, to_key_preds
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from utils.metrics.rank import accumulate_ranks
import datasets
from training_modules.supervised_classification import SupervisedClassificationModule

ROOT = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4')
RUN_LR_SWEEP = True

profiling_dataset = DPAv4(
    root=ROOT, train=True
)
attack_dataset = DPAv4(
    root=ROOT, train=False
)

data_module = datasets.load('dpav4', train_batch_size=1024, eval_batch_size=10000, root=ROOT)
learning_rates = [x*1e-6 for x in range(1, 10)] + [x*1e-5 for x in range(1, 10)] + [1e-4]
weight_decays = [0.0, 1e-2]
additive_noises = [0.0, 0.25]
if RUN_LR_SWEEP:
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for additive_noise in additive_noises:
                logging_dir = os.path.join(get_trial_dir(), f'learning_rate={learning_rate}__weight_decay={weight_decay}__additive_noise={additive_noise}')
                training_module = SupervisedClassificationModule(
                    model_name='sca-cnn',
                    optimizer_name='AdamW',
                    model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace)},
                    optimizer_kwargs={'lr': learning_rate, 'weight_decay': weight_decay},
                    additive_noise_augmentation=additive_noise
                )
                trainer = Trainer(
                    max_epochs=int(10000*1024/len(profiling_dataset)),
                    default_root_dir=logging_dir,
                    accelerator='gpu',
                    devices=1,
                    logger=TensorBoardLogger(logging_dir, name='lightning_output')
                )
                trainer.fit(training_module, datamodule=data_module)
                trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
                ea = event_accumulator.EventAccumulator(os.path.join(logging_dir, 'lightning_output', 'version_0'))
                ea.Reload()
                with open(os.path.join(logging_dir, 'training_curves.pickle'), 'wb') as f:
                    pickle.dump({
                        key: extract_trace(ea.Scalars(key)) for key in ['train-loss', 'val-loss', 'train-rank', 'val-rank']
                    }, f)
                rank_over_time = accumulate_ranks(training_module, int_var_to_key_fn=to_key_preds, args=('plaintext', 'offset'), constants=['mask'])
                with open(os.path.join(logging_dir, 'rank_over_time.pickle'), 'wb') as f:
                    pickle.dump(rank_over_time, f)