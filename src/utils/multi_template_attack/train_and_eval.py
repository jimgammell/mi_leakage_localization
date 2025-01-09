from typing import *
from copy import copy
from tqdm.auto import tqdm
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from lightning import LightningModule, Trainer as LightningTrainer

from .template_attack import MultiTemplateAttack
from utils.metrics import get_rank

class TimeSubsampleTransform(nn.Module):
    def __init__(self, time_indices: Sequence[int]):
        super().__init__()
        self.time_indices = time_indices
    def forward(self, x):
        return x[..., self.time_indices]

def apply_time_subsample_transform(dataset: Dataset, time_indices: Sequence[int]):
    if dataset.transform is None:
        dataset.transform = TimeSubsampleTransform(time_indices)
    else:
        dataset.transform = Compose([dataset.transform, TimeSubsampleTransform(time_indices)])

class TemplateAttackTrainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        window_size: int = 5,
        max_parallel_timesteps: Optional[int] = None
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.window_size = window_size
        self.max_parallel_timesteps = max_parallel_timesteps if max_parallel_timesteps is not None else self.profiling_dataset.timesteps_per_trace
        self.class_count = self.profiling_dataset.class_count
        self.means = np.zeros((self.profiling_dataset.timesteps_per_trace-self.window_size+1, self.class_count, self.window_size), dtype=np.float32)
        self.p_y = np.zeros(self.class_count)
        for trace, label in self.profiling_dataset:
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()
            self.means[:, label, :] += torch.tensor(trace).reshape(-1).unfold(-1, self.window_size, 1).numpy()
            self.p_y[label] += 1
        self.means /= self.p_y.reshape(1, -1, 1)
        self.p_y /= self.p_y.sum()
    
    def get_training_module(self, timesteps: Sequence[int] = None):
        if timesteps is None:
            timesteps = np.arange(self.profiling_dataset.timesteps_per_trace)
        module = TemplateAttackModule(len(timesteps), self.class_count, window_size=self.window_size, p_y=self.p_y, means=self.means[timesteps, ...])
        return module
    
    def get_subsampled_datasets(self, timesteps: Sequence[int] = None):
        if timesteps is None:
            timesteps = np.arange(self.profiling_dataset.timesteps_per_trace)
        profiling_dataset = copy(self.profiling_dataset)
        attack_dataset = copy(self.attack_dataset)
        apply_time_subsample_transform(profiling_dataset, timesteps)
        apply_time_subsample_transform(attack_dataset, timesteps)
        return profiling_dataset, attack_dataset
    
    def next_timesteps_sequence(self):
        t0 = 0
        t1 = self.max_parallel_timesteps
        done = False
        while not done:
            if t1 >= self.profiling_dataset.timesteps_per_trace:
                done = True
                t1 = self.profiling_dataset.timesteps_per_trace
            yield torch.arange(t0, t1)
            t0 = t1 - self.window_size//2
            t1 = t0 + self.max_parallel_timesteps
    
    def get_sequence_info(self, timesteps):
        training_module = self.get_training_module(timesteps)
        profiling_dataset, attack_dataset = self.get_subsampled_datasets(timesteps)
        profiling_dataloader = DataLoader(profiling_dataset, batch_size=len(profiling_dataset))
        attack_dataloader = DataLoader(attack_dataset, batch_size=len(attack_dataset))
        logger = logging.getLogger('pytorch_lightning')
        logger.setLevel(logging.ERROR)
        trainer = LightningTrainer(
            max_steps=1, logger=False, enable_checkpointing=False, enable_progress_bar=False
        )
        trainer.fit(training_module, train_dataloaders=profiling_dataloader)
        logger.setLevel(logging.INFO)
        with torch.no_grad():
            attack_traces, attack_labels = next(iter(attack_dataloader))
            log_p_y_mid_x = training_module.template_attacker.get_log_p_y_mid_x(attack_traces).cpu().numpy()
            batch_size, window_count, class_count = log_p_y_mid_x.shape
            _attack_labels = attack_labels.reshape(batch_size, 1).repeat(1, window_count).cpu().numpy()
            rank = get_rank(log_p_y_mid_x.reshape(-1, self.class_count), _attack_labels.reshape(-1)).reshape(batch_size, window_count).mean(axis=0)
            mutinf = training_module.template_attacker.get_pointwise_mutinf(attack_traces).cpu().numpy()
            info = {
                'log_p_y_mid_x': log_p_y_mid_x.mean(axis=0),
                'rank': rank,
                'mutinf': mutinf
            }
        return info
    
    def get_info(self):
        info = {'log_p_y_mid_x': [], 'rank': [], 'mutinf': []}
        for timesteps in tqdm(self.next_timesteps_sequence()):
            _info = self.get_sequence_info(timesteps)
            for key, val in _info.items():
                info[key].append(val)
        info = {key: np.concatenate(val, axis=0) for key, val in info.items()}
        return info

class TemplateAttackModule(LightningModule):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5,
        p_y: Optional[Sequence[float]] = None,
        means: Optional[Sequence[float]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.template_attacker = MultiTemplateAttack(
            self.hparams.timesteps_per_trace,
            self.hparams.class_count,
            window_size=self.hparams.window_size,
            p_y=self.hparams.p_y,
            means=self.hparams.means
        )
    
    def configure_optimizers(self):
        self.optimizer = optim.LBFGS(
            self.template_attacker.parameters(), max_iter=100, history_size=10, line_search_fn='strong_wolfe'
        )
        return {'optimizer': self.optimizer}
    
    def training_step(self, batch):
        trace, labels = batch
        optimizer = self.optimizers()
        loss = None
        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self.template_attacker(trace)
            batch_size, window_count, class_count = logits.shape
            _labels = labels.unsqueeze(1).repeat(1, window_count)
            loss = nn.functional.nll_loss(logits.reshape(-1, self.hparams.class_count), _labels.reshape(-1), reduction='none')
            loss = loss.reshape(batch_size, window_count).sum(dim=-1).mean()
            self.manual_backward(loss)
            return loss
        optimizer.step(closure)
        print(loss)
        return loss