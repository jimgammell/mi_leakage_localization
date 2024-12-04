from typing import *
import os
from copy import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
import lightning as L

from common import *
from utils.calculate_dataset_stats import calculate_dataset_stats

def set_transforms(dataset, transform, target_transform):
    if isinstance(dataset, Subset):
        set_transforms(dataset.dataset, transform, target_transform)
    elif isinstance(dataset, ConcatDataset):
        for _dataset in dataset.datasets:
            set_transforms(_dataset, transform, target_transform)
    else:
        assert isinstance(dataset, Dataset)
        dataset.transform = transform
        dataset.target_transform = target_transform

class DataModule(L.LightningDataModule):
    def __init__(self,
        profiling_dataset,
        attack_dataset,
        val_prop: float = 0.2,
        train_batch_size: int = 2048,
        aug_train_batch_size: int = 256,
        eval_batch_size: int = 2048,
        adversarial_mode: bool = False,
        data_mean: Optional[Union[float, Sequence[float]]] = None,
        data_var: Optional[Union[float, Sequence[float]]] = None,
        dataloader_kwargs: dict = {},
        aug_transform: Optional[Callable] = None
    ):
        super().__init__()
        self.profiling_dataset = copy(profiling_dataset)
        self.attack_dataset = copy(attack_dataset)
        self.val_prop = val_prop
        self.train_batch_size = train_batch_size
        self.aug_train_batch_size = aug_train_batch_size
        self.eval_batch_size = eval_batch_size
        self.adversarial_mode = adversarial_mode
        self.data_mean = data_mean
        self.data_var = data_var
        self.dataloader_kwargs = dataloader_kwargs
        self.aug_transform = aug_transform
        self.train_indices = self.val_indices = None
        self.setup('')
    
    def setup(self, stage: str):
        if self.data_mean is None or self.data_var is None:
            self.data_mean, self.data_var = calculate_dataset_stats(self.profiling_dataset)
        self.data_mean, self.data_var = map(
            lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(torch.float32), (self.data_mean, self.data_var)
        )
        basic_transform_mods = [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: (x - self.data_mean)/self.data_var.sqrt().clamp(min=1e-6))
        ]
        self.data_transform = transforms.Compose(basic_transform_mods)
        self.target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.val_length = int(len(self.profiling_dataset)*self.val_prop)
        self.train_length = len(self.profiling_dataset) - self.val_length
        if (self.train_indices is None) or (self.val_indices is None):
            indices = np.random.choice(len(self.profiling_dataset), len(self.profiling_dataset), replace=False)
            self.train_indices = indices[:self.train_length]
            self.val_indices = indices[self.train_length:]
        self.train_dataset = Subset(copy(self.profiling_dataset), self.train_indices)
        self.val_dataset = Subset(copy(self.profiling_dataset), self.val_indices)
        set_transforms(self.train_dataset, self.data_transform, self.target_transform)
        set_transforms(self.val_dataset, self.data_transform, self.target_transform)
        set_transforms(self.attack_dataset, self.data_transform, self.target_transform)
        if self.adversarial_mode:
            if self.aug_transform is not None:
                self.aug_data_transform = transforms.Compose(basic_transform_mods + [self.aug_transform])
            else:
                self.aug_data_transform = self.data_transform
            self.aug_train_dataset = copy(self.train_dataset)
            set_transforms(self.aug_train_dataset, self.aug_data_transform, self.target_transform)
        dataloader_kwargs = {
            'num_workers': max(os.cpu_count()//2, 1),
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 4
        }
        dataloader_kwargs.update(self.dataloader_kwargs)
        self.dataloader_kwargs = dataloader_kwargs
    
    def train_dataloader(self, override_batch_size=None, override_aug_batch_size=None):
        train_batch_size = self.train_batch_size if override_batch_size is None else override_batch_size
        aug_batch_size = self.aug_train_batch_size if override_aug_batch_size is None else override_aug_batch_size
        train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, **self.dataloader_kwargs)
        if self.adversarial_mode:
            aug_dataloader = DataLoader(self.aug_train_dataset, batch_size=aug_batch_size, shuffle=True, **self.dataloader_kwargs)
            return [aug_dataloader, train_dataloader]
        else:
            return train_dataloader
    
    def val_dataloader(self, override_batch_size=None):
        batch_size = self.eval_batch_size if override_batch_size is None else override_batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)
    
    def test_dataloader(self, override_batch_size=None):
        batch_size = self.eval_batch_size if override_batch_size is None else override_batch_size
        return DataLoader(self.attack_dataset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)

    def on_save_checkpoint(self):
        return {'train_indices': self.train_indices, 'val_indices': self.val_indices}
    
    def on_load_checkpoint(self, checkpoint):
        self.train_indices = checkpoint.get('train_indices', None)
        self.val_indices = checkpoint.get('val_indices', None)