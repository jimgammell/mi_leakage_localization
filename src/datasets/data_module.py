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

class MultiDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
    
    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        return self

    def __next__(self):
        batches = []
        for idx, (iterator, dataloader) in enumerate(zip(self.iterators, self.dataloaders)):
            try:
                batch = next(iterator)
            except StopIteration:
                self.iterators[idx] = iter(dataloader)
                batch = next(self.iterators[idx])
            batches.append(batch)
        return (val for batch in batches for val in batch)
    
    def __len__(self):
        return max(len(dataloader) for dataloader in self.dataloaders)

class DataModule(L.LightningDataModule):
    def __init__(self,
        profiling_dataset,
        attack_dataset,
        val_prop: float = 0.2,
        train_batch_size: int = 256,
        aug_train_batch_size: int = 2048,
        eval_batch_size: int = 2048,
        adversarial_mode: bool = False,
        data_mean: Optional[Union[float, Sequence[float]]] = None,
        data_var: Optional[Union[float, Sequence[float]]] = None,
        dataloader_kwargs: dict = {},
        aug_transform: Optional[Callable] = None
    ):
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
        indices = np.random.choice(len(self.profiling_dataset), len(self.profiling_dataset), replace=False)
        self.train_indices = indices[:self.train_length]
        self.val_indices = indices[self.train_length:]
        self.train_dataset = Subset(copy(self.profiling_dataset), self.train_indices)
        self.val_dataset = Subset(copy(self.profiling_dataset), self.val_indices)
        set_transforms(self.train_dataset, self.data_transform, self.target_transform)
        set_transforms(self.val_dataset, self.data_transform, self.target_transform)
        set_transforms(self.test_dataset, self.data_transform, self.target_transform)
        if self.adversarial_mode:
            if self.aug_transform is not None:
                self.aug_data_transform = transforms.Compose(basic_transform_mods + [self.aug_transform])
            else:
                self.aug_data_transform = self.data_transform
            self.aug_train_dataset = copy(self.train_dataset)
            set_transforms(self.aug_train_dataset, self.aug_data_transform, self.target_transform)
        dataloader_kwargs = {
            'num_workers': max(os.cpu_count//2, 1),
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
            return MultiDataLoader([train_dataloader, aug_dataloader])
        else:
            return train_dataloader
    
    def val_dataloader(self, override_batch_size=None):
        batch_size = self.eval_batch_size if override_batch_size is None else override_batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)
    
    def test_dataloader(self, override_batch_size=None):
        batch_size = self.eval_batch_size if override_batch_size is None else override_batch_size
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, **self.dataloader_kwargs)