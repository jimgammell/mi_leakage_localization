from typing import *
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

class GaussianDataset(Dataset):
    def generate_datapoint(self):
        y = np.random.choice(2)
        y_tilde = np.random.choice(2)
        x_dep = np.random.standard_normal() +  2*np.float32(y) - 1
        x_ind = np.random.standard_normal() + 2*np.float32(y_tilde) - 1
        x = np.stack([x_dep, x_ind])
        return x, y
    
    def __getitem__(self, _):
        x, y = self.generate_datapoint()
        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.long)
        return x, y
    
    def __len__(self):
        return 8192

class DataModule(L.LightningDataModule):
    def __init__(self,
        train_batch_size: int,
        eval_batch_size: int,
        dataloader_kwargs: dict = {}
    ):
        for key, val in locals().items():
            if key not in ('key', 'val', 'self'):
                setattr(self, key, val)
        super().__init__()
    
    def setup(self, *args, **kwargs):
        self.train_dataset = GaussianDataset()
        self.val_dataset = GaussianDataset()
        self.test_dataset = GaussianDataset()
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = os.cpu_count()//2
    
    def _dataloader(self, prefix: Literal['train', 'val', 'test'], override_batch_size: Optional[int] = None):
        dataset = getattr(self, f'{prefix}_dataset')
        batch_size = (self.train_batch_size if prefix=='train' else self.eval_batch_size) if override_batch_size is None else override_batch_size
        shuffle = True if prefix == 'train' else False
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, **self.dataloader_kwargs)
    
    def train_dataloader(self, **kwargs): return self._dataloader('train', **kwargs)
    def val_dataloader(self, **kwargs): return self._dataloader('val', **kwargs)
    def test_dataloader(self, **kwargs): return self._dataloader('test', **kwargs)