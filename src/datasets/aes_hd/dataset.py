import os
import numpy as np
from numba import jit
import h5py
import torch
from torch.utils.data import Dataset

from utils.aes import *

@jit(nopython=True)
def to_key_preds(int_var_preds, args, constants=None):
    if args.ndim == 1:
        ciphertext_11 = args[0]
        ciphertext_7 = args[1]
    elif args.ndim == 2:
        ciphertext_11 = args[:, 0]
        ciphertext_7 = args[:, 1]
    return int_var_preds[AES_INVERSE_SBOX[np.arange(256, dtype=np.uint8) ^ ciphertext_11] ^ ciphertext_7]

class AES_HD(Dataset):
    def __init__(self,
        root=None,
        train=True,
        target_values='last_state',
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.target_values = [target_values] if isinstance(target_values, str) else target_values
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        self.construct()
    
    def construct(self):
        if self.train:
            self.traces = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_traces_AES_HD.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_labels_AES_HD.npy')).astype(np.uint8)
            self.ciphertexts = np.load(os.path.join(self.root, 'AES_HD_dataset', 'profiling_ciphertext_AES_HD.npy')).astype(np.uint8)
        else:
            self.traces = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_traces_AES_HD.npy')).astype(np.float32)
            self.targets = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_labels_AES_HD.npy')).astype(np.uint8)
            self.ciphertexts = np.load(os.path.join(self.root, 'AES_HD_dataset', 'attack_ciphertext_AES_HD.npy')).astype(np.uint8)
        self.metadata = {
            'ciphertext': self.ciphertexts,
            'ciphertext_11': self.ciphertexts[:, 11],
            'ciphertext_7': self.ciphertexts[:, 7],
            'last_state': self.targets,
            'key': np.zeros_like(self.targets)
        }
        self.dataset_length = len(self.traces)
        assert self.dataset_length == len(self.targets) == len(self.ciphertexts)
        self.data_shape = self.traces[0].shape
        self.timesteps_per_trace = self.data_shape[-1]
    
    def __getitem__(self, indices):
        trace = self.traces[indices, np.newaxis, :]
        target = self.targets[indices].squeeze()
        metadata = {key: val[indices].squeeze() for key, val in self.metadata.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return self.dataset_length