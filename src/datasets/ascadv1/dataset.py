import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.aes import *

class ASCADv1(Dataset):
    def __init__(self,
        root=None,
        train=True,
        target_byte=2,
        target_values='subbytes',
        desync=0,
        variable_keys=False,
        raw_traces=False,
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.target_byte = np.arange(16) if target_byte == 'all' else np.array([target_byte]) if not hasattr(target_byte, '__len__') else target_byte
        self.target_values = [target_values] if isinstance(target_values, str) else target_values
        self.desync = desync
        self.variable_keys = variable_keys
        self.raw_traces = raw_traces
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        if raw_traces:
            raise NotImplementedError
        if variable_keys:
            raise NotImplementedError
        
        self.construct()
    
    def construct(self):
        if self.variable_keys:
            assert False
        else:
            self.data_path = os.path.join(self.root, f'ascadv1f_d{self.desync}.h5')
            if self.train:
                self.data_indices = np.arange(0, 50000)
            else:
                self.data_indices = np.arange(0, 10000)
        self.dataset_length = len(self.data_indices)
        self.traces, self.metadata = self._load_datapoints_from_disk(self.data_indices)
        eg_trace, _ = self.load_datapoints(0)
        self.data_shape = eg_trace.shape
        self.timesteps_per_trace = self.data_shape[-1]
    
    def _load_datapoints_from_disk(self, indices):
        with h5py.File(self.data_path) as _database_file:
            if self.train:
                database_file = _database_file['Profiling_traces']
            else:
                database_file = _database_file['Attack_traces']
            traces = np.array(database_file['traces'][indices, :], dtype=np.int8)
            if traces.ndim == 1:
                traces = traces[np.newaxis, :]
            else:
                traces = traces[:, np.newaxis, :]
            metadata = {
                'plaintext': np.array(database_file['metadata']['plaintext'][indices, :], dtype=np.uint8),
                'ciphertext': np.array(database_file['metadata']['ciphertext'][indices, :], dtype=np.uint8),
                'key': np.array(database_file['metadata']['key'][indices, :], dtype=np.uint8),
                'masks': np.array(database_file['metadata']['masks'][indices, :], dtype=np.uint8)
            }
        return traces, metadata
    
    def _load_datapoints_from_ram(self, indices):
        traces = self.traces[indices, :, :]
        metadata = {key: val[indices, :] for key, val in self.metadata.items()}
        return traces, metadata
    
    def load_datapoints(self, indices):
        return self._load_datapoints_from_ram(indices)
    
    def compute_target(self, metadata):
        key = metadata['key']
        plaintext = metadata['plaintext']
        masks = metadata['masks']
        if key.ndim > 1:
            batch_size = key.shape[0]
            assert plaintext.shape[0] == batch_size
            assert masks.shape[0] == batch_size
        else:
            batch_size = 1
            key = key[np.newaxis, :]
            plaintext = plaintext[np.newaxis, :]
            masks = masks[np.newaxis, :]
        assert all((key.shape[0] == batch_size, plaintext.shape[0] == batch_size, masks.shape[0] == batch_size))
        r_in = masks[:, -2, np.newaxis]
        r_out = masks[:, -1, np.newaxis]
        r = np.concatenate([np.zeros((batch_size, 2), dtype=np.uint8), masks[:, :-2]], axis=1)
        aux_metadata = {
            'subbytes': AES_SBOX[key ^ plaintext],
            'subbytes__r': AES_SBOX[key ^ plaintext] ^ r,
            'subbytes__r_out': AES_SBOX[key ^ plaintext] ^ r_out,
            'r_in': r_in,
            'r_out': r_out,
            'r': r
        }
        targets = []
        for target_val in self.target_values:
            if target_val == 'subbytes':
                target = aux_metadata['subbytes'][:, self.target_byte]
            elif target_val == 'subbytes__r':
                target = aux_metadata['subbytes__r'][:, self.target_byte]
            elif target_val == 'subbytes__r_out':
                target = aux_metadata['subbytes__r_out'][:, self.target_byte]
            elif target_val == 'r_in':
                target = aux_metadata['r_in']
            elif target_val == 'r_out':
                target = aux_metadata['r_out']
            elif target_val == 'r':
                target = aux_metadata['r'][:, self.target_byte]
            else:
                assert False
            targets.append(target.squeeze())
        return np.stack(targets, axis=1), aux_metadata
    
    def __getitem__(self, indices):
        indices = self.data_indices[indices]
        trace, metadata = self.load_datapoints(indices)
        target, aux_metadata = self.compute_target(metadata)
        metadata.update(aux_metadata)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return len(self.data_indices)