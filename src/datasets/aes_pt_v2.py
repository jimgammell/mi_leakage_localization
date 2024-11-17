from typing import *
import h5py
import numpy as np
from numba import jit
import torch
from torch.data.utils import Dataset

from utils.aes import *

VALID_DEVICES = Literal['D1', 'D2', 'D3', 'D4', 'Pinata']
VALID_COUNTERMEASURES = Literal['MS1', 'MS2', 'Unprotected']

@jit(nopython=True)
def to_key_preds(int_var_preds, plaintext, constants=None):
    return int_var_preds[AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintext]]

class AES_PTv2(Dataset):
    def __init__(self,
        root: str,
        train: bool = True,
        devices: Union[VALID_DEVICES, Sequence[VALID_DEVICES]] = 'D1'
        countermeasure: VALID_COUNTERMEASURES = 'Unprotected',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = root
        self.devices = devices
        self.countermeasure = countermeasure
        self.transform = transform
        self.target_transform = target_transform
        traces, labels, metadata = [], [], []
        for device in self.devices:
            path = os.path.join(self.root, f'AES_PTv2_{device}.h5')
            with h5py.File(path) as database_file:
                if self.train:
                    database_file = database_file['Profiling']
                else:
                    database_file = database_file['Attack']
                _traces = np.array(database_file['Traces'], dtype=np.float32)
                _labels = np.array(database_file['Labels'], dtype=np.uint8)
                _metadata = {key: np.array(val, dtype=np.uint8) for key, val in database_file['MetaData'].items()}
                traces.append(_traces)
                labels.append(_labels)
                metadata.append(_metadata)
        self.traces = np.concatenate(traces, axis=0)[:, np.newaxis, :]
        self.labels = np.concatenate(labels, axis=0)
        self.metadata = {key: np.concatenate([_metadata[key] for _metadata in metadata], axis=0) for key in metadata[0].keys()}
        self.length = self.traces.shape[0]
        assert self.length == labels.shape[0]
        assert all(self.length == val.shape[0] for val in self.metadata.values())
        self.return_metadata = False

    def compute_target(self, metadata):
        key = metadata['key']
        plaintext = metadata['plaintext']
        if self.countermeasure != 'Unprotected':
            r_in = metadata['masks'][0]
            r_out = metadata['masks'][1]
        

    def __getitem__(self, indices):
        trace = self.traces[indices, ...]
        label = self.labels[indices]
        metadata = {key: val[indices] for key, val in self.metadata.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            label = self.transform(label)
        if self.return_metadata:
            return trace, label, metadata
        else:
            return trace, label

    def __len__(self):
        return self.length