import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class OneTruthPrevails(Dataset):
    def __init__(self,
        root=None,
        train=True,
        transform=None,
        target_transform=None,
        mmap_profiling_dataset=False
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        self.mmap_profiling_dataset = mmap_profiling_dataset
        
        if self.train:
            kwargs = {'mmap_mode': 'r'} if self.mmap_profiling_dataset else {}
            self.traces = np.load(os.path.join(self.root, '1024', 'p.npy'), **kwargs)
            self.labels = np.loadtxt(os.path.join(self.root, '1024', 'p_labels.txt'), dtype=np.uint8)
        else:
            self.traces = np.load(os.path.join(self.root, '1024', 'a.npy'))
            self.labels = np.load(os.path.join(self.root, '1024', 'a_labels.npy')).astype(np.uint8)
        self.dataset_length = len(self.traces)
        assert self.dataset_length == len(self.labels)
        self.data_shape = self.traces[0, np.newaxis, :].shape
        self.timesteps_per_trace = np.prod(self.data_shape)
    
    def __getitem__(self, indices):
        trace = self.traces[indices, np.newaxis, :].astype(np.float32)
        label = self.labels[indices]
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_metadata:
            return trace, label, {'label': label}
        else:
            return trace, label
    
    def __len__(self):
        return self.dataset_length