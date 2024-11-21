import os
import numpy as np
import h5py

class OneTruthPrevails(Dataset):
    def __init__(self,
        root=None,
        train=True,
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if self.train:
            self.traces = np.load(os.path.join(self.root, '1024', 'p.npy'), mmap_mode='r')
            self.labels = np.full(len(self.traces), -1, dtype=np.int8)
            with open(os.path.join(self.root, '1024', 'p_labels.txt'), 'r') as labels_file:
                for line_idx, line in enumerate(labels_file):
                    label = int(line[0])
                    self.labels[line_idx] = label
            assert np.all(self.labels != -1)
        else:
            self.traces = np.load(os.path.join(self.root, '1024', 'a.npy'))
            self.labels = np.full(len(self.traces), 2, dtypenp.int8)
            with open(os.path.join(self.root, '1024', 'a_labels.txt'), 'r') as labels_file:
                for line_idx, line in enumerate(labels_file):
                    label = int(line[0])
                    self.labels[line_idx] = label
            assert np.all(self.labels != -1)