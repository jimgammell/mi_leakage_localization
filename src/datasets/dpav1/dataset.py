import os
import numpy as np
from torch.utils.data import Dataset

class DPAv1(Dataset):
    def __init__(self,
        root=None,
        target_byte=0,
        target_values='subbytes',
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.target_byte = target_byte
        self.target_values = target_values
        self.transform = transform
        self.target_transform = target_transform

        base_dir = os.path.join(self.root, 'secmatv1_2006_04_0809', 'secmatv1_2006_04_0809')
        self.traces = []
        self.keys = []
        self.plaintexts = []
        self.ciphertexts = []
        