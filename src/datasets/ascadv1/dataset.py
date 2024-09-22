import torch
from torch.utils.data import Dataset

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
        self.target_byte = target_byte
        self.target_values = target_values
        self.desync = desync
        self.variable_keys = variable_keys
        self.raw_traces = raw_traces
        self.transform = transform
        self.target_transform = target_transform