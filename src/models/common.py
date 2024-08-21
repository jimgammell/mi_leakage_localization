from typing import *
import numpy as np
import torch
from torch import nn

class Module(nn.Module):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.construct()
        self.init_weights()
        if hasattr(self, 'input_shape'):
            self.output_shape = self(torch.randn(1, *self.input_shape)).shape[1:]
    
    def construct(self):
        raise NotImplementedError
    
    def init_weights(self):
        pass

    def extra_repr(self):
        if not hasattr(self, 'param_count'):
            self.param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rv = []
        if hasattr(self, 'input_shape') and hasattr(self, 'output_shape'):
            rv.append(f'Input: {self.input_shape} -> Output: {self.output_shape}')
        rv.append(f'Parameter count: {self.param_count}')
        rv = '\n'.join(rv)
        return rv