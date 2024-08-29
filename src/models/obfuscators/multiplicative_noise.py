from typing import *
import numpy as np
import torch
from torch import nn

from ..common import *

class MultiplicativeNoiseCombiner(Module):
    def __init__(self, output_shape: Sequence[int], ret_noise: bool = True):
        super().__init__(**{key: val for key, val in locals().items() if key not in ('self', 'key', 'val')})
    
    def construct(self):
        self.unsquashed_weights = nn.Parameter(torch.empty(1, *self.output_shape, dtype=torch.float), requires_grad=True)
    
    def init_weights(self):
        nn.init.constant_(self.unsquashed_weights, 0.)
    
    def forward(self, x):
        batch_size = x.size(0)
        binary_noise = nn.functional.sigmoid(self.unsquashed_weights).expand(batch_size, *self.output_shape).bernoulli()
        x_obf = binary_noise*x
        if self.ret_noise:
            x_obf = torch.cat([x_obf, binary_noise.detach()], dim=1)
        return x_obf