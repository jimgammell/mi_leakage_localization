from typing import *
import numpy as np
import torch
from torch import nn

from ..common import *

class ConvexNoiseCombiner(Module):
    def __init__(self,
            output_shape: Sequence[int],
            noise_mean: Optional[Union[float, Sequence[float]]] = None,
            noise_std: Optional[Union[float, Sequence[float]]] = None
    ):
        super().__init__(output_shape=output_shape)
        if noise_mean is None:
            noise_mean = np.zeros(output_shape, dtype=np.float32)
        elif isinstance(noise_mean, float):
            noise_mean = (noise_mean * np.ones(output_shape)).astype(np.float32)
        if noise_std is None:
            noise_std = np.ones(output_shape, dtype=np.float32)
        elif isinstance(noise_std, float):
            noise_std = (noise_std * np.ones(output_shape)).astype(np.float32)
        self.register_buffer('noise_mean', torch.as_tensor(noise_mean))
        self.register_buffer('noise_std', torch.as_tensor(noise_std))
    
    def construct(self):
        self.unsquashed_weights = nn.Parameter(torch.empty(1, *self.output_shape, dtype=torch.float), requires_grad=True)
    
    def init_weights(self):
        nn.init.constant_(self.unsquashed_weights, 0.)
    
    def forward(self, x):
        batch_size = x.size(0)
        squashed_weights = nn.functional.sigmoid(self.unsquashed_weights).expand(batch_size, *self.output_shape)
        noise = self.noise_std*squashed_weights + self.noise_mean
        x_obf = (1 - squashed_weights)*x + squashed_weights*noise
        return x_obf