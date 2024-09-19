from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from .common import *
from .soft_xor_layer import SoftXOR

class MultilayerPerceptron(Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        hidden_dims: int = 512,
        hidden_layers: int = 3,
        hidden_activation: Union[str, nn.Module] = nn.SELU,
        xor_output: bool = False
    ):
        if isinstance(hidden_activation, str):
            hidden_activation = getattr(nn, hidden_activation)
        super().__init__(**{key: val for key, val in locals().items() if key not in ('self', 'key', 'val')})
    
    def construct(self):
        modules = []
        in_dims = np.prod(self.input_shape)
        if self.hidden_layers > 0:
            out_dims = self.hidden_dims
            for layer_idx in range(self.hidden_layers):
                layer = nn.Linear(in_dims, out_dims)
                modules.append((f'layer_{layer_idx+1}', layer))
                modules.append((f'act_{layer_idx+1}', self.hidden_activation()))
                in_dims = out_dims
                out_dims = self.hidden_dims
        out_dims = self.output_classes
        modules.append((f'layer_{self.hidden_layers+1 if "layer_idx" in locals().keys() else 1}', nn.Linear(in_dims, out_dims) if not self.xor_output else SoftXOR(in_dims, int(np.log(out_dims)/np.log(2)))))
        self.model = nn.Sequential(OrderedDict(modules))
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.model(x)
        return x