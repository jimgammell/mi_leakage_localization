from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class MultilayerPerceptron_1d(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        layer_count: int = 3,
        hidden_dim: int = 500,
        noise_conditional: bool = False,
        use_dropout: bool = False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.layer_count = layer_count
        self.hidden_dim = hidden_dim
        self.noise_conditional = noise_conditional
        self.use_dropout = use_dropout
        
        modules = []
        if self.use_dropout and not(self.noise_conditional):
            modules.append(('dropout_in', nn.Dropout(0.1)))
        modules.append(('dense_in', nn.Linear(2*np.prod(self.input_shape) if self.noise_conditional else np.prod(self.input_shape), self.hidden_dim)))
        modules.append(('act_in', nn.ReLU()))
        for layer_num in range(1, self.layer_count+1):
            if self.use_dropout:
                modules.append((f'dropout_h{layer_num}', nn.Dropout(0.2)))
                modules.append((f'dense_h{layer_num}', nn.Linear(self.hidden_dim, self.hidden_dim)))
                modules.append((f'act_h{layer_num}', nn.ReLU()))
        if self.use_dropout:
            modules.append(('dropout_out', nn.Dropout(0.3)))
        modules.append(('dense_out', nn.Linear(self.hidden_dim, self.output_classes)))
        self.model = nn.Sequential(OrderedDict(modules))
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0.01)
    
    def forward(self, *args):
        if self.noise_conditional:
            (x, noise) = args
            x = torch.cat([x, noise], dim=1)
        else:
            (x,) = args
        x = x.flatten(start_dim=1)
        x = self.model(x)
        return x