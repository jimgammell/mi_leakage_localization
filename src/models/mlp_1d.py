from typing import *
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

class MultilayerPerceptron_1d(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        noise_conditional: bool = False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.noise_conditional = noise_conditional
        
        self.model = nn.Sequential(OrderedDict([
            *([('dropout_in', nn.Dropout(0.1))] if self.noise_conditional else []),
            ('dense_in', nn.Linear(2*np.prod(self.input_shape) if self.noise_conditional else np.prod(self.input_shape), 500)),
            ('act_in', nn.ReLU()),
            ('dropout_h1', nn.Dropout(0.2)),
            ('dense_h1', nn.Linear(500, 500)),
            ('act_h1', nn.ReLU()),
            ('dropout_h2', nn.Dropout(0.2)),
            ('dense_h2', nn.Linear(500, 500)),
            ('act_h2', nn.ReLU()),
            ('dropout_out', nn.Dropout(0.3)),
            ('dense_out', nn.Linear(500, self.output_classes))
        ]))
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