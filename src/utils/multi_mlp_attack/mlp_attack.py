from typing import *
import numpy as np
import torch
from torch import nn

class MultiMLP(nn.Module):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.class_count = class_count
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        
        self.dense_1 = nn.Conv1d(
            self.timesteps_per_trace-self.window_size//2,
            self.hidden_dim*(self.timesteps_per_trace-self.window_size//2),
            kernel_size=self.window_size,
            groups=self.timesteps_per_trace-self.window_size//2
        )
        self.relu = nn.ReLU()
        self.dense_2 = nn.Conv1d(self.hidden_dim, self.class_count, kernel_size=1)
    
    def forward(self, x):
        batch_size, _, timestep_count = x.shape
        x = x.reshape(batch_size, timestep_count).unfold(-1, self.window_size, 1)
        x = self.dense_1(x).reshape(batch_size, self.hidden_dim, self.timesteps_per_trace-self.window_size//2)
        x = self.relu(x)
        x = self.dense_2(x).permute(0, 2, 1)
        return x

class MultiMLPAttack(nn.Module):
    def __init__(self,
        timesteps_per_trace: int,
        class_count: int,
        window_size: int = 5
    ):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.class_count = class_count
        self.window_size = window_size