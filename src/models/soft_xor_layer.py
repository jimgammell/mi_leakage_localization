import numpy as np
from numba import jit
import torch
from torch import nn

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.common import Module

@jit(nopython=True)
def get_oh_to_bits_table(bit_count=8):
    table = np.zeros((2**bit_count, bit_count), dtype=np.float32)
    for bit in range(bit_count):
        for oh in range(2**bit_count):
            if oh & (1<<bit) != 0:
                table[oh, bit] = 1.
    return table

@torch.compile
def _soft_xor(share_a_oh, share_b_oh, oh_to_bits_table):
    share_a_bits = share_a_oh @ oh_to_bits_table
    share_b_bits = share_b_oh @ oh_to_bits_table
    xor_probs_bits = share_a_bits*(1-share_b_bits) + (1-share_a_bits)*share_b_bits
    xor_probs_oh = ((xor_probs_bits.unsqueeze(1) ** oh_to_bits_table) * ((1-xor_probs_bits.unsqueeze(1)) ** (1-oh_to_bits_table))).prod(dim=-1).squeeze(-1)
    return xor_probs_oh

class SoftXOR(Module):
    def __init__(self, input_dims, bit_count=8):
        super().__init__(input_dims=input_dims, bit_count=bit_count)
    
    def construct(self):
        self.register_buffer('oh_to_bits_table', torch.as_tensor(get_oh_to_bits_table(self.bit_count), dtype=torch.float))
        self.share_a_predictor = nn.Sequential(
            nn.Linear(self.input_dims, 2**self.bit_count, bias=False),
            nn.Softmax(dim=-1)
        )
        self.share_b_predictor = nn.Sequential(
            nn.Linear(self.input_dims, 2**self.bit_count, bias=False),
            nn.Softmax(dim=-1)
        )
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.share_a_predictor[0].weight)
        nn.init.xavier_uniform_(self.share_b_predictor[0].weight)
    
    def forward(self, x):
        share_a = self.share_a_predictor(x)
        share_b = self.share_b_predictor(x)
        return _soft_xor(share_a, share_b, self.oh_to_bits_table).log()

if __name__ == '__main__':
    _test()