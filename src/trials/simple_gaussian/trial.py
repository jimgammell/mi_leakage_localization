from typing import *
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset

from common import *
from .gmm_sequence import *

class Trial:
    def __init__(self,
        base_dir: Union[str, os.PathLike]
    ):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def numerical_experiments(self):
        output_dir = os.path.join(self.base_dir, 'numerical_experiments')
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, 'conditional_mi_sweep.pickle')):
            with open(os.path.join(output_dir, 'conditional_mi_sweep.pickle'), 'rb') as f:
                results = pickle.load(f)
        else:
            results = sweep_conditional_mi()
            with open(os.path.join(output_dir, 'conditional_mi_sweep.pickle'), 'wb') as f:
                pickle.dump(results, f)
        fig = plot_conditional_mi(results)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'conditional_mi_sweep.pdf'))
    
    def snr_sweep(self):
        pass
    
    def leaky_point_count_sweep(self):
        pass
    
    def dataset_size_sweep(self):
        pass
    
    def leakage_order_sweep(self):
        pass