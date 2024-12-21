from torch.utils.data import Dataset

from .functional import *

class TempateAttack:
    def __init__(self, points_of_interest):
        self.points_of_interest = points_of_interest
    
    def profile(self, profiling_dataset: Dataset):
        pass
    
    def get_ranks(self, attack_dataset: Dataset):
        pass