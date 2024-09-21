from typing import *
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import numpy as np
from numba import jit
from scipy.stats import kendalltau

from utils.template_attack import TemplateAttack
from utils.template_attack.functional import *

@jit(nopython=True)
def get_per_timestep_advantage(poi_choices, performance, timestep_count):
    advantages = np.full((timestep_count,), np.nan, dtype=np.float32)
    for timestep in range(timestep_count):
        pos_perf = np.full_like(performance, np.nan)
        pos_count = 0
        neg_perf = np.full_like(performance, np.nan)
        neg_count = 0
        for poi_choice, perf in zip(poi_choices, performance):
            if timestep in poi_choice:
                pos_perf[pos_count] = perf
                pos_count += 1
            else:
                neg_perf[neg_count] = perf
                neg_count += 1
        advantages[timestep] = np.mean(pos_perf[:pos_count]) - np.mean(neg_perf[:neg_count])
    assert np.all(np.isfinite(advantages))
    return advantages

class AdvantageCorrelation:
    def __init__(self,
        leakage_measurements,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        target_keys: Union[str, Sequence[str]] = 'subbytes',
        target_bytes: Optional[Union[int, Sequence[int]]] = None
    ):
        self.leakage_measurements = leakage_measurements
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.target_keys = target_keys
        self.target_bytes = target_bytes
    
    def measure_performance(self,
        poi_count: int = 10,
        attack_count: int = 100
    ):
        poi_choices = np.zeros((attack_count, poi_count), dtype=int)
        performance = np.full((attack_count,), np.nan, dtype=np.float32)
        for attack_idx in tqdm(range(attack_count)):
            pois = np.random.choice(len(self.leakage_measurements), poi_count, replace=False)
            template_attack = TemplateAttack(pois, target_key='subbytes')
            template_attack.profile(self.profiling_dataset)
            rank_over_time = template_attack.attack(self.attack_dataset, n_repetitions=1000, n_traces=1000)
            poi_choices[attack_idx, :] = pois
            performance[attack_idx] = rank_over_time.mean()
        advantages = get_per_timestep_advantage(poi_choices, performance, len(self.leakage_measurements))
        correlation = kendalltau(advantages, self.leakage_measurements).statistic
        return correlation