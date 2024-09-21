from typing import *
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from multiprocessing import pool
from scipy.stats import kendalltau

from .functional import *
from common import *
from utils.template_attack import TemplateAttack

class MeasurePerformanceCorrelation:
    def __init__(self,
        leakage_measurements,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        attack_type: Literal['template-attack'] = 'template-attack',
        target_keys: Union[str, Sequence[str]] = 'subbytes',
        target_bytes: Optional[Union[int, Sequence[int]]] = None
    ):
        self.leakage_measurements = leakage_measurements
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.attack_type = attack_type
        self.target_keys = [target_keys] if isinstance(target_keys, str) else target_keys
        self.target_bytes = [target_bytes] if (isinstance(target_bytes, int) or (target_bytes is None)) else target_bytes
    
    def measure_performance(self,
            poi_count: int = 10, seed_count: int = 1, attack_seed_count: int = 1000,
            dataset_size: Optional[int] = None, worker_count: Optional[int] = 0
    ):
        measurement_count = len(self.leakage_measurements)
        performance_metrics = np.full((attack_seed_count, measurement_count-poi_count+1), np.nan, dtype=np.float32)
        ranking = self.leakage_measurements.argsort()
        for measurement_idx in range(measurement_count-poi_count+1):
            points_of_interest = ranking[measurement_idx:measurement_idx+poi_count]
            template_attack = TemplateAttack(points_of_interest, target_key='subbytes')
            template_attack.profile(self.profiling_dataset)
            rank_over_time = template_attack.attack(self.attack_dataset, n_repetitions=attack_seed_count, n_traces=1)
            performance_metrics[:, measurement_idx] = rank_over_time.mean(axis=-1)
        correlation = soft_kendall_tau(performance_metrics, np.arange(performance_metrics.shape[1]))
        return correlation, performance_metrics