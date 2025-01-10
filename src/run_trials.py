import os
from copy import copy
import yaml
import argparse
import time
from torch.utils.data import DataLoader

from common import *
from utils.flatten_dict import flatten_dict, unflatten_dict
from training_modules.supervised_deep_sca import SupervisedTrainer
from training_modules.cooperative_leakage_localization import LeakageLocalizationTrainer
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from trials.utils import *
from utils.gmm_performance_correlation import GMMPerformanceCorrelation
from trials.real_dataset_baseline_comparison import Trial as RealBaselineComparisonTrial

AVAILABLE_DATASETS = [x.split('.')[0] for x in os.listdir(CONFIG_DIR) if x.endswith('.yaml') and not(x in ['default_config.yaml', 'global_variables.yaml'])]
with open(os.path.join(CONFIG_DIR, 'default_config.yaml'), 'r') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

SYNTHETIC_LEAKAGE_TYPES = ['1o', '2o', '12o', 'shuffling', 'no_ops', 'multi_1o']
GAUSSIAN_LEAKAGE_TYPES = ['1o-sweep']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    parser.add_argument('--seed-count', type=int, default=1, action='store')
    parser.add_argument('--lr-count', type=int, default=20, action='store')
    parser.add_argument('--lambda-count', type=int, default=20, action='store')
    parser.add_argument('--trial-dir', default=None, action='store')
    parser.add_argument('--leakage-type', default=None, action='store', choices=SYNTHETIC_LEAKAGE_TYPES+GAUSSIAN_LEAKAGE_TYPES)
    parser.add_argument('--override-theta-pretrain-path', default=None, action='store')
    clargs = parser.parse_args()
    dataset = clargs.dataset
    seed_count = clargs.seed_count
    if dataset in ['synthetic', 'simple-gaussian']:
        leakage_type = clargs.leakage_type
        assert (dataset in SYNTHETIC_LEAKAGE_TYPES) if (dataset == 'synthetic') else (dataset in GAUSSIAN_LEAKAGE_TYPES) if (dataset == 'gaussian') else False
        trial_dir = os.path.join(OUTPUT_DIR, f'{dataset}_{leakage_type}' if clargs.trial_dir is None else clargs.trial_dir)
    else:
        assert clargs.leakage_type is None
        trial_dir = os.path.join(OUTPUT_DIR, dataset if clargs.trial_dir is None else clargs.trial_dir)
    assert seed_count > 0
    
    if dataset == 'synthetic':
        from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
        kwargs = {'lpf_beta': 0.9}
        if leakage_type == '1o':
            pass
        elif leakage_type == '2o':
            kwargs.update({'leaking_timestep_count_1o': 0, 'leaking_timestep_count_2o': 1})
        elif leakage_type == '12o':
            kwargs.update({'leaking_timestep_count_2o': 1})
        elif leakage_type == 'shuffling':
            kwargs.update({'shuffle_locs': 16})
        elif leakage_type == 'no_ops':
            kwargs.update({'max_no_ops': 16})
        elif leakage_type == 'multi_1o':
            kwargs.update({'leaking_timestep_count_1o': 16})
        else:
            assert False
        profiling_datasets = [SyntheticAES(epoch_length=10000, timesteps_per_trace=250, **kwargs)]
        attack_datasets = [SyntheticAESLike(profiling_dataset, epoch_length=10000, fixed_key=0) for profiling_dataset in profiling_datasets]
    elif dataset == 'simple_gaussian':
        from trials.simple_gaussian import Trial
        trial = Trial(trial_dir)
        trial.numerical_experiments()
        trial.sigma_sweep()
        trial.leaky_point_count_sweep()
    else:
        with open(os.path.join(CONFIG_DIR, f'{dataset}.yaml'), 'r') as f:
            trial_config = yaml.load(f, Loader=yaml.FullLoader)
        trial = RealBaselineComparisonTrial(
            dataset_name=dataset,
            trial_config=trial_config,
            seed_count=seed_count,
            logging_dir=trial_dir
        )
        trial()

if __name__ == '__main__':
    main()