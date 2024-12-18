import os
import yaml
import argparse

from common import *
from utils.flatten_dict import flatten_dict, unflatten_dict
from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer
from trials.trial import Trial

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
        trial.leaky_point_count_sweep()
        trial.sigma_sweep()
    else:
        supervised_classifier_kwargs = DEFAULT_CONFIG['supervised_classifier_kwargs']
        all_kwargs = DEFAULT_CONFIG['all_kwargs']
        with open(os.path.join(CONFIG_DIR, f'{dataset}.yaml'), 'r') as f:
            trial_config = yaml.load(f, Loader=yaml.FullLoader)
        all_kwargs.update(trial_config['all_kwargs'])
        if dataset == 'aes_hd':
            from datasets.aes_hd import AES_HD as DatasetClass
        elif dataset == 'aes_ptv2_multi':
            from datasets.aes_pt_v2 import AES_PTv2 as DatasetClass
        elif dataset == 'aes_ptv2_single':
            from datasets.aes_pt_v2 import AES_PTv2 as DatasetClass
        elif dataset == 'ascadv1_fixed':
            from datasets.ascadv1 import ASCADv1 as DatasetClass
        elif dataset == 'ascadv1_variable':
            from datasets.ascadv1 import ASCADv1 as DatasetClass
        elif dataset == 'dpav4':
            from datasets.dpav4 import DPAv4 as DatasetClass
        elif dataset == 'otiait':
            from datasets.ed25519_wolfssl import ED25519 as DatasetClass
        elif dataset == 'otp':
            from datasets.one_truth_prevails import OneTruthPrevails as DatasetClass
        else:
            assert False, f'Dataset `{dataset}` is not implemented. Available choices: `{"`, `".join(AVAILABLE_DATASETS)}`.'
        profiling_dataset = DatasetClass(root=trial_config['data_dir'], train=True)
        attack_dataset = DatasetClass(root=trial_config['data_dir'], train=False)
        from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer
        kwargs = trial_config['all_kwargs']
        kwargs.update({'timesteps_per_trace': profiling_dataset.timesteps_per_trace})
        trainer = AdversarialLeakageLocalizationTrainer(profiling_dataset, attack_dataset, max_epochs=trial_config['max_epochs'], default_training_module_kwargs=kwargs)
        trainer.train_gamma(trial_dir)
    
    #if 'classifiers_kwargs' in all_kwargs['default_training_module_kwargs']:
    #    all_kwargs['default_training_module_kwargs']['classifiers_kwargs']['input_shape'] = (1, profiling_datasets[0].timesteps_per_trace)
    #else:
    #    all_kwargs['default_training_module_kwargs']['classifiers_kwargs'] = {'input_shape': (1, profiling_datasets[0].timesteps_per_trace)}

if __name__ == '__main__':
    main()