import os
import yaml
import argparse

from common import *
from training_modules.adversarial_leakage_localization import AdversarialLeakageLocalizationTrainer
from trials.trial import Trial

AVAILABLE_DATASETS = [x.split('.')[0] for x in os.listdir(CONFIG_DIR) if x.endswith('.yaml') and not(x in ['default_config.yaml', 'global_variables.yaml'])]
with open(os.path.join(CONFIG_DIR, 'default_config.yaml'), 'r') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    parser.add_argument('--seed-count', type=int, default=1, action='store')
    parser.add_argument('--lr-count', type=int, default=20, action='store')
    parser.add_argument('--lambda-count', type=int, default=20, action='store')
    parser.add_argument('--trial-dir', default=None, action='store')
    clargs = parser.parse_args()
    dataset = clargs.dataset
    seed_count = clargs.seed_count
    trial_dir = os.path.join(OUTPUT_DIR, dataset if clargs.trial_dir is None else clargs.trial_dir)
    assert seed_count > 0
    
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
    if 'classifiers_kwargs' in all_kwargs['default_training_module_kwargs']:
        all_kwargs['default_training_module_kwargs']['classifiers_kwargs']['input_shape'] = (1, profiling_dataset.timesteps_per_trace)
    else:
        all_kwargs['default_training_module_kwargs']['classifiers_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
    trial = Trial(base_dir=trial_dir, profiling_dataset=profiling_dataset, attack_dataset=attack_dataset, supervised_classifier_kwargs=supervised_classifier_kwargs, all_kwargs=all_kwargs)
    trial.all_theta_lr_sweep(np.logspace(-6, -2, clargs.lr_count))
    #trial.all_theta_smith_lr_sweep()

if __name__ == '__main__':
    main()