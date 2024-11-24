import os
import argparse

from common import *
from trials.trial import Trial

AVAILABLE_DATASETS = [
    'ASCADv1-fixed',
    'ASCADv1-variable',
    'DPAv4',
    'AES_HD',
    'AES_PTv2',
    'OTiAiT',
    'OTP'
]
DATA_DIR = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads')

set_verbosity(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    parser.add_argument('--seed-count', type=int, default=5, action='store')
    clargs = parser.parse_args()
    dataset = clargs.dataset
    seed_count = clargs.seed_count
    assert seed_count > 0
    
    supervised_classifier_kwargs = dict(
        model_name='sca-cnn',
        optimizer_name='AdamW',
        optimizer_kwargs={'lr': 2e-4}
    )
    all_style_classifier_kwargs = dict(
        classifier_name='sca-cnn',
        classifier_optimizer_name='AdamW',
        classifier_optimizer_kwargs={'lr': 2e-4},
        obfuscator_optimizer_name='AdamW',
        obfuscator_optimizer_kwargs={'lr': 1e-2},
        obfuscator_batch_size_multiplier=8,
        obfuscator_l2_norm_penalty=0.5*np.log(256)
    )
    
    if dataset == 'ASCADv1-fixed':
        from datasets.ascadv1 import ASCADv1_DataModule
        data_module = ASCADv1_DataModule(root=os.path.join(DATA_DIR, 'ascadv1'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 100
    elif dataset == 'ASCADv1-variable':
        from datasets.ascadv1 import ASCADv1_DataModule
        data_module = ASCADv1_DataModule(root=os.path.join(DATA_DIR, 'ascadv1'), dataset_kwargs={'variable_keys': True})
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.profiling_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 100
    elif dataset == 'DPAv4':
        from datasets.dpav4 import DPAv4_DataModule
        data_module = DPAv4_DataModule(root=os.path.join(DATA_DIR, 'dpav4'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 25
    elif dataset == 'AES_HD':
        from datasets.aes_hd import AES_HD_DataModule
        data_module = AES_HD_DataModule(root=os.path.join(DATA_DIR, 'aes_hd'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_dataset.timesteps_per_trace),
            'head_kwargs': {'hidden_dims': 64}
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 100
    elif dataset == 'AES_PTv2':
        from datasets.aes_pt_v2 import AES_PTv2
        raise NotImplementedError
    elif dataset == 'OTiAiT':
        from datasets.ed25519_wolfssl import ED25519_DataModule
        data_module = ED25519_DataModule(root=os.path.join(DATA_DIR, 'one_trace_is_all_it_takes'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_dataset.timesteps_per_trace),
            'output_classes': 16
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 10
    elif dataset == 'OTP':
        from datasets.one_truth_prevails import OneTruthPrevails_DataModule
        data_module = OneTruthPrevails_DataModule(root=os.path.join(DATA_DIR, 'one_truth_prevails'), train_prop=0.999, val_prop=0.001)
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_dataset.timesteps_per_trace),
            'output_classes': 2
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 1
    else:
        assert False
    
    trial = Trial(
        base_dir=os.path.join(OUTPUT_DIR, f'{dataset}_results'),
        profiling_dataset=profiling_dataset,
        attack_dataset=attack_dataset,
        data_module=data_module,
        epoch_count=epoch_count,
        seed_count=seed_count,
        default_supervised_classifier_kwargs=supervised_classifier_kwargs,
        default_all_style_classifier_kwargs=all_style_classifier_kwargs
    )
    trial.compute_random_baseline()
    trial.compute_first_order_baselines()
    trial.supervised_lr_sweep(classifier_learning_rates)
    trial.train_optimal_supervised_classifier()
    trial.train_optimal_all_classifier()
    trial.compute_neural_net_explainability_baselines()
    trial.eval_leakage_assessments(template_attack=dataset in ['DPAv4', 'AES_HD'])
    trial.plot_everything()

if __name__ == '__main__':
    main()