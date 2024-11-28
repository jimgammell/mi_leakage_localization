import os
import argparse
from torch.utils.data import ConcatDataset

from common import *
from trials.trial import Trial
from trials.portability_trial import PortabilityTrial

AVAILABLE_DATASETS = [
    'ASCADv1-fixed',
    'ASCADv1-variable',
    'DPAv4',
    'AES_HD',
    'AES_PTv2-single',
    'AES_PTv2-multi',
    'OTiAiT',
    'OTP'
]
DATA_DIR = os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads')
LAMBDA_SWEEP_COUNT = 5

set_verbosity(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=AVAILABLE_DATASETS)
    parser.add_argument('--seed-count', type=int, default=5, action='store')
    parser.add_argument('--plot-only', action='store_true', default=False)
    parser.add_argument('--long-lambda-sweep', default=False, action='store_true')
    parser.add_argument('--lambda-idx', default=None, type=int, action='store')
    clargs = parser.parse_args()
    dataset = clargs.dataset
    seed_count = clargs.seed_count
    assert seed_count > 0
    
    supervised_classifier_kwargs = dict(
        model_name='sca-cnn',
        optimizer_name='AdamW',
        optimizer_kwargs={'lr': 2e-4},
        additive_noise_augmentation=0.25
    )
    all_style_classifier_kwargs = dict(
        classifier_name='sca-cnn',
        classifier_optimizer_name='AdamW',
        classifier_optimizer_kwargs={'lr': 2e-4},
        obfuscator_optimizer_name='AdamW',
        obfuscator_optimizer_kwargs={'lr': 1e-3},
        obfuscator_batch_size_multiplier=8,
        obfuscator_l2_norm_penalty=0.5*np.log(256),
        additive_noise_augmentation=0.25
    )
    
    if dataset == 'ASCADv1-fixed':
        from datasets.ascadv1 import ASCADv1_DataModule
        data_module = ASCADv1_DataModule(root=os.path.join(DATA_DIR, 'ascadv1'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_dataset.timesteps_per_trace),
            'head_kwargs': {'xor_output': True}
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        lambda_vals = np.log(256)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 100
        obf_epoch_count = 100
        poi_count = 10
    elif dataset == 'ASCADv1-variable':
        from datasets.ascadv1 import ASCADv1_DataModule
        data_module = ASCADv1_DataModule(root=os.path.join(DATA_DIR, 'ascadv1'), dataset_kwargs={'variable_keys': True})
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.profiling_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
        classifier_learning_rates = np.logspace(-6, -2, 25)
        lambda_vals = np.log(256)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 100
        obf_epoch_count = 100
        poi_count = 10
    elif dataset == 'DPAv4':
        from datasets.dpav4 import DPAv4_DataModule
        data_module = DPAv4_DataModule(root=os.path.join(DATA_DIR, 'dpav4'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {'input_shape': (1, profiling_dataset.timesteps_per_trace)}
        classifier_learning_rates = np.logspace(-6, -2, 25)
        lambda_vals = np.log(256)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 25
        obf_epoch_count = 250
        poi_count = 5
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
        lambda_vals = np.log(256)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 100
        poi_count = 10
    elif dataset == 'AES_PTv2-single':
        from datasets.aes_pt_v2 import AES_PTv2, AES_PTv2_DataModule
        profiling_datasets = [AES_PTv2(root=os.path.join(DATA_DIR, 'aes_pt_v2'), train=True, devices=f'D{dev_id}', countermeasure='Unprotected') for dev_id in [1, 2, 3, 4]]
        attack_datasets = [AES_PTv2(root=os.path.join(DATA_DIR, 'aes_pt_v2'), train=False, devices=f'D{dev_id}', countermeasure='Unprotected') for dev_id in [1, 2, 3, 4]]
        data_modules = [AES_PTv2_DataModule(profiling_dataset, attack_dataset) for profiling_dataset, attack_dataset in zip(profiling_datasets, attack_datasets)]
        for data_module in data_modules:
            data_module.setup('')
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_datasets[0].timesteps_per_trace)
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        lambda_vals = np.log(256)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 10
        obf_epoch_count = 100
        seed_count = 1
        poi_count = 5
    elif dataset == 'AES_PTv2-multi':
        from datasets.aes_pt_v2 import AES_PTv2, AES_PTv2_DataModule
        profiling_datasets = [AES_PTv2(root=os.path.join(DATA_DIR, 'aes_pt_v2'), train=True, devices=f'D{dev_id}') for dev_id in [1, 2, 3, 4]]
        attack_datasets = [AES_PTv2(root=os.path.join(DATA_DIR, 'aes_pt_v2'), train=False, devices=f'D{dev_id}') for dev_id in [1, 2, 3, 4]]
        combined_profiling_datasets = [
            ConcatDataset([profiling_datasets[dev_id-1] for dev_id in [1, 2, 3, 4] if not dev_id == profile_id])
            for profile_id in [1, 2, 3, 4]
        ]
        combined_attack_datasets = [
            ConcatDataset([attack_datasets[dev_id-1] for dev_id in [1, 2, 3, 4] if not dev_id == profile_id])
            for profile_id in [1, 2, 3, 4]
        ]
        data_modules = [
            AES_PTv2_DataModule(combined_profiling_dataset, combined_attack_dataset)
            for combined_profiling_dataset, combined_attack_dataset in zip(combined_profiling_datasets, combined_attack_datasets)]
        for data_module in data_modules:
            data_module.setup('')
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_datasets[0].timesteps_per_trace)
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        epoch_count = 10
        obf_epoch_count = 10
        seed_count = 1
        poi_count = 5
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
        lambda_vals = np.log(16)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 10
        obf_epoch_count = 500
        poi_count = 5
    elif dataset == 'OTP':
        from datasets.one_truth_prevails import OneTruthPrevails_DataModule
        data_module = OneTruthPrevails_DataModule(root=os.path.join(DATA_DIR, 'one_truth_prevails'))
        data_module.setup('')
        profiling_dataset = data_module.profiling_dataset
        attack_dataset = data_module.attack_dataset
        supervised_classifier_kwargs['model_kwargs'] = all_style_classifier_kwargs['classifier_kwargs'] = {
            'input_shape': (1, profiling_dataset.timesteps_per_trace),
            'output_classes': 2
        }
        classifier_learning_rates = np.logspace(-6, -2, 25)
        lambda_vals = np.log(2)*np.logspace(-6, 0, LAMBDA_SWEEP_COUNT)
        epoch_count = 10
        obf_epoch_count = 10
        poi_count = 10
    else:
        assert False
    
    if not('AES_PTv2' in dataset):
        trial = Trial(
            base_dir=os.path.join(OUTPUT_DIR, f'{dataset}_results'),
            profiling_dataset=profiling_dataset,
            attack_dataset=attack_dataset,
            data_module=data_module,
            epoch_count=epoch_count,
            obf_epoch_count=obf_epoch_count,
            seed_count=seed_count,
            default_supervised_classifier_kwargs=supervised_classifier_kwargs,
            default_all_style_classifier_kwargs=all_style_classifier_kwargs,
            template_attack_poi_count=poi_count
        )
        if not clargs.plot_only:
            trial.compute_random_baseline()
            trial.compute_first_order_baselines()
            trial.supervised_lr_sweep(classifier_learning_rates)
            trial.train_optimal_supervised_classifier()
            trial.train_optimal_all_classifier()
            trial.lambda_sweep(lambda_vals)
            if clargs.long_lambda_sweep:
                _lambda_vals = [lambda_vals[clargs.lambda_idx]] if clargs.lambda_idx is not None else lambda_vals
                trial.long_lambda_sweep(_lambda_vals)
            trial.run_optimal_all()
            trial.compute_neural_net_explainability_baselines()
            trial.eval_leakage_assessments(template_attack=dataset in ['DPAv4', 'AES_HD'])
        trial.plot_everything()
    else:
        trial = PortabilityTrial(
            base_dir=os.path.join(OUTPUT_DIR, f'{dataset}_results'),
            profiling_datasets=profiling_datasets,
            attack_datasets=attack_datasets,
            data_modules=data_modules,
            epoch_count=epoch_count,
            obf_epoch_count=obf_epoch_count,
            seed_count=seed_count,
            template_attack_poi_count=poi_count,
            default_supervised_classifier_kwargs=supervised_classifier_kwargs,
            default_all_style_classifier_kwargs=all_style_classifier_kwargs
        )
        if not clargs.plot_only:
            trial.compute_random_baseline()
            #trial.compute_first_order_baselines()
            trial.supervised_lr_sweep(classifier_learning_rates)
            trial.train_optimal_supervised_classifier()
            trial.train_optimal_all_classifier()
            trial.lambda_sweep(lambda_vals)
            trial.run_optimal_all()
            trial.eval_leakage_assessments()
        trial.plot_everything()

if __name__ == '__main__':
    main()