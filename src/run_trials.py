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
        default_kwargs = trial_config['default_kwargs']
        if trial_config['compute_first_order_stats']:
            stats_dir = os.path.join(trial_dir, 'first_order_stats')
            os.makedirs(stats_dir, exist_ok=True)
            if not os.path.exists(os.path.join(stats_dir, 'stats.npy')):
                print('Computing first-order statistical assessments...')
                first_order_statistics = FirstOrderStatistics(profiling_dataset)
                snr = first_order_statistics.snr_vals['label'].reshape(-1)
                print('\tDone computing SNR.')
                sosd = first_order_statistics.sosd_vals['label'].reshape(-1)
                print('\tDone computing SOSD.')
                cpa = first_order_statistics.cpa_vals['label'].reshape(-1)
                print('\tDone computing CPA.')
                np.save(os.path.join(stats_dir, 'stats.npy'), np.stack([snr, sosd, cpa]))
            else:
                rv = np.load(os.path.join(stats_dir, 'stats.npy'))
                snr = rv[0, :]
                sosd = rv[1, :]
                cpa = rv[2, :]
            plot_leakage_assessment(snr, os.path.join(stats_dir, 'snr.png'))
            plot_leakage_assessment(sosd, os.path.join(stats_dir, 'sosd.png'))
            plot_leakage_assessment(cpa, os.path.join(stats_dir, 'cpa.png'))
        supervised_model_dir = os.path.join(trial_dir, 'supervised_model')
        os.makedirs(supervised_model_dir, exist_ok=True)
        if trial_config['train_supervised_model']:
            print('Training supervised model...')
            supervised_trainer = SupervisedTrainer(profiling_dataset, attack_dataset, default_training_module_kwargs=trial_config['supervised_training_kwargs'])
            supervised_trainer.run(logging_dir=supervised_model_dir, max_steps=trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        if trial_config['compute_nn_attributions']:
            print('Computing neural net attribution assessments...')
            nn_attr_dir = os.path.join(trial_dir, 'nn_attr_assessments')
            os.makedirs(nn_attr_dir, exist_ok=True)
            assert os.path.exists(os.path.join(supervised_model_dir, 'final_checkpoint.ckpt'))
            profiling_dataloader = DataLoader(profiling_dataset, batch_size=1024, shuffle=False)
            nn_attributor = NeuralNetAttribution(profiling_dataloader, supervised_model_dir)
            if not os.path.exists(os.path.join(nn_attr_dir, 'gradvis.npy')):
                gradvis = nn_attributor.compute_gradvis().reshape(-1)
                print('\tDone computing GradVis.')
                np.save(os.path.join(nn_attr_dir, 'gradvis.npy'), gradvis)
            else:
                gradvis = np.load(os.path.join(nn_attr_dir, 'gradvis.npy'))
            if not os.path.exists(os.path.join(nn_attr_dir, 'saliency.npy')):
                saliency = nn_attributor.compute_saliency().reshape(-1)
                print('\tDone computing saliency.')
                np.save(os.path.join(nn_attr_dir, 'saliency.npy'), saliency)
            else:
                saliency = np.load(os.path.join(nn_attr_dir, 'saliency.npy'))
            if not os.path.exists(os.path.join(nn_attr_dir, 'occlusion.npy')):
                occlusion = nn_attributor.compute_occlusion().reshape(-1)
                print('\tDone computing occlusion.')
                np.save(os.path.join(nn_attr_dir, 'occlusion.npy'), occlusion)
            else:
                occlusion = np.load(os.path.join(nn_attr_dir, 'occlusion.npy'))
            if not os.path.exists(os.path.join(nn_attr_dir, 'inputxgrad.npy')):
                inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
                print('\tDone computing inputxgrad.')
                np.save(os.path.join(nn_attr_dir, 'inputxgrad.npy'), inputxgrad)
            else:
                inputxgrad = np.load(os.path.join(nn_attr_dir, 'inputxgrad.npy'))
            plot_leakage_assessment(gradvis, os.path.join(nn_attr_dir, 'gradvis.png'))
            plot_leakage_assessment(saliency, os.path.join(nn_attr_dir, 'saliency.png'))
            plot_leakage_assessment(occlusion, os.path.join(nn_attr_dir, 'occlusion.png'))
            plot_leakage_assessment(inputxgrad, os.path.join(nn_attr_dir, 'inputxgrad.png'))
        baselines = {
            'snr': np.abs(snr), 'sosd': np.abs(sosd), 'cpa': np.abs(cpa),
            'gradvis': np.abs(gradvis), 'saliency': np.abs(saliency), 'occlusion': np.abs(occlusion), 'inputxgrad': np.abs(inputxgrad)
        }
        for baseline_name, baseline_val in baselines.items():
            metric = GMMPerformanceCorrelation(baseline_val.argsort(), device='cuda')
            metric.profile(profiling_dataset)
            print(f'{baseline_name} GMM performance correlation: {metric(attack_dataset)}')
        leakage_localization_kwargs = copy(default_kwargs)
        leakage_localization_kwargs.update(trial_config['leakage_localization_kwargs'])
        trainer = LeakageLocalizationTrainer(
            profiling_dataset,
            attack_dataset,
            default_training_module_kwargs=default_kwargs,
            reference_leakage_assessment=baselines
        )
        if trial_config['pretrain_classifiers']:
            classifiers_pretrain_kwargs = copy(default_kwargs)
            classifiers_pretrain_kwargs.update(trial_config['classifiers_pretrain_kwargs'])
            classifiers_pretrain_dir = os.path.join(trial_dir, 'classifiers_pretrain')
            os.makedirs(classifiers_pretrain_dir, exist_ok=True)
            trainer.pretrain_classifiers(
                logging_dir=classifiers_pretrain_dir,
                max_steps=trial_config['max_classifiers_pretrain_steps'],
                override_kwargs=classifiers_pretrain_kwargs
            )
        if trial_config['hparam_tune']:
            hparam_tuning_dir = os.path.join(trial_dir, 'hparam_tune')
            os.makedirs(hparam_tuning_dir, exist_ok=True)
            trainer.hparam_tune(
                logging_dir=hparam_tuning_dir,
                pretrained_classifiers_logging_dir=classifiers_pretrain_dir if trial_config['pretrain_classifiers'] else None,
                max_steps=trial_config['max_leakage_localization_steps'],
                override_kwargs=leakage_localization_kwargs
            )
        leakage_localization_dir = os.path.join(trial_dir, 'leakage_localization')
        os.makedirs(leakage_localization_dir, exist_ok=True)
        trainer.run(
            logging_dir=leakage_localization_dir,
            pretrained_classifiers_logging_dir=classifiers_pretrain_dir if trial_config['pretrain_classifiers'] else None,
            max_steps=trial_config['max_leakage_localization_steps'],
            override_kwargs=leakage_localization_kwargs
        )

if __name__ == '__main__':
    main()