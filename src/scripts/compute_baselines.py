import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
from torchvision import transforms

from _common import *
from datasets.aes_hd import AES_HD
from datasets.aes_rd import AES_RD
from datasets.ascadv1 import ASCADv1
from datasets.dpav4 import DPAv4
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map
from utils.performance_correlation import MeasurePerformanceCorrelation
from training_modules import SupervisedClassificationModule

COMPUTE_LEAKAGE_ASSESSMENTS = False
EVALUATE_LEAKAGE_ASSESSMENTS = True

experiments = {
#    'aes-hd': {
#        'constructor': AES_HD,
#        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_hd'),
#        'weights_path': r'/home/jgammell/Desktop/mi_leakage_localization/outputs/aes_hd_supervised_htune/learning_rate=2e-05__weight_decay=0.0__additive_noise=0.25/final_checkpoint.ckpt',
#        'target': 'last_state'
#    },
#    'aes-rd': {
#        'constructor': AES_RD,
#        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_rd'),
#        'weights_path': r'/home/jgammell/Desktop/mi_leakage_localization/outputs/aes_rd_supervised_htune/learning_rate=4e-05__weight_decay=0.0__additive_noise=0.0/final_checkpoint.ckpt',
#        'target': 'subbytes'
#    },
    'ascadv1f': {
        'constructor': ASCADv1,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
        'weights_path': [
            f'/home/jgammell/Desktop/mi_leakage_localization/outputs/ascadv1f_supervised_error_bars/learning_rate=6e-05__weight_decay=0.0__additive_noise=0.25__seed={seed}/lightning_output/version_0/checkpoints/best_model.ckpt'
            for seed in [0, 1, 2, 3, 4]
        ],
        'target': ['subbytes', 'r', 'subbytes__r']
    },
    'dpav4': {
        'constructor': DPAv4,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4'),
        'weights_path': [
            f'/home/jgammell/Desktop/mi_leakage_localization/outputs/dpav4_supervised_error_bars/learning_rate=6e-05__weight_decay=0.0__additive_noise=0.25__seed={seed}/lightning_output/version_0/checkpoints/best_model.ckpt'
            for seed in [0, 1, 2, 3, 4]
        ],
        'target': 'subbytes'
    }
}

def plot_leakage_assessment(leakage_assessments, ylabel=None, savepath=None):
    for target, leakage_assessment in leakage_assessments.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(leakage_assessment.squeeze(), marker='.', linestyle='none', color='blue', markersize=3, rasterized=True)
        ax.set_xlabel('Timestep $t$')
        ax.set_ylabel(ylabel)
        int_var = target[0]
        ax.set_title(r'Intermediate variable: \texttt{' + int_var.replace('_', r'\_') + '}')
        fig.tight_layout()
        target_savepath = savepath.split('.')[0] + f'__{int_var}.' + savepath.split('.')[-1]
        fig.savefig(target_savepath)
        plt.close('all')

for experiment_name, experiment in experiments.items():
    save_dir = os.path.join(get_trial_dir(), experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    profiling_dataset = experiment['constructor'](root=experiment['root'], train=True)
    attack_dataset = experiment['constructor'](root=experiment['root'], train=False)
    profiling_dataset.return_metadata = attack_dataset.return_metadata = True
    if COMPUTE_LEAKAGE_ASSESSMENTS:
        cpa_leakage_assessment = calculate_cpa(profiling_dataset, profiling_dataset, experiment['target'])
        snr_leakage_assessment = calculate_snr(profiling_dataset, profiling_dataset, experiment['target'])
        sosd_leakage_assessment = calculate_sosd(profiling_dataset, profiling_dataset, experiment['target'])
        with open(os.path.join(save_dir, 'cpa.pickle'), 'wb') as f:
            pickle.dump(cpa_leakage_assessment, f)
        with open(os.path.join(save_dir, 'snr.pickle'), 'wb') as f:
            pickle.dump(snr_leakage_assessment, f)
        with open(os.path.join(save_dir, 'sosd.pickle'), 'wb') as f:
            pickle.dump(sosd_leakage_assessment, f)
        plot_leakage_assessment(cpa_leakage_assessment, 'Correlation power analysis', os.path.join(save_dir, 'cpa.pdf'))
        plot_leakage_assessment(snr_leakage_assessment, 'Signal-noise ratio', os.path.join(save_dir, 'snr.pdf'))
        plot_leakage_assessment(sosd_leakage_assessment, 'Sum of squared differences', os.path.join(save_dir, 'sosd.pdf'))
        profiling_dataset.return_metadata = attack_dataset.return_metadata = False
        profiling_dataset.transform = attack_dataset.transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float))
        profiling_dataset.target_transform = attack_dataset.target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        for seed, weights_path in enumerate(experiment['weights_path']):
            training_module = SupervisedClassificationModule.load_from_checkpoint(
                experiment['weights_path'][seed],
                model_name='sca-cnn',
                model_kwargs={'input_shape': (1, profiling_dataset.timesteps_per_trace)},
                optimizer_name='AdamW',
                optimizer_kwargs={'lr': 1.0} # unused but we have to specify it
            )
            ablation_assessment = compute_feature_ablation_map(training_module, profiling_dataset)
            with open(os.path.join(save_dir, f'ablation__seed={seed}.pickle'), 'wb') as f:
                pickle.dump(ablation_assessment, f)
            plot_leakage_assessment({('subbytes', None): ablation_assessment}, 'Input ablation', os.path.join(save_dir, 'ablation.pdf'))
            gradvis_assessment = compute_gradvis(training_module, profiling_dataset)
            with open(os.path.join(save_dir, f'gradvis__seed={seed}.pickle'), 'wb') as f:
                pickle.dump(gradvis_assessment, f)
            plot_leakage_assessment({('subbytes', None): gradvis_assessment}, 'Gradient Visualization', os.path.join(save_dir, 'gradvis.pdf'))
            input_x_gradient_assessment = compute_input_x_gradient(training_module, profiling_dataset)
            with open(os.path.join(save_dir, f'inputxgrad__seed={seed}.pickle'), 'wb') as f:
                pickle.dump(input_x_gradient_assessment, f)
            plot_leakage_assessment({('subbytes', None): input_x_gradient_assessment}, 'Input $*$ gradient', os.path.join(save_dir, 'inputxgrad.pdf'))
    
    if EVALUATE_LEAKAGE_ASSESSMENTS:
        with open(os.path.join(save_dir, 'cpa.pickle'), 'rb') as f:
            cpa_leakage_assessment = pickle.load(f)
        with open(os.path.join(save_dir, 'snr.pickle'), 'rb') as f:
            snr_leakage_assessment = pickle.load(f)
        with open(os.path.join(save_dir, 'sosd.pickle'), 'rb') as f:
            sosd_leakage_assessment = pickle.load(f)
        gradvis_leakage_assessment = []
        inputxgrad_leakage_assessment = []
        ablation_leakage_assessment = []
        all_leakage_assessment = []
        for seed in range(5):
            with open(os.path.join(save_dir, f'gradvis__seed={seed}.pickle'), 'rb') as f:
                _gradvis_leakage_assessment = pickle.load(f)
                gradvis_leakage_assessment.append(_gradvis_leakage_assessment)
            with open(os.path.join(save_dir, f'inputxgrad__seed={seed}.pickle'), 'rb') as f:
                _inputxgrad_leakage_assessment = pickle.load(f)
                inputxgrad_leakage_assessment.append(_inputxgrad_leakage_assessment)
            with open(os.path.join(save_dir, f'ablation__seed={seed}.pickle'), 'rb') as f:
                _ablation_leakage_assessment = pickle.load(f)
                ablation_leakage_assessment.append(_ablation_leakage_assessment)
            try:
                with open(os.path.join(save_dir, f'all_results__seed={seed}.pickle'), 'rb') as f:
                    all_results = pickle.load(f)
                    _all_leakage_assessment = all_results['erasure_probs']
                    all_leakage_assessment.append(_all_leakage_assessment)
            except:
                pass
        random_leakage_assessment = [np.random.randn(*_gradvis_leakage_assessment.shape) for _ in range(5)]
        if experiment_name == 'ascadv1f':
            approaches = [
                ('all', all_leakage_assessment),
                ('random', random_leakage_assessment),
                ('cpa', np.abs(cpa_leakage_assessment[('subbytes', None)])),
                ('snr', snr_leakage_assessment[('subbytes', None)]),
                ('sosd', sosd_leakage_assessment[('subbytes', None)]),
                ('cpa_gt', np.abs(cpa_leakage_assessment[('r', None)]) + np.abs(cpa_leakage_assessment[('subbytes__r', None)])),
                ('snr_gt', snr_leakage_assessment[('r', None)] + snr_leakage_assessment[('subbytes__r', None)]),
                ('sosd_gt', sosd_leakage_assessment[('r', None)] + sosd_leakage_assessment[('subbytes__r', None)]),
                ('gradvis', gradvis_leakage_assessment),
                ('inputxgrad', inputxgrad_leakage_assessment),
                ('ablation', ablation_leakage_assessment)
            ]
        else:
            approaches = [
                ('all', all_leakage_assessment),
                ('random', random_leakage_assessment),
                ('cpa', np.abs(cpa_leakage_assessment[(experiment['target'], None)])),
                ('snr', snr_leakage_assessment[(experiment['target'], None)]),
                ('sosd', sosd_leakage_assessment[(experiment['target'], None)]),
                ('gradvis', gradvis_leakage_assessment),
                ('inputxgrad', inputxgrad_leakage_assessment),
                ('ablation', ablation_leakage_assessment)
            ]
        print(experiment_name)
        for approach_name, leakage_assessment in approaches:
            if isinstance(leakage_assessment, list):
                for seed, _leakage_assessment in enumerate(leakage_assessment):
                    if os.path.exists(os.path.join(save_dir, f'{approach_name}__perfcorr__seed={seed}.pickle')):
                        continue
                    print(approach_name)
                    metric = MeasurePerformanceCorrelation(_leakage_assessment, profiling_dataset, attack_dataset, target_keys=experiment['target'])
                    performance_correlation, performance_means, performance_stds = metric.measure_performance(fast=False)
                    print(performance_correlation, performance_means, performance_stds)
                    print()
                    with open(os.path.join(save_dir, f'{approach_name}__perfcorr__seed={seed}.pickle'), 'wb') as f:
                        pickle.dump({'corr': performance_correlation, 'means': performance_means, 'stds': performance_stds}, f)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.fill_between(np.arange(len(performance_means)), performance_means-performance_stds, performance_means+performance_stds, color='blue', alpha=0.25)
                    ax.plot(performance_means, color='blue', linestyle='-')
                    ax.set_xlabel('Estimated leakage (relative)')
                    ax.set_ylabel('Rank of correct key')
                    fig.tight_layout()
                    fig.savefig(os.path.join(save_dir, f'{approach_name}__perfcorr__seed={seed}.pdf'))
            else:
                if os.path.exists(os.path.join(save_dir, f'{approach_name}__perfcorr.pickle')):
                    continue
                print(approach_name)
                metric = MeasurePerformanceCorrelation(leakage_assessment, profiling_dataset, attack_dataset, target_keys=experiment['target'])
                performance_correlation, performance_means, performance_stds = metric.measure_performance(fast=False)
                print(performance_correlation, performance_means, performance_stds)
                print()
                with open(os.path.join(save_dir, f'{approach_name}__perfcorr.pickle'), 'wb') as f:
                    pickle.dump({'corr': performance_correlation, 'means': performance_means, 'stds': performance_stds}, f)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.fill_between(np.arange(len(performance_means)), performance_means-performance_stds, performance_means+performance_stds, color='blue', alpha=0.25)
                ax.plot(performance_means, color='blue', linestyle='-')
                ax.set_xlabel('Estimated leakage (relative)')
                ax.set_ylabel('Rank of correct key')
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, f'{approach_name}__perfcorr.pdf'))