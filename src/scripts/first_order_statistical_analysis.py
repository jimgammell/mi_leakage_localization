import os
import pickle

from _common import *
from datasets.ascadv1 import ASCADv1
from datasets.dpav4 import DPAv4
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd

experiments = {
    'ascadv1-fixed': {
        'constructor': ASCADv1,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
        'target': ['subbytes', 'r', 'subbytes__r'],
        'kwargs': {}
    },
    'ascadv1-variable': {
        'constructor': ASCADv1,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'),
        'target': ['subbytes', 'r', 'subbytes__r'],
        'kwargs': {'variable_keys': True}
    },
    'dpav4': {
        'constructor': DPAv4,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4'),
        'kwargs': {},
        'target': ['subbytes']
    },
    'aes-ptv2': {
        'constructor': AES_PTv2,
        'root': os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_pt_v2'),
        'kwargs': {'devices': ['D1', 'D2', 'D3', 'D4'], 'countermeasure': 'Unprotected'},
        'target': ['subbytes', 'r_out', 'subbytes__r_out']
    }
}

for experiment_name, experiment in experiments.items():
    save_dir = os.path.join(get_trial_dir(), experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    profiling_dataset = experiment['constructor'](root=experiment['root'], train=True, **experiment['kwargs'])
    attack_dataset = experiment['constructor'](root=experiment['root'], train=False, **experiment['kwargs'])
    profiling_dataset.return_metadata = attack_dataset.return_metadata = True
    cpa_leakage_assessment = calculate_cpa(profiling_dataset, profiling_dataset, experiment['target'])
    snr_leakage_assessment = calculate_snr(profiling_dataset, profiling_dataset, experiment['target'])
    sosd_leakage_assessment = calculate_sosd(profiling_dataset, profiling_dataset, experiment['target'])
    with open(os.path.join(save_dir, 'cpa.pickle'), 'wb') as f:
        pickle.dump(cpa_leakage_assessment, f)
    with open(os.path.join(save_dir, 'snr.pickle'), 'wb') as f:
        pickle.dump(snr_leakage_assessment, f)
    with open(os.path.join(save_dir, 'sosd.pickle'), 'wb') as f:
        pickle.dump(sosd_leakage_assessment, f)
    
    fig, axes = plt.subplots(3, len(experiment['target']), figsize=(4*len(experiment['target']), 4), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    for axes_c, target in zip(axes.transpose((1, 0)), experiment['target']):
        axes_c[0].plot(cpa_leakage_assessment[(target, None)].squeeze(), color='blue', linestyle='none', marker='.')
        axes_c[1].plot(snr_leakage_assessment[(target, None)].squeeze(), color='blue', linestyle='none', marker='.')
        axes_c[2].plot(sosd_leakage_assessment[(target, None)].squeeze(), color='blue', linestyle='none', marker='.')
    fig.tight_layout()
    fig.savefig(os.path.join(get_trial_dir(), experiment_name))