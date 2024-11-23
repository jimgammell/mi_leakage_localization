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
    
    if dataset == 'ASCADv1-fixed':
        from datasets.ascadv1 import ASCADv1
        profiling_dataset = ASCADv1(root=os.path.join(DATA_DIR, 'ascadv1'), train=True)
        attack_dataset = ASCADv1(root=os.path.join(DATA_DIR, 'ascadv1'), train=False)
    elif dataset == 'ASCADv1-variable':
        from datasets.ascadv1 import ASCADv1
        profiling_dataset = ASCADv1(root=os.path.join(DATA_DIR, 'ascadv1'), train=True, variable_keys=True)
        attack_dataset = ASCADv1(root=os.path.join(DATA_DIR, 'ascadv1'), train=False, variable_keys=True)
    elif dataset == 'DPAv4':
        from datasets.dpav4 import DPAv4
        profiling_dataset = DPAv4(root=os.path.join(DATA_DIR, 'dpav4'), train=True)
        attack_dataset = DPAv4(root=os.path.join(DATA_DIR, 'dpav4'), train=False)
    elif dataset == 'AES_HD':
        from datasets.aes_hd import AES_HD
        profiling_dataset = AES_HD(root=os.path.join(DATA_DIR, 'aes_hd'), train=True)
        attack_dataset = AES_HD(root=os.path.join(DATA_DIR, 'aes_hd'), train=False)
    elif dataset == 'AES_PTv2':
        from datasets.aes_pt_v2 import AES_PTv2
        raise NotImplementedError
    elif dataset == 'OTiAiT':
        from datasets.ed25519_wolfssl import ED25519
        profiling_dataset = ED25519(root=os.path.join(DATA_DIR, 'one_trace_is_all_it_takes'), train=True)
        attack_dataset = ED25519(root=os.path.join(DATA_DIR, 'one_trace_is_all_it_takes'), train=False)
    elif dataset == 'OTP':
        from datasets.one_truth_prevails import OneTruthPrevails
        profiling_dataset = OneTruthPrevails(root=os.path.join(DATA_DIR, 'one_truth_prevails'), train=True, mmap_profiling_dataset=True)
        attack_dataset = OneTruthPrevails(root=os.path.join(DATA_DIR, 'one_truth_prevails'), train=False)
    else:
        assert False
    
    trial = Trial(
        base_dir=os.path.join(OUTPUT_DIR, f'{dataset}_results'),
        profiling_dataset=profiling_dataset,
        attack_dataset=attack_dataset,
        seed_count=seed_count
    )
    trial.compute_random_baseline()
    trial.compute_first_order_baselines()
    trial.eval_leakage_assessments(template_attack=not(dataset in ['OTiAiT', 'OTP']))
    trial.plot_everything()

if __name__ == '__main__':
    main()