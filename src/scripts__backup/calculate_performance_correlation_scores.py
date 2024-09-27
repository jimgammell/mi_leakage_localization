import os
import numpy as np
import pickle

from _common import *
from utils.performance_correlation import MeasurePerformanceCorrelation
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd
from utils.calculate_cpa import calculate_cpa
from datasets.aes_hd import AES_HD, to_key_preds as aes_hd_to_key_preds
from datasets.aes_rd import AES_RD, to_key_preds as aes_rd_to_key_preds
from datasets.dpav4 import DPAv4, to_key_preds as dpav4_to_key_preds
from datasets.ascadv1 import ASCADv1, to_key_preds as ascadv1_to_key_preds

RUN_RANDOM_BASELINE = True

SEED_COUNT = 5

SUPERVISED_AES_HD_DIR = os.path.join(OUTPUT_DIR, 'supervised_aes_hd__noise')
SUPERVISED_AES_RD_DIR = os.path.join(OUTPUT_DIR, 'supervised_aes_rd')
SUPERVISED_DPAV4_DIR = os.path.join(OUTPUT_DIR, 'supervised_dpav4')
SUPERVISED_ASCAD_DIR = os.path.join(OUTPUT_DIR, 'supervised_ascadv1f')

aes_hd_profiling_dataset = AES_HD(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_hd'), train=True)
aes_rd_profiling_dataset = AES_RD(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_rd'), train=True)
dpav4_profiling_dataset = DPAv4(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4'), train=True)
ascadv1_profiling_dataset = ASCADv1(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'), train=True)
aes_hd_attack_dataset = AES_HD(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_hd'), train=False)
aes_rd_attack_dataset = AES_RD(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'aes_rd'), train=False)
dpav4_attack_dataset = DPAv4(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'dpav4'), train=False)
ascadv1_attack_dataset = ASCADv1(root=os.path.join('/mnt', 'hdd', 'jgammell', 'leakage_localization', 'downloads', 'ascadv1'), train=False)

if RUN_RANDOM_BASELINE:
    random_aes_hd = np.full((SEED_COUNT,), np.nan, dtype=np.float32)
    random_aes_rd = np.full((SEED_COUNT,), np.nan, dtype=np.float32)
    random_dpav4 = np.full((SEED_COUNT,), np.nan, dtype=np.float32)
    random_ascadv1 = np.full((SEED_COUNT,), np.nan, dtype=np.float32)
    for idx in range(SEED_COUNT):
        aes_hd_metric = MeasurePerformanceCorrelation(
            np.random.randn(aes_hd_profiling_dataset.timesteps_per_trace), aes_hd_profiling_dataset, aes_hd_attack_dataset,
            target_keys='last_state', int_var_to_key_fn=
        )
        random_aes_hd[idx], _ = aes_hd_metric.measure_performance()
        aes_rd_metric = MeasurePerformanceCorrelation(
            np.random.randn(aes_rd_profiling_dataset.timesteps_per_trace), aes_rd_profiling_dataset, aes_rd_attack_dataset
        )
        random_aes_rd[idx], _ = aes_rd_metric.measure_performance()
        dpav4_metric = MeasurePerformanceCorrelation(
            np.random.randn(dpav4_profiling_dataset.timesteps_per_trace), dpav4_profiling_dataset, dpav4_attack_dataset
        )
        random_dpav4[idx], _ = dpav4_metric.measure_performance()
        ascadv1_metric = MeasurePerformanceCorrelation(
            np.random.randn(ascadv1_profiling_dataset.timesteps_per_trace), ascadv1_profiling_dataset, ascadv1_attack_dataset
        )
        random_ascadv1[idx], _ = ascadv1_metric.measure_performance()
    with open(os.path.join(get_trial_dir(), 'random_baseline.pickle'), 'wb') as f:
        pickle.dump({
            'aes-hd': random_aes_hd, 'aes-rd': random_aes_rd, 'dpav4': random_dpav4, 'ascadv1': random_ascadv1
        }, f)
    print('Random baselines:')
    print(f'\tAES-HD: {random_aes_hd.mean()} +/- {random_aes_hd.std()}')
    print(f'\tAES-RD: {random_aes_rd.mean()} +/- {random_aes_rd.std()}')
    print(f'\tDPAv4: {random_dpav4.mean()} +/- {random_dpav4.std()}')
    print(f'\tASCADv1: {random_ascadv1.mean()} +/- {random_ascadv1.std()}')