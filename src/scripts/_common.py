import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(sys.path[0], '..')))
import argparse
import torch

from common import *

available_devices = ['cpu']
if torch.cuda.is_available():
    available_devices.extend(['cuda', *[f'cuda:{idx}' for idx in range(torch.cuda.device_count())]])

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, action='store', help='Random seed to use for this trial.')
parser.add_argument('--device', type=str, default='cpu', choices=available_devices, action='store', help='Device to use for this trial.')
parser.add_argument('--name', type=str, default=None, action='store', help='Specify the trial name.')
parser.add_argument('--overwrite', default=False, action='store_true', help='If there is already an output directory with the specified name, delete it before proceeding.')
parser.add_argument('--quiet', default=False, action='store_true', help='Disable printing to the terminal for this trial.')
clargs = parser.parse_args()

DEVICE = clargs.device
SEED = set_seed(clargs.seed)
if clargs.name is not None:
    if os.path.exists(os.path.join(OUTPUT_DIR, clargs.name)):
        if clargs.overwrite:
            shutil.rmtree(os.path.join(OUTPUT_DIR, clargs.name))
        else:
            assert False, f'Directory already exists: `{os.path.join(OUTPUT_DIR, clargs.name)}`'
    rename_trial(clargs.name)
set_verbosity(not clargs.quiet)

print(f'Starting trial with name `{get_trial_name()}`.')
print(f'\tProject directory: `{PROJ_DIR}`.')
print(f'\tSource directory: `{SRC_DIR}`.')
print(f'\tConfig directory: `{CONFIG_DIR}`.')
print(f'\tResource directory: `{RESOURCE_DIR}`.')
print(f'\tOutput directory: `{OUTPUT_DIR}`.')
print(f'\tTrial directory: `{get_trial_dir()}`.')
print(f'\tDevice: `{DEVICE}`.')
print(f'\tRandom seed: `{SEED}`.')