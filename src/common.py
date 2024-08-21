import builtins
import os
import shutil
import time
import datetime
from typing import *
import random
import numpy as np
import torch

from utils.constant import Constant

SRC_DIR = Constant(os.path.dirname(os.path.realpath(__file__)))
PROJ_DIR = Constant(os.path.abspath(os.path.join(SRC_DIR, '..')))
CONFIG_DIRNAME = Constant('config')
CONFIG_DIR = Constant(os.path.join(PROJ_DIR, CONFIG_DIRNAME))
RESOURCE_DIRNAME = Constant('resources')
RESOURCE_DIR = Constant(os.path.join(PROJ_DIR, RESOURCE_DIRNAME))
OUTPUT_DIRNAME = Constant('outputs')
OUTPUT_DIR = Constant(os.path.join(PROJ_DIR, OUTPUT_DIRNAME))
numpy_rng = np.random.default_rng()
_trial_name = None
_verbose = None
_seed = None

def get_trial_dir():
    assert _trial_name is not None
    trial_dir = os.path.join(OUTPUT_DIR, _trial_name)
    os.makedirs(trial_dir, exist_ok=True)
    return trial_dir

def get_log_path():
    log_path = os.path.join(get_trial_dir(), 'log.out')
    return log_path

def set_trial_name(name: str):
    global _trial_name
    _trial_name = name

def rename_trial(name: str):
    global _trial_name
    new_trial_dir = os.path.join(OUTPUT_DIR, name)
    shutil.copytree(get_trial_dir(), new_trial_dir)
    shutil.rmtree(get_trial_dir())
    _trial_name = name

def set_seed(
    seed: Optional[int] = None
) -> int:
    if seed is None:
        seed = time.time_ns() & 0xFFFFFFFF
    global NUMPY_RNG
    NUMPY_RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return seed

def set_verbosity(
    verbose: bool = True
):
    global _verbose
    _verbose = verbose

def print(*args, **kwargs):
    with open(get_log_path(), 'a+') as f:
        builtins.print(*args, file=f, **kwargs)
    if _verbose:
        builtins.print(*args, **kwargs)

_seed = set_seed()
_verbose = set_verbosity()
set_trial_name('trial__{date:%Y_%m_%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)