import os
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from lightning import Trainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from .utils import *
from training_modules.supervised_classification import SupervisedClassificationModule
from training_modules.discrete_adversarial_localization import DiscreteAdversarialLocalizationTrainer as ALLTrainer
from utils.localization_via_interpretability import compute_gradvis, compute_input_x_gradient, compute_feature_ablation_map
from utils.calculate_cpa import calculate_cpa
from utils.calculate_snr import calculate_snr
from utils.calculate_sosd import calculate_sosd



class Trial:
    def __init__(self,
    ):
        pass
    