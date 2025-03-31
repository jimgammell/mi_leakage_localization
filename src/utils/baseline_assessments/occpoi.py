# Implementation of the OccPOI algorithm (Yap 2025) from https://eprint.iacr.org/2023/1055.pdf

from typing import Union, Optional, Sequence, Literal
import os
from tqdm import tqdm
from copy import copy
from random import shuffle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from training_modules.supervised_deep_sca import SupervisedModule
from models.zaid_wouters_nets import pretrained_models
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator

class OccludedModel(nn.Module):
    def __init__(self, model: nn.Module, points_to_occlude: Sequence[int]):
        super().__init__()
        self.model = model
        self.points_to_occlude = points_to_occlude
        self.input_shape = self.model.input_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        occluded_x = x.clone()
        occluded_x[..., self.points_to_occlude] = 0
        logits = self.model(occluded_x)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

class OccPOI:
    def __init__(self,
        attack_dataloader, model: Union[nn.Module, str], seed: Optional[int] = None, device: Optional[str] = None,
        dataset_name: Literal['dpav4', 'aes_hd', 'ascadv1_fixed', 'ascadv1_variable'] = 'dpav4'
    ):
        if dataset_name == 'dpav4':
            attack_traces = 10
        elif dataset_name == 'ascadv1_fixed':
            attack_traces = 1000
        elif dataset_name == 'ascadv1_variable':
            attack_traces = 10000
        elif dataset_name == 'aes_hd':
            attack_traces = 10000
        else:
            assert False
        attack_dataset = Subset(attack_dataloader.dataset, np.arange(attack_traces))
        #attack_dataset.dataset.traces = torch.from_numpy(attack_dataset.dataset.traces).to(device)
        self.attack_dataloader = DataLoader(attack_dataset, batch_size=attack_traces)
        if isinstance(model, str):
            if 'ZaidNet' in model or 'Wouters' in model:
                model_class = getattr(pretrained_models, model)
                assert seed is not None
                model = model_class(pretrained_seed=seed)
            else:
                logging_dir = model
                assert os.path.exists(os.path.join(logging_dir, 'best_checkpoint.ckpt'))
                training_module = SupervisedModule.load_from_checkpoint(os.path.join(logging_dir, 'best_checkpoint.ckpt'))
                model = training_module.classifier
        self.base_model = model
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.seed = seed
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset_name
        self.trace_shape = self.base_model.input_shape
        self.model = OccludedModel(self.base_model, [])
        base_guessing_entropy = self.compute_guessing_entropy([])
        self.lbda = base_guessing_entropy + 1 # generalizes lambda in paper to settings where we don't get down to zero guessing entropy
    
    def compute_guessing_entropy(self, points_to_occlude: Sequence[int]):
        self.model.points_to_occlude = points_to_occlude
        multi_trace_evaluator = AESMultiTraceEvaluator(
            dataloader=self.attack_dataloader, model=self.model, seed=self.seed, device=self.device, dataset_name=self.dataset_name 
        )
        rank_over_time = multi_trace_evaluator()
        guessing_entropy = rank_over_time[-1]
        return guessing_entropy
    
    def run_kgo_procedure(self, starting_queue: Optional[Sequence[int]] = None):
        queue = copy(list(starting_queue)) if starting_queue is not None else list(range(self.trace_shape[-1]))
        has_converged = False
        points_to_occlude = list(set(range(self.trace_shape[-1])) - set(queue))
        iteration = 0
        while not has_converged:
            has_converged = True
            shuffle(queue)
            important_index = []
            print(f'OccPOI iteration {iteration}...')
            for spt in (progress_bar := tqdm(queue)):
                points_to_occlude.append(spt)
                guessing_entropy = self.compute_guessing_entropy(points_to_occlude)
                progress_bar.set_description(f'Guessing entropy: {guessing_entropy}')
                if guessing_entropy >= self.lbda:
                    important_index.append(spt)
                    points_to_occlude.pop()
                else:
                    has_converged = False
            print(f'Iteration complete. Current \'important point\' list:')
            print(set(important_index))
            queue = important_index
            points_to_occlude = list(set(range(self.trace_shape[-1])) - set(queue))
            iteration += 1
        return queue
    
    def run_extended_kgo_procedure(self):
        print(f'Running OccPOI. Lambda value: {self.lbda}.')
        all_timesteps = set(range(self.trace_shape[-1]))
        occpois = set(self.run_kgo_procedure())
        nonextended_pois = copy(occpois)
        print(f'Pre-start: occpois={occpois}')
        while (len(all_timesteps - occpois) > 0) and (self.compute_guessing_entropy(list(occpois)) < self.lbda):
            queue = list(all_timesteps - occpois)
            new_occpois = set(self.run_kgo_procedure(queue))
            if len(new_occpois) == 0:
                break
            occpois = occpois.union(new_occpois)
            print(f'New sub-trial finished. Current occpois: {occpois}')
        return list(occpois), list(nonextended_pois)
    
    def __call__(self):
        occpois, nonextended_pois = self.run_extended_kgo_procedure()
        ranked_occpois = []
        base_ge = self.compute_guessing_entropy([])
        for x in occpois:
            occluded_ge = self.compute_guessing_entropy(list((set(range(self.trace_shape[-1])) - set(occpois)) + set([x])))
            ranked_occpois.append(occluded_ge - base_ge)
        ranked_nonextended_pois = []
        for x in nonextended_pois:
            occluded_ge = self.compute_guessing_entropy(list((set(range(self.trace_shape[-1])) - set(nonextended_pois)) + set([x])))
            ranked_nonextended_pois.append(occluded_ge - base_ge)
        leakage_assessment = np.zeros(self.trace_shape, dtype=np.float32).squeeze()
        leakage_assessment[..., occpois] = ranked_occpois
        nonextended_leakage_assessment = np.zeros(self.trace_shape, dtype=np.float32).squeeze()
        nonextended_leakage_assessment[..., nonextended_pois] = ranked_nonextended_pois
        return leakage_assessment, nonextended_leakage_assessment