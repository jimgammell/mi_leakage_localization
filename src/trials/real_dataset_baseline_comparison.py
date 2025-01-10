from typing import *
from copy import copy
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from common import *
from .utils import *
from datasets.dpav4 import DPAv4
from datasets.ascadv1 import ASCADv1
from datasets.ed25519_wolfssl import ED25519
from datasets.one_truth_prevails import OneTruthPrevails
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from training_modules import SupervisedTrainer, LeakageLocalizationTrainer

class Trial:
    def __init__(self,
        dataset_name: Literal['dpav4', 'ascadv1-fixed', 'ascadv1-variable', 'otiait', 'otp'],
        trial_config: dict,
        seed_count: int = 1,
        logging_dir: Optional[Union[str, os.PathLike]] = None
    ):
        self.dataset_name = dataset_name
        self.trial_config = trial_config
        self.seed_count = seed_count
        self.logging_dir = logging_dir if logging_dir is not None else dataset_name
        os.makedirs(self.logging_dir, exist_ok=True)
        self.stats_dir = os.path.join(self.logging_dir, 'first_order_stats')
        os.makedirs(self.stats_dir, exist_ok=True)
        self.supervised_model_dir = os.path.join(self.logging_dir, 'supervised_model')
        os.makedirs(self.supervised_model_dir, exist_ok=True)
        self.nn_attr_dir = os.path.join(self.logging_dir, 'nn_attr_assessments')
        os.makedirs(self.nn_attr_dir, exist_ok=True)
        self.ll_classifiers_pretrain_dir = os.path.join(self.logging_dir, 'll_classifiers_pretrain')
        os.makedirs(self.ll_classifiers_pretrain_dir, exist_ok=True)
        self.leakage_localization_dir = os.path.join(self.logging_dir, 'leakage_localization')
        os.makedirs(self.leakage_localization_dir, exist_ok=True)
        
        print('Constructing datasets...')
        if self.dataset_name == 'dpav4':
            self.profiling_dataset = DPAv4(root=trial_config['data_dir'], train=True)
            self.attack_dataset = DPAv4(root=trial_config['data_dir'], train=False)
        elif self.dataset_name == 'ascadv1-fixed':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif self.dataset_name == 'ascadv1-variable':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif dataset_name == 'otiait':
            self.profiling_dataset = ED25519(root=trial_config['data_dir'], train=True)
            self.attack_dataset = ED25519(root=trial_config['data_dir'], train=False)
        elif dataset_name == 'otp':
            self.profiling_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=True)
            self.attack_dataset = OneTruthPrevails(root=trial_config['data_dir'], train=False)
        else:
            assert False
        print('\tDone.')
        
    def compute_first_order_stats(self):
        if not os.path.exists(os.path.join(self.stats_dir, 'stats.npy')):
            print('Computing first-order statistical assessments...')
            first_order_stats = FirstOrderStatistics(self.profiling_dataset)
            snr = first_order_stats.snr_vals['label'].reshape(-1)
            sosd = first_order_stats.sosd_vals['label'].reshape(-1)
            cpa = first_order_stats.cpa_vals['label'].reshape(-1)
            np.save(os.path.join(self.stats_dir, 'stats.npy'), np.stack([snr, sosd, cpa]))
            print('\tDone.')
        else:
            rv = np.load(os.path.join(self.stats_dir, 'stats.npy'))
            snr = rv[0, :]
            sosd = rv[1, :]
            cpa = rv[2, :]
            print('Found precomputed first-order statistical assessments.')
        plot_leakage_assessment(snr, os.path.join(self.stats_dir, 'snr.png'))
        plot_leakage_assessment(sosd, os.path.join(self.stats_dir, 'sosd.png'))
        plot_leakage_assessment(cpa, os.path.join(self.stats_dir, 'cpa.png'))
        self.first_order_stats = {
            'snr': snr, 'sosd': sosd, 'cpa': cpa
        }
    
    def train_supervised_model(self):
        if not os.path.exists(os.path.join(self.supervised_model_dir, 'final_checkpoint.ckpt')):
            print('Training supervised model...')
            supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=self.trial_config['supervised_training_kwargs'])
            supervised_trainer.run(logging_dir=self.supervised_model_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found pretrained supervised model.')
    
    def compute_neural_net_attributions(self):
        profiling_dataloader = DataLoader(self.profiling_dataset, shuffle=False, batch_size=1024)
        nn_attributor = NeuralNetAttribution(profiling_dataloader, self.supervised_model_dir)
        if not os.path.exists(os.path.join(self.nn_attr_dir, 'gradvis.npy')):
            print('Computing GradVis...')
            gradvis = nn_attributor.compute_gradvis().reshape(-1)
            np.save(os.path.join(self.nn_attr_dir, 'gradvis.npy'), gradvis)
            print('\tDone.')
        else:
            gradvis = np.load(os.path.join(self.nn_attr_dir, 'gradvis.npy'))
            print('Found precomputed GradVis.')
        if not os.path.exists(os.path.join(self.nn_attr_dir, 'saliency.npy')):
            print('Computing saliency...')
            saliency = nn_attributor.compute_saliency().reshape(-1)
            np.save(os.path.join(self.nn_attr_dir, 'saliency.npy'), saliency)
            print('\tDone.')
        else:
            saliency = np.load(os.path.join(self.nn_attr_dir, 'saliency.npy'))
            print('Found precomputed saliency.')
        if not os.path.exists(os.path.join(self.nn_attr_dir, 'occlusion.npy')):
            print('Computing occlusion...')
            occlusion = nn_attributor.compute_occlusion().reshape(-1)
            np.save(os.path.join(self.nn_attr_dir, 'occlusion.npy'), occlusion)
            print('\tDone.')
        else:
            occlusion = np.load(os.path.join(self.nn_attr_dir, 'occlusion.npy'))
            print('Found precomputed occlusion.')
        if not os.path.exists(os.path.join(self.nn_attr_dir, 'inputxgrad.npy')):
            print('Computing inputxgrad...')
            inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
            np.save(os.path.join(self.nn_attr_dir, 'inputxgrad.npy'), inputxgrad)
            print('\tDone.')
        else:
            inputxgrad = np.load(os.path.join(self.nn_attr_dir, 'inputxgrad.npy'))
            print('Found precomputed inputxgrad.')
        plot_leakage_assessment(gradvis, os.path.join(self.nn_attr_dir, 'gradvis.png'))
        plot_leakage_assessment(saliency, os.path.join(self.nn_attr_dir, 'saliency.png'))
        plot_leakage_assessment(occlusion, os.path.join(self.nn_attr_dir, 'occlusion.png'))
        plot_leakage_assessment(inputxgrad, os.path.join(self.nn_attr_dir, 'inputxgrad.png'))
        self.nn_attr_assessments = {
            'gradvis': gradvis, 'saliency': saliency, 'occlusion': occlusion, 'inputxgrad': inputxgrad
        }
    
    def pretrain_leakage_localization_classifiers(self):
        if not os.path.exists(os.path.join(self.pretrain_leakage_localization_classifiers, 'final_checkpoint.ckpt')):
            print('Pretraining leakage localization classifiers...')
            trainer = LeakageLocalizationTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=self.trial_config['default_kwargs']
            )
            classifiers_pretrain_kwargs = copy(self.trial_config['default_kwargs'])
            classifiers_pretrain_kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            trainer.pretrain_classifiers(
                logging_dir=self.ll_classifiers_pretrain_dir,
                max_steps=self.trial_config['max_classifiers_pretrain_steps'],
                override_kwargs=classifiers_pretrain_kwargs
            )
            print('\tDone.')
        else:
            print('Found pretrained leakage localization classifiers.')
    
    def run_leakage_localization(self):
        if not os.path.exists(os.path.join(self.leakage_localization_dir, 'final_checkpoint.ckpt')):
            print('Running leakage localization...')
            trainer = LeakageLocalizationTrainer(
                self.profiling_dataset, self.attack_dataset,
                default_training_module_kwargs=self.trial_config['default_kwargs'],
                reference_leakage_assessment=self.get_leakage_assessments()
            )
            leakage_localization_kwargs = copy(self.trial_config['default_kwargs'])
            leakage_localization_kwargs.update(self.trial_config['leakage_localization_kwargs'])
            leakage_assessment = trainer.run(
                logging_dir=self.leakage_localization_dir,
                pretrained_classifiers_logging_dir=self.ll_classifiers_pretrain_dir if hasattr(self.trial_config, 'pretrain_classifiers') and self.trial_config['pretrain_classifiers'] else None,
                max_steps=self.trial_config['max_leakage_localization_steps'],
                override_kwargs=leakage_localization_kwargs
            )
            self.leakage_localization_assessments = {
                'leakage_localization': leakage_assessment
            }
            print('\tDone.')
        else:
            print('Found preexisting leakage localization output.')
    
    def get_leakage_assessments(self):
        leakage_assessments = {}
        if hasattr(self, 'first_order_stats'):
            leakage_assessments.update(self.first_order_stats)
        if hasattr(self, 'nn_attr_assessments'):
            leakage_assessments.update(self.nn_attr_assessments)
        if hasattr(self, 'leakage_localization_assessments'):
            leakage_assessments.update(self.leakage_localization_assessments)
        return leakage_assessments
    
    def plot_leakage_assessments(self):
        leakage_assessments = self.get_leakage_assessments()
        row_count = 2
        col_count = int(np.ceil(len(leakage_assessments)/row_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(col_count*PLOT_WIDTH, row_count*PLOT_WIDTH))
        for ax, (la_name, la) in zip(axes.flatten(), leakage_assessments.items()):
            ax.plot(la, color='blue', marker='.', linestyle='-', markersize=1, linewidth=0.1, **PLOT_KWARGS)
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakage of $X_t$')
            ax.set_title(f'Technique: {la_name}')
        for ax in axes.flatten()[len(leakage_assessments):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'leakage_assessments.pdf'), **SAVEFIG_KWARGS)
    
    def __call__(self):
        if ('compute_first_order_stats' in self.trial_config) and self.trial_config['compute_first_order_stats']:
            self.compute_first_order_stats()
        if ('train_supervised_model' in self.trial_config) and self.trial_config['train_supervised_model']:
            self.train_supervised_model()
        if ('compute_nn_attributions' in self.trial_config) and self.trial_config['compute_nn_attributions']:
            self.compute_neural_net_attributions()
        if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers']:
            self.pretrain_leakage_localization_classifiers()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_leakage_localization()
        self.plot_leakage_assessments()