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
from datasets.aes_hd import AES_HD
from datasets.ed25519_wolfssl import ED25519
from datasets.one_truth_prevails import OneTruthPrevails
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution
from training_modules import SupervisedTrainer, LeakageLocalizationTrainer
from training_modules.supervised_deep_sca.plot_things import plot_hparam_sweep

class Trial:
    def __init__(self,
        dataset_name: Literal['dpav4', 'ascadv1_fixed', 'ascadv1_variable', 'otiait', 'otp', 'aes_hd'],
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
        self.supervised_hparam_sweep_dir = os.path.join(self.logging_dir, 'supervised_hparam_sweep')
        os.makedirs(self.supervised_hparam_sweep_dir, exist_ok=True)
        
        print('Constructing datasets...')
        if self.dataset_name == 'dpav4':
            self.profiling_dataset = DPAv4(root=trial_config['data_dir'], train=True)
            self.attack_dataset = DPAv4(root=trial_config['data_dir'], train=False)
        elif self.dataset_name == 'ascadv1_fixed':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif self.dataset_name == 'ascadv1_variable':
            self.profiling_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=True, train=True)
            self.attack_dataset = ASCADv1(root=trial_config['data_dir'], variable_keys=False, train=False)
        elif self.dataset_name == 'aes_hd':
            self.profiling_dataset = AES_HD(root=trial_config['data_dir'], train=True)
            self.attack_dataset = AES_HD(root=trial_config['data_dir'], train=False)
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
    
    def run_supervised_hparam_sweep(self):
        if not os.path.exists(os.path.join(self.supervised_hparam_sweep_dir, 'results.pickle')):
            print('Running supervised hparam sweep...')
            supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=self.trial_config['supervised_training_kwargs'])
            supervised_trainer.hparam_tune(logging_dir=self.supervised_hparam_sweep_dir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
            print('\tDone.')
        else:
            print('Found existing supervised hparam sweep.')
        self.optimal_hparams = plot_hparam_sweep(self.supervised_hparam_sweep_dir)
    
    def train_supervised_model(self):
        for seed in range(self.seed_count):
            subdir = os.path.join(self.supervised_model_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            if not os.path.exists(os.path.join(subdir, 'final_checkpoint.ckpt')):
                print('Training supervised model...')
                training_module_kwargs = copy(self.trial_config['supervised_training_kwargs'])
                training_module_kwargs.update(self.optimal_hparams)
                supervised_trainer = SupervisedTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=training_module_kwargs)
                supervised_trainer.run(logging_dir=subdir, max_steps=self.trial_config['max_classifiers_pretrain_steps'])
                print('\tDone.')
            else:
                print('Found pretrained supervised model.')
    
    def compute_neural_net_attributions(self, wouters_zaid_model=None):
        profiling_dataloader = DataLoader(self.profiling_dataset, shuffle=False, batch_size=1024)
        gradviss, saliencies, occlusions, inputxgrads, lrps = [], [], [], [], []
        for seed in range(self.seed_count):
            subdir = os.path.join(self.nn_attr_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            nn_attributor = NeuralNetAttribution(profiling_dataloader, os.path.join(self.supervised_model_dir, f'seed={seed}') if wouters_zaid_model is None else wouters_zaid_model, seed=seed)
            to_name = lambda x: x if wouters_zaid_model is None else f'zaid_{x}' if 'Zaid' in wouters_zaid_model else f'wouters_{x}' if 'Wouters' in wouters_zaid_model else None
            assert to_name('') is not None
            if wouters_zaid_model is None and not os.path.exists(os.path.join(subdir, to_name('lrp.npy'))):
                print('Computing LRP...')
                lrp = nn_attributor.compute_lrp().reshape(-1)
                np.save(os.path.join(subdir, to_name('lrp.npy')), lrp)
                print('\tDone.')
            elif wouters_zaid_model is None:
                lrp = np.load(os.path.join(subdir, to_name('lrp.npy')))
                print('Found precomputed LRP.')
            if not os.path.exists(os.path.join(subdir, to_name('gradvis.npy'))):
                print('Computing GradVis...')
                gradvis = nn_attributor.compute_gradvis().reshape(-1)
                np.save(os.path.join(subdir, to_name('gradvis.npy')), gradvis)
                print('\tDone.')
            else:
                gradvis = np.load(os.path.join(subdir, to_name('gradvis.npy')))
                print('Found precomputed GradVis.')
            if not os.path.exists(os.path.join(subdir, to_name('saliency.npy'))):
                print('Computing saliency...')
                saliency = nn_attributor.compute_saliency().reshape(-1)
                np.save(os.path.join(subdir, to_name('saliency.npy')), saliency)
                print('\tDone.')
            else:
                saliency = np.load(os.path.join(subdir, to_name('saliency.npy')))
                print('Found precomputed saliency.')
            r"""if not os.path.exists(os.path.join(subdir, to_name('occlusion.npy'))):
                print('Computing occlusion...')
                occlusion = nn_attributor.compute_occlusion().reshape(-1)
                np.save(os.path.join(subdir, to_name('occlusion.npy')), occlusion)
                print('\tDone.')
            else:
                occlusion = np.load(os.path.join(subdir, to_name('occlusion.npy')))
                print('Found precomputed occlusion.')"""
            if not os.path.exists(os.path.join(subdir, to_name('inputxgrad.npy'))):
                print('Computing inputxgrad...')
                inputxgrad = nn_attributor.compute_inputxgrad().reshape(-1)
                np.save(os.path.join(subdir, to_name('inputxgrad.npy')), inputxgrad)
                print('\tDone.')
            else:
                inputxgrad = np.load(os.path.join(subdir, to_name('inputxgrad.npy')))
                print('Found precomputed inputxgrad.')
            plot_leakage_assessment(gradvis, os.path.join(subdir, to_name('gradvis.png')))
            plot_leakage_assessment(saliency, os.path.join(subdir, to_name('saliency.png')))
            #plot_leakage_assessment(occlusion, os.path.join(subdir, to_name('occlusion.png')))
            plot_leakage_assessment(inputxgrad, os.path.join(subdir, to_name('inputxgrad.png')))
            if wouters_zaid_model is None:
                plot_leakage_assessment(lrp, os.path.join(subdir, to_name('lrp.png')))
                lrps.append(lrp)
            gradviss.append(gradvis)
            saliencies.append(saliency)
            #occlusions.append(occlusion)
            inputxgrads.append(inputxgrad)
        setattr(self, to_name('nn_attr_assessments'), {
            to_name('gradvis'): np.stack(gradviss), to_name('saliency'): np.stack(saliencies), to_name('inputxgrad'): np.stack(inputxgrads),# to_name('occlusion'): np.stack(occlusions)
            **({to_name('lrp'): np.stack(lrps)} if wouters_zaid_model is None else {})
        })
    
    def pretrain_leakage_localization_classifiers(self):
        if not os.path.exists(os.path.join(self.ll_classifiers_pretrain_dir, 'final_checkpoint.ckpt')):
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
                pretrained_classifiers_logging_dir=self.ll_classifiers_pretrain_dir if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers'] else None,
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
        if hasattr(self, 'zaid_nn_attr_assessments'):
            leakage_assessments.update(self.zaid_nn_attr_assessments)
        if hasattr(self, 'wouters_nn_attr_assessments'):
            leakage_assessments.update(self.wouters_nn_attr_assessments)
        if hasattr(self, 'leakage_localization_assessments'):
            leakage_assessments.update(self.leakage_localization_assessments)
        return leakage_assessments
    
    def plot_leakage_assessments(self):
        leakage_assessments = self.get_leakage_assessments()
        row_count = 2
        col_count = int(np.ceil(len(leakage_assessments)/row_count))
        fig, axes = plt.subplots(row_count, col_count, figsize=(col_count*PLOT_WIDTH, row_count*PLOT_WIDTH))
        for ax, (la_name, la) in zip(axes.flatten(), leakage_assessments.items()):
            if la.ndim == 1:
                ax.plot(la, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            elif la.ndim == 2:
                median = np.median(la, axis=0)
                min = np.min(la, axis=0)
                max = np.max(la, axis=0)
                #ax.errorbar(range(len(median)), median, yerr=[median-min, max-median], fmt='none', ecolor='red', label='min--max')
                ax.plot(median, color='blue', marker='.', linestyle='none', markersize=1, label='median', **PLOT_KWARGS)
                ax.legend()
            else:
                assert False
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakage of $X_t$')
            la_name = la_name.replace('_', r'\_')
            ax.set_title(f'Technique: {la_name}')
        for ax in axes.flatten()[len(leakage_assessments):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.logging_dir, 'leakage_assessments.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def __call__(self):
        if ('compute_first_order_stats' in self.trial_config) and self.trial_config['compute_first_order_stats']:
            self.compute_first_order_stats()
        if ('run_supervised_hparam_sweep' in self.trial_config) and self.trial_config['run_supervised_hparam_sweep']:
            self.run_supervised_hparam_sweep()
        if ('train_supervised_model' in self.trial_config) and self.trial_config['train_supervised_model']:
            self.train_supervised_model()
        if ('compute_nn_attributions' in self.trial_config) and self.trial_config['compute_nn_attributions']:
            self.compute_neural_net_attributions()
            if self.dataset_name == 'dpav4':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__DPAv4')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__DPAv4')
            elif self.dataset_name == 'ascadv1_fixed':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__ASCADv1f')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__ASCADv1f')
            elif self.dataset_name == 'aes_hd':
                self.compute_neural_net_attributions(wouters_zaid_model='ZaidNet__AES_HD')
                self.compute_neural_net_attributions(wouters_zaid_model='WoutersNet__AES_HD')
        if ('pretrain_classifiers' in self.trial_config) and self.trial_config['pretrain_classifiers']:
            self.pretrain_leakage_localization_classifiers()
        if ('run_leakage_localization' in self.trial_config) and self.trial_config['run_leakage_localization']:
            self.run_leakage_localization()
        self.plot_leakage_assessments()