import torch
from torch.utils.data import Dataset

from .functional import *
from utils.performance_correlation import soft_kendall_tau

class TemplateAttack:
    def __init__(self, partitions, partitions_per_pass: int = 50):
        self.partitions = partitions
        self.partitions_per_pass = partitions_per_pass
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def has_profiled(self):
        return hasattr(self, 'class_count') and hasattr(self, 'means') and hasattr(self, 'Ls')
    
    def load_dataset(self, dataset):
        traces, labels = extract_dataset(dataset, self.partitions)
        traces = torch.tensor(traces, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.uint8, device=self.device)
        return traces, labels
    
    def profile(self, profiling_dataset: Dataset):
        traces, labels = self.load_dataset(profiling_dataset)
        self.class_count = get_class_count(labels)
        self.means, covs = fit_means_and_covs(traces, labels, self.class_count)
        self.Ls = get_choldecomp_covs(covs)
        self.log_p_y = get_log_p_y(labels)
        assert torch.all(torch.isfinite(self.means))
        assert torch.all(torch.isfinite(covs))
        assert torch.all(torch.isfinite(self.Ls))
        assert torch.all(torch.isfinite(self.log_p_y))
    
    def get_nll(self, attack_dataset: Dataset):
        assert self.has_profiled()
        traces, labels = self.load_dataset(attack_dataset)
        log_p_x_given_y = []
        for i in range(int(np.ceil(traces.size(0)/self.partitions_per_pass))):
            start_idx = i*self.partitions_per_pass
            end_idx = min((i+1)*self.partitions_per_pass, traces.size(0))
            log_p_x_given_y.append(get_log_p_x_given_y(
                traces[start_idx:end_idx, ...], self.means[start_idx:end_idx, ...], self.Ls[start_idx:end_idx, ...]
            ))
        log_p_x_given_y = torch.cat(log_p_x_given_y, dim=0)
        predictions = (log_p_x_given_y + self.log_p_y.reshape(1, self.class_count, 1)).permute(0, 2, 1)
        labels = labels.unsqueeze(0).expand(predictions.size(0), -1).to(torch.long)
        nll = -predictions.gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        return nll.cpu().numpy()
    
    def get_mean_gmm_ktcc(self, attack_dataset: Dataset): # Takes about 1.2 seconds to run this on the DPAv4 dataset
        nll = self.get_nll(attack_dataset)
        metric = soft_kendall_tau(nll.mean(axis=-1), np.arange(len(nll)), x_var=nll.var(axis=-1), y_var=np.zeros((len(nll),)))
        return metric