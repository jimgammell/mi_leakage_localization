from typing import *
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils.metrics import get_rank

@torch.no_grad()
def compute_dnn_performance_auc(
    dataloader: DataLoader,
    dnn: nn.Module,
    leakage_assessment: np.ndarray,
    device: Optional[str] = None,
    cluster_count: int = 100
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dnn = dnn.to(device)
    traces, labels = next(iter(dataloader))
    traces, labels = traces.to(device), labels.to(device)
    timesteps_per_trace = traces.shape[-1]
    indices = leakage_assessment.reshape(-1).argsort()
    if not len(indices) % cluster_count == 0:
        indices = np.concatenate([
            indices, indices[:-(cluster_count - (len(indices)%cluster_count))]
        ])
    indices = torch.tensor(indices.reshape(cluster_count, -1), dtype=torch.long)
    mask = torch.zeros(1, timesteps_per_trace, dtype=torch.float, device=device)
    ranks = []
    for index_cluster in indices:
        mask[:, index_cluster] = 1.
        masked_traces = mask.unsqueeze(0)*traces
        logits = dnn(masked_traces)
        rank = get_rank(logits, labels).mean()
        ranks.append(rank)
    auc = np.mean(ranks)
    return auc