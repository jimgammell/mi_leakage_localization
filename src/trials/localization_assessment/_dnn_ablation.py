import numpy as np
import torch

from utils.metrics.rank import get_rank

@torch.no_grad()
def _dnn_ablation(classifier, attack_dataloader, leakage_assessment, patch_size=10):
    datapoint_count = len(attack_dataloader.dataset)
    batch_size = attack_dataloader.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    trace, _ = next(iter(attack_dataloader))
    mask = torch.ones_like(trace[0, ...]).unsqueeze(0).to(device)
    leakage_ranking = leakage_assessment.argsort()[::-1]
    def compute_performance():
        logits = np.full((datapoint_count, classifier.output_classes), np.nan, np.float32)
        targets = np.full((datapoint_count,), -1, int)
        for batch_idx, (trace, target) in enumerate(attack_dataloader):
            trace, target = trace.to(device), target.to(device)
            trace = trace * mask
            _logits = classifier(trace)
            min_idx = batch_idx*batch_size
            max_idx = min((batch_idx+1)*batch_size, datapoint_count)
            logits[min_idx:max_idx, :] = _logits.cpu().numpy()
            targets[min_idx:max_idx] = target.cpu().numpy()
        rank = get_rank(logits, targets).astype(np.float32).mean()
        return rank
    ranks = np.empty(())