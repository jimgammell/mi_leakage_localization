# Implementation of the technique in https://eprint.iacr.org/2018/1196.pdf

import torch
from torch import nn
from torch.utils.data import DataLoader

def compute_gradvis(training_module, dataset):
    model = training_module.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    dataloader = DataLoader(dataset, batch_size=256)
    count = 0
    trace_shape = model.input_shape
    attribution_map = torch.zeros(*trace_shape)
    for trace, target in dataloader:
        trace, target = trace.to(device), target.to(device)
        batch_size = trace.size(0)
        trace.requires_grad = True
        logits = model(trace).squeeze()
        loss = nn.functional.cross_entropy(logits, target)
        model.zero_grad()
        loss.backward()
        trace_grad = trace.grad.detach().abs().mean(dim=0).cpu()
        attribution_map = (count/(count+batch_size))*attribution_map + (batch_size/(count+batch_size))*trace_grad
        count += batch_size
    model = model.cpu()
    return attribution_map.numpy()