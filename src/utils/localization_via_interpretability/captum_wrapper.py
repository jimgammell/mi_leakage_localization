from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from captum.attr import InputXGradient, FeatureAblation

class SqueezeOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).squeeze()

def compute_input_x_gradient(training_module, dataset):
    model = training_module.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    input_x_gradient = InputXGradient(SqueezeOutput(model))
    dataloader = DataLoader(dataset, batch_size=256)
    count = 0
    trace_shape = model.input_shape
    attribution_map = torch.zeros(*trace_shape, device=device)
    for trace, target in dataloader:
        trace, target = trace.to(device), target.to(device)
        trace.requires_grad = True
        batch_size = trace.size(0)
        batch_attribution = input_x_gradient.attribute(trace, target=target).abs().mean(axis=0).detach()
        attribution_map = (count/(count+batch_size))*attribution_map + (batch_size/(count+batch_size))*batch_attribution
        count += batch_size
    model = model.cpu()
    return attribution_map.cpu().numpy()

@torch.no_grad()
def compute_feature_ablation_map(training_module, dataset):
    model = training_module.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    ablator = FeatureAblation(SqueezeOutput(model))
    dataloader = DataLoader(dataset, batch_size=1000)
    count = 0
    trace_shape = model.input_shape
    attribution_map = torch.zeros(*trace_shape, device=device)
    for trace, target in tqdm(dataloader):
        trace, target = trace.to(device), target.to(device)
        batch_size = trace.size(0)
        batch_attribution = ablator.attribute(trace, target=target, perturbations_per_eval=10).abs().mean(axis=0).detach()
        attribution_map = (count/(count+batch_size))*attribution_map + (batch_size/(count+batch_size))*batch_attribution
        count += batch_size
    model = model.cpu()
    return attribution_map.cpu().numpy()