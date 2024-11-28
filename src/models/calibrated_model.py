import numpy as np
import torch
from torch import nn, optim

class CalibratedModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.input_shape = self.classifier.input_shape
        self.output_classes = self.classifier.output_classes
        self.noise_conditional = self.classifier.noise_conditional

        if self.noise_conditional:
            self.to_unsquashed_temperature = nn.Linear(np.prod(self.input_shape), 1)
            self.to_unsquashed_temperature.requires_grad_(False)
            nn.init.constant_(self.to_unsquashed_temperature.weight, 0)
            nn.init.constant_(self.to_unsquashed_temperature.bias, np.log(np.exp(1)-1))
        else:
            self.unsquashed_temperature = nn.Parameter(
                torch.tensor(np.log(np.exp(1)-1), dtype=torch.float), requires_grad=False
            )
    
    def uncalibrated_forward(self, *args):
        return self.classifier(*args)
    
    def calibrated_forward(self, *args):
        logits = self.classifier(*args)
        if self.noise_conditional:
            (_, noise) = args
            unsquashed_temperature = self.to_unsquashed_temperature(noise)
            temperature = nn.functional.softplus(unsquashed_temperature).view(-1, 1)
        else:
            temperature = nn.functional.softplus(self.unsquashed_temperature)
        return logits/temperature
    
    def calibrate_temperature(self, *args, **kwargs):
        if self.noise_conditional:
            self._calibrate_temperature_conditional(*args, **kwargs)
        else:
            self._calibrate_temperature_unconditional(*args, **kwargs)
    
    def _calibrate_temperature_unconditional(self, dataloader):
        logits, targets = [], []
        for trace, target in dataloader:
            targets.append(target)
            with torch.no_grad():
                logits.append(self.classifier(trace))
        logits, targets = map(lambda x: torch.cat(x, dim=0), (logits, targets))
        self.unsquashed_temperature.requires_grad_(True)
        optimizer = optim.LBFGS([self.unsquashed_temperature], line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                logits/nn.functional.softplus(self.unsquashed_temperature).view(-1, 1), targets
            )
            loss.backward()
            return loss
        optimizer.step(closure)
        self.unsquashed_temperature.requires_grad_(False)
    
    def _calibrate_temperature_conditional(self, dataloader, noise_generator, noise_samples=1):
        logits, targets, noises = [], [], []
        for trace, target in dataloader:
            for _ in range(noise_samples):
                with torch.no_grad():
                    noise = noise_generator(trace)
                    logits.append(self.classifier(noise*trace, noise))
                    noises.append(noise)
                    targets.append(target)
        logits, targets, noises = map(lambda x: torch.cat(x, dim=0), (logits, targets, noises))
        self.to_unsquashed_temperature.requires_grad_(True)
        optimizer = optim.LBFGS(self.to_unsquashed_temperature.parameters(), line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                logits/nn.functional.softplus(self.to_unsquashed_temperature(noises)), targets
            )
            loss.backward()
            return loss
        optimizer.step(closure)
        self.to_unsquashed_temperature.requires_grad_(False)