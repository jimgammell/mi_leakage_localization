import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import quad
import torch
from torch import nn, optim

from _common import *

RERUN_LAMBDA_SWEEP = False
RERUN_VARIANCE_SWEEP = False

def compute_mi(variance=1.):
    def integrand(x):
        def p_y(y):
            return 0.5
        def p_x_given_y(x, y):
            return np.exp(-0.5*((x-y)**2)/variance) / np.sqrt(2*np.pi*variance)
        def p_x_y(x, y):
            return p_y(y) * p_x_given_y(x, y)
        def log_p_x_given_y(x, y):
            return -0.5*((x-y)**2)/variance - 0.5*np.log(2*np.pi*variance)
        def log_p_x(x):
            return logsumexp([-0.5*(x**2)/variance, -0.5*((x-1)**2)/variance]) - np.log(2*np.sqrt(2*np.pi*variance))
        return sum(p_x_y(x, y) * (log_p_x_given_y(x, y) - log_p_x(x)) for y in [0, 1])
    y, err = quad(integrand, -np.inf, np.inf)
    return y

def get_minibatch(dims, batch_size, variance=1.):
    target = torch.randint(2, size=(batch_size,), device=DEVICE)
    trace = np.sqrt(variance)*torch.randn(batch_size, dims, device=DEVICE, dtype=torch.float32) + target.to(torch.float32).unsqueeze(-1)
    return trace, target

def _get_log_likelihoods(traces, targets, binary_masks, batch_size, variance=1.):
    targets = targets.to(torch.float32).unsqueeze(-1).repeat(1, traces.size(-1))
    log_likelihoods = (
        (-0.5*binary_masks*((traces-targets)**2)/variance).sum(dim=-1)
        - torch.logsumexp(torch.stack([(-0.5*binary_masks*((traces-traces.new_full(traces.shape, y))**2)/variance).sum(dim=-1) for y in [0, 1]], dim=-1), dim=-1)
    )
    return log_likelihoods
get_log_likelihoods = torch.compile(_get_log_likelihoods)

def get_loss_and_gradient(
        trace, target, unsquashed_obfuscation_weights, baseline_log_likelihood, log_likelihood_ema, l2_norm_penalty, batch_size,
        log_likelihoods=None, binary_mask=None, variance=1.
):
    gamma = nn.functional.sigmoid(unsquashed_obfuscation_weights)
    dims = gamma.size(-1)
    norm = gamma.norm(p=2)**2
    norm_grad = gamma*gamma*(1-gamma)
    if log_likelihoods is None:
        binary_mask = (1-gamma).repeat(batch_size, 1).bernoulli()
        log_likelihoods = get_log_likelihoods(trace, target, binary_mask, batch_size, variance=variance)
    else:
        assert binary_mask is not None
    if baseline_log_likelihood is None:
        baseline_log_likelihood = log_likelihoods.mean()
    else:
        baseline_log_likelihood = log_likelihood_ema*baseline_log_likelihood + (1-log_likelihood_ema)*log_likelihoods.mean()
    loss = 0.5*l2_norm_penalty*norm + log_likelihoods.mean()
    gamma = gamma.unsqueeze(0)
    gradient = (
        l2_norm_penalty*norm_grad + ((log_likelihoods-baseline_log_likelihood).unsqueeze(-1)*((1-binary_mask)*(1-gamma) - binary_mask*gamma)).mean(dim=0)
    )
    return loss, gradient

@torch.no_grad()
def train_weights(steps=80000, l2_norm_penalty=1.0, batch_size=8192, log_likelihood_ema=0.9, dims=1, use_mlp=False, variance=1.):
    loss_vals = np.empty((steps,), dtype=np.float32)
    unsquashed_obfuscation_weights = torch.zeros((dims,), requires_grad=True, device=DEVICE, dtype=torch.float)
    opt = optim.Adam([unsquashed_obfuscation_weights], lr=2e-4)
    if use_mlp:
        mlp = nn.Sequential(nn.Linear(2*dims, 512), nn.ReLU(), nn.Linear(512, 2)).to(DEVICE)
        mlp_opt = optim.Adam(mlp.parameters(), lr=2e-4)
        mlp_loss_vals = []
    baseline_log_likelihood = None
    for step_idx in tqdm(range(steps)):
        trace, target = get_minibatch(dims, batch_size, variance=variance)
        if use_mlp:
            with torch.set_grad_enabled(True):
                with torch.no_grad():
                    binary_mask = (1-nn.functional.sigmoid(unsquashed_obfuscation_weights)).repeat(batch_size, 1).bernoulli()
                logits = mlp(torch.cat([binary_mask*trace, binary_mask], dim=-1))
                loss = nn.functional.cross_entropy(logits, target)
                mlp_loss_vals.append(loss.detach().cpu().numpy())
                with torch.no_grad():
                    log_likelihoods = -nn.functional.cross_entropy(logits, target, reduction='none')
                mlp_opt.zero_grad()
                loss.backward()
                mlp_opt.step()
        else:
            log_likelihoods = None
            binary_mask = None
        loss, gradient = get_loss_and_gradient(
            trace, target, unsquashed_obfuscation_weights, baseline_log_likelihood, log_likelihood_ema, l2_norm_penalty, batch_size,
            log_likelihoods=log_likelihoods, binary_mask=binary_mask, variance=variance
        )
        loss_vals[step_idx] = loss
        unsquashed_obfuscation_weights.grad = gradient
        opt.step()
        loss_vals[step_idx] = loss.detach().cpu().numpy()
    gamma = nn.functional.sigmoid(unsquashed_obfuscation_weights).detach().cpu().numpy()
    if use_mlp:
        return gamma, loss_vals, mlp_loss_vals
    else:
        return gamma, loss_vals

baseline_mi = compute_mi()
with open(os.path.join(get_trial_dir(), 'baseline_mi.pickle'), 'wb') as f:
    pickle.dump(baseline_mi, f)
print(f'Numerically-computed mutual information value: {baseline_mi}')

l2_norm_penalties = np.logspace(-2, 2, 20)
if RERUN_LAMBDA_SWEEP:
    final_gammas = np.empty_like(l2_norm_penalties)
    final_mlp_gammas = np.empty_like(l2_norm_penalties)
    for idx, l2_norm_penalty in enumerate(l2_norm_penalties):
        mlp_gamma, mlp_loss_vals, cls_loss_vals = train_weights(l2_norm_penalty=l2_norm_penalty, dims=1, use_mlp=True)
        final_mlp_gammas[idx] = mlp_gamma[0]
        print(f'MLP gamma: {final_mlp_gammas[idx]}')
        gamma, loss_vals = train_weights(l2_norm_penalty=l2_norm_penalty, dims=1)
        final_gammas[idx] = gamma[0]
        print(f'Dist gamma: {final_gammas[idx]}')
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(mlp_loss_vals, color='red')
        axes[0].plot(cls_loss_vals, color='blue')
        axes[1].plot(loss_vals, color='red')
        fig.savefig(os.path.join(get_trial_dir(), f'losses_{idx}.png'))
        plt.close(fig)
    with open(os.path.join(get_trial_dir(), 'obf_weight_sweep_true_dist.pickle'), 'wb') as f:
        pickle.dump(final_gammas, f)
    with open(os.path.join(get_trial_dir(), 'obf_weight_sweep_mlp.pickle'), 'wb') as f:
        pickle.dump(final_mlp_gammas, f)
else:
    with open(os.path.join(get_trial_dir(), 'obf_weight_sweep_true_dist.pickle'), 'rb') as f:
        final_gammas = pickle.load(f)
    with open(os.path.join(get_trial_dir(), 'obf_weight_sweep_mlp.pickle'), 'rb') as f:
        final_mlp_gammas = pickle.load(f)
#fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
fig = plt.figure(figsize=(6.5, 4))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.1], hspace=0.4, wspace=0.4)
axes = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis('off')
#fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.4, hspace=0.4)
theoretical_gammas = baseline_mi*np.ones_like(final_gammas) / l2_norm_penalties
theoretical_gammas[theoretical_gammas > 1.0] = 1.0
axes[0].plot(l2_norm_penalties, theoretical_gammas, linestyle='-', color='red', label='Theoretical')
axes[0].plot(l2_norm_penalties, final_gammas, marker='o', markersize=3, linestyle='none', color='blue', label='Empirical (true $p_{Y \mid X}$)')
axes[0].plot(l2_norm_penalties, final_mlp_gammas, marker='+', markersize=7, linestyle='none', color='green', label='Empirical (MLP classifier)')
axes[0].set_xlabel('$\lambda$')
axes[0].set_ylabel('$\gamma^*$')
axes[0].set_title('Norm penalty sweep')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
#axes[0].legend(ncol=2, fontsize=8, markerscale=0.5, loc='center', bbox_to_anchor=(-1.0, 1.5))

variances = np.logspace(-4, 4, 20)
if RERUN_VARIANCE_SWEEP:
    final_gammas = np.empty_like(variances)
    final_mlp_gammas = np.empty_like(variances)
    for idx, variance in enumerate(variances):
        mlp_gamma, mlp_loss_vals, cls_loss_vals = train_weights(l2_norm_penalty=1., dims=1, use_mlp=True, variance=variance)
        final_mlp_gammas[idx] = mlp_gamma[0]
        print(f'MLP gamma: {final_mlp_gammas[idx]}')
        gamma, loss_vals = train_weights(l2_norm_penalty=1., dims=1, variance=variance)
        final_gammas[idx] = gamma[0]
        print(f'Dist gamma: {final_gammas[idx]}')
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(mlp_loss_vals, color='red')
        axes[0].plot(cls_loss_vals, color='blue')
        axes[1].plot(loss_vals, color='red')
        fig.savefig(os.path.join(get_trial_dir(), f'var_losses_{idx}.png'))
        plt.close(fig)
    with open(os.path.join(get_trial_dir(), 'var_sweep_true_dist.pickle'), 'wb') as f:
        pickle.dump(final_gammas, f)
    with open(os.path.join(get_trial_dir(), 'var_sweep_mlp.pickle'), 'wb') as f:
        pickle.dump(final_mlp_gammas, f)
else:
    with open(os.path.join(get_trial_dir(), 'var_sweep_true_dist.pickle'), 'rb') as f:
        final_gammas = pickle.load(f)
    with open(os.path.join(get_trial_dir(), 'var_sweep_mlp.pickle'), 'rb') as f:
        final_mlp_gammas = pickle.load(f)
mutual_informations = np.array([compute_mi(variance=variance) for variance in variances])
theoretical_gammas = mutual_informations
theoretical_gammas[theoretical_gammas > 1.0] = 1.0
axes[1].plot(variances, theoretical_gammas, linestyle='-', color='red', label='Theoretical')
axes[1].plot(variances, final_gammas, marker='o', markersize=3, linestyle='none', color='blue', label='Empirical (true $p_{Y \mid X}$)')
axes[1].plot(variances, final_mlp_gammas, marker='+', markersize=7, linestyle='none', color='green', label='Empirical (MLP classifier)')
axes[1].set_xlabel('$\sigma^2$')
axes[1].set_ylabel('$\gamma^*$')
axes[1].set_title('Variance sweep')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
handles, labels = [], []
for handle, label in zip(*axes[0].get_legend_handles_labels()):
    handles.append(handle)
    labels.append(label)
ax_legend.legend(handles, labels, loc='center', ncol=3)
fig.savefig(os.path.join(get_trial_dir(), 'sanity_check.pdf'))