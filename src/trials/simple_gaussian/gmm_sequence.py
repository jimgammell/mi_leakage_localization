from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numba import jit, prange
from scipy.stats import norm
#from scipy.special import logsumexp
from scipy.integrate import nquad
from tqdm.auto import tqdm

from common import *

@jit(nopython=True)
def loggaussianpdf(x, mu, sigma):
    return -((x - mu)**2)/(2*sigma**2) - 0.5*np.log(2*np.pi*sigma**2)

@jit(nopython=True)
def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

@jit(nopython=True)
def get_log_p_x_mid_y(x, y, sigma=1):
    assert y in [-1, 1]
    return loggaussianpdf(x, y, sigma) #norm.logpdf(x, loc=y, scale=sigma)

@jit(nopython=True)
def get_log_p_x(x, sigma=1):
    return np.log(0.5) + logsumexp(np.array([(get_log_p_x_mid_y(x, y, sigma)).sum() for y in [-1, 1]]))

@jit(nopython=True, parallel=True)
def mc_get_dent_x(dim, sigma, count=int(1e8)):
    vals = np.full((count,), np.nan, dtype=np.float64)
    xx = sigma*np.random.randn(count, dim)
    for idx in prange(count):
        x = xx[idx] + 2*np.random.randint(low=0, high=2) - 1
        vals[idx] = -get_log_p_x(x, sigma) / count
    assert np.all(np.isfinite(vals))
    return vals.sum()

def sweep_conditional_mi(dim_vals=np.arange(1, 20), sigma_vals=np.logspace(-1, 1, 10)):
    results = {sigma_val: np.full((len(dim_vals),), np.nan, dtype=np.float64) for sigma_val in sigma_vals}
    for sigma_val in sigma_vals:
        h_x = np.full((len(dim_vals),), np.nan, dtype=np.float64)
        h_x_y = 0.5*np.log(2*np.pi*np.exp(1)*sigma_val**2)
        for idx, dim in enumerate(tqdm(dim_vals)):
            h_x[idx] = mc_get_dent_x(dim, sigma_val)
            if idx == 0:
                results[sigma_val][idx] = h_x[idx] - h_x_y
            else:
                results[sigma_val][idx] = h_x[idx] - h_x[idx-1] - h_x_y
        assert np.all(np.isfinite(h_x))
    return results

def plot_conditional_mi(results):
    fig, ax = plt.subplots(figsize=(1*PLOT_WIDTH, 1*PLOT_WIDTH))
    cmap = plt.cm.plasma
    norm = LogNorm(vmin=min(results.keys()), vmax=max(results.keys()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    for (sigma_val, trace), color in zip(results.items(), cmap(norm(list(results.keys())))):
        ax.plot(np.arange(1, len(trace)+1), trace, color=color)
    ax.axhline(np.log(2), label='$\mathbb{H}[Y]$', color='black', linestyle='--')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$\mathbb{I}[Y; X_1 \mid X_2, \dots, X_n]$')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.set_ylim(1e-3, 1.1*np.log(2))
    ax.legend()
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('$\sigma$')
    return fig