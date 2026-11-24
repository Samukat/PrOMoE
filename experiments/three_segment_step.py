"""Minimal MoE-PrO example: three linear-Gaussian experts on a three-segment toy.

    python experiments/three_segment_step.py

Fits the model with the log score and saves a diagnostic figure to experiments/figs/.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from pcuq_moe_utils import (Prior, ExperimentConfigMoE, gate_weights,
                            gaussian_log_likelihood, log_score_term_moe,
                            run_experiments_moe)
from pcuq_plot import Plotter

jax.config.update("jax_enable_x64", True)

# --- data: y = 0 on [-6, 0), 5 on [0, 3), +/-5 on [3, 6], noise N(0, 0.4^2)
SIGMA = 0.4
rng = np.random.default_rng(0)
x = rng.uniform(-6, 6, 500)
mean = np.where(x < 0, 0.0, 5.0)
mean = np.where((x >= 3) & (rng.random(x.size) < 0.5), -5.0, mean)
y = mean + SIGMA * rng.normal(size=x.size)

# --- model: expert e is y ~ N(theta_0 + theta_1 x, SIGMA^2), prior N(0, 5 I)
PRIOR_SCALE = 5.0 ** 0.5
prior = Prior(
    sample=lambda key, shape, dtype: PRIOR_SCALE * jax.random.normal(key, shape, dtype),
    log_pdf=lambda th: jnp.sum(jax.scipy.stats.norm.logpdf(th, scale=PRIOR_SCALE)),
)


def log_lik(theta, xy):
    return gaussian_log_likelihood(theta, xy, SIGMA)


def sample_y(theta, xg, key, n):
    """Draws from one particle's predictive at the grid xg: (len(xg), n)."""
    mu = theta[0] + theta[1] * xg
    return mu[:, None] + SIGMA * jax.random.normal(key, (xg.shape[0], n))


# --- run: E experts with M particles each; lam = n * M is thesis lambda = 1
E, M = 3, 16
cfg = ExperimentConfigMoE(
    n_iter=30_000, E=E, M=M, d=2,
    step_size=1e-4,               # particles (Langevin)
    gate_step_size=1e-3,          # gate (Adam learning rate)
    gate_optimizer="adam",
    gate_burn_in=2_000,           # gate frozen while the particles settle
    lam=x.size * M,
    label=f"three segments, E={E}",
)

plotter = Plotter(
    data=(x, y), sample_y=sample_y, gate_weights=gate_weights,
    outdir=os.path.normpath(os.path.join(HERE, "..", "figs")),
    defaults=dict(t_range=(-6, 6), save=True),
)
results = run_experiments_moe(cfg, log_score_term_moe, prior,
                              jnp.asarray(x), jnp.asarray(y), log_lik, plotter)

res = results[cfg.label]
pi = np.asarray(gate_weights(res["phi_final"], jnp.asarray(x)).mean(0))
print("expert weights pi_e:", np.round(pi, 3))
print("posterior means per expert:\n",
      np.asarray(res["thetas_final"]).reshape(E, M, 2).mean(1).round(2))
