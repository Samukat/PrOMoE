"""MoE-PrO with quadratic experts on a sine wave (softmax-linear gate).

    python experiments/sine_quadratic.py

Data y = sin(pi x) + N(0, 0.2^2), x ~ U(-1, 1). Each expert is
y ~ N(theta_0 + theta_1 x + theta_2 x^2, 0.2^2), so each one can cover
one half of the period, and the gate decides where the switch is. Saves a diagnostic figure to experiments/figs/.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from pcuq_moe_utils import (Prior, ExperimentConfigMoE, gate_weights,
                            log_score_term_moe, run_experiments_moe)
from pcuq_plot import Plotter

jax.config.update("jax_enable_x64", True)

# --- data
SIGMA = 0.2
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 500)
y = np.sin(np.pi * x) + SIGMA * rng.normal(size=x.size)

# --- model: quadratic expert, prior N(0, 5 I) on (theta_0, theta_1, theta_2)
PRIOR_SCALE = 5.0 ** 0.5
prior = Prior(
    sample=lambda key, shape, dtype: PRIOR_SCALE * jax.random.normal(key, shape, dtype),
    log_pdf=lambda th: jnp.sum(jax.scipy.stats.norm.logpdf(th, scale=PRIOR_SCALE)),
)


def mean_fn(theta, x):
    return theta[0] + theta[1] * x + theta[2] * x ** 2


def log_lik(theta, xy):
    x_i, y_i = xy
    return jax.scipy.stats.norm.logpdf(y_i, mean_fn(theta, x_i), SIGMA)


def sample_y(theta, xg, key, n):
    """Draws from one particle's predictive at the grid xg: (len(xg), n)."""
    return mean_fn(theta, xg)[:, None] + SIGMA * jax.random.normal(key, (xg.shape[0], n))


# --- run: lam = n * M is thesis lambda = 1
E, M = 4, 16
cfg = ExperimentConfigMoE(
    n_iter=40_000, E=E, M=M, d=3,
    step_size=2e-4,               # particles (Langevin)
    gate_step_size=1e-2,          # gate (Adam learning rate)
    gate_optimizer="adam",
    gate_burn_in=2_000,
    lam=x.size * M,
    label=f"sine wave, quadratic experts, E={E}",
)

plotter = Plotter(
    data=(x, y), sample_y=sample_y, gate_weights=gate_weights,
    true_function=lambda t: np.sin(np.pi * t),
    outdir=os.path.normpath(os.path.join(HERE, "..", "figs")),
    defaults=dict(t_range=(-1, 1), save=True,
                  param_ix=(1, 2), param_labels=(r"$\theta_1$", r"$\theta_2$"),
                  param_titles=("linear", "quadratic")),
)
results = run_experiments_moe(cfg, log_score_term_moe, prior,
                              jnp.asarray(x), jnp.asarray(y), log_lik, plotter)

res = results[cfg.label]
pi = np.asarray(gate_weights(res["phi_final"], jnp.asarray(x)).mean(0))
print("expert weights pi_e:", np.round(pi, 3))
print("posterior means per expert (theta_0, theta_1, theta_2):\n",
      np.asarray(res["thetas_final"]).reshape(E, M, 3).mean(1).round(2))
