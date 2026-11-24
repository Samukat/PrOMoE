"""Thesis Section 4.1: three-segment regression on a cubic, with linear experts.

    python experiments/three_segment_cubic.py                  # E = 1, 2, 3, 5, 200k steps
    python experiments/three_segment_cubic.py --E 3 --seed 4
    python experiments/three_segment_cubic.py --n_iter 20000   # quick check

Data follow Kock et al. (2026, Appendix B.1): x ~ U[-2, 2], y = x^3 + N(0, 0.1),
n = 1000. Each expert is the misspecified line y ~ N(theta_1 + theta_2 x, 0.1),
prior N(0, 5 I), log score with thesis lambda = 1, M = 32 particles per expert,
bandwidth h = 1, step 5e-5, gate frozen for 5000 steps then Adam at 1e-3.
For each E it prints the expert weights and held-out scores, and saves a
diagnostic figure to figs/.
"""
import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from pcuq_moe_utils import (Prior, ExperimentConfigMoE, gate_weights,
                            gaussian_log_likelihood, log_score_term_moe,
                            make_moe_score, make_step_moe,
                            make_objective_moe, objective_trace_moe)
from pcuq_plot import Plotter

jax.config.update("jax_enable_x64", True)

SIGMA = 0.1 ** 0.5
N_TRAIN, N_TEST = 1000, 10_000
PRIOR_SCALE = 5.0 ** 0.5
M = 32


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    return x, x ** 3 + SIGMA * rng.normal(size=n)


prior = Prior(
    sample=lambda key, shape, dtype: PRIOR_SCALE * jax.random.normal(key, shape, dtype),
    log_pdf=lambda th: jnp.sum(jax.scipy.stats.norm.logpdf(th, scale=PRIOR_SCALE)),
)


def log_lik(theta, xy):
    return gaussian_log_likelihood(theta, xy, SIGMA)


def sample_y(theta, xg, key, n):
    mu = theta[0] + theta[1] * xg
    return mu[:, None] + SIGMA * jax.random.normal(key, (xg.shape[0], n))


def run(E, n_iter, seed, thin):
    """Run one fit, keeping the particles and gate every `thin` steps."""
    x, y = make_data(N_TRAIN, 0)
    data = (jnp.asarray(x), jnp.asarray(y))
    cfg = ExperimentConfigMoE(
        n_iter=n_iter, E=E, M=M, d=2, step_size=5e-5,
        gate_step_size=1e-3, gate_optimizer="adam", gate_burn_in=5000,
        h=1.0, lam=N_TRAIN * M, seed=seed, label=f"cubic E={E} seed={seed}")
    step = make_step_moe(cfg, make_moe_score(log_score_term_moe, M), prior, data,
                         log_lik, cfg.lam, cfg.step_size, cfg.gate_step_size,
                         gate_burn_in=cfg.gate_burn_in)

    key, k_th, k_phi = jax.random.split(jax.random.key(seed), 3)
    th0 = prior.sample(k_th, (E * M, 2), jnp.float64)
    phi0 = cfg.phi_init_scale * jax.random.normal(k_phi, (E, 2), jnp.float64)
    keys = jax.random.split(key, n_iter).reshape(n_iter // thin, thin)

    def chunk(carry, ks):                       # `thin` steps, then record
        carry, _ = jax.lax.scan(lambda c, k: (step(c, k)[0], None), carry, ks)
        return carry, (carry[0], carry[1])

    z = jnp.zeros_like(phi0)
    (th, phi, *_), (th_traj, phi_traj) = jax.jit(
        lambda c: jax.lax.scan(chunk, c, keys))((th0, phi0, jnp.int32(0), z, z))
    th_traj = jnp.concatenate([th0[None], th_traj])
    phi_traj = jnp.concatenate([phi0[None], phi_traj])
    return cfg, data, th, phi, th_traj, phi_traj


def held_out(th, phi, E):
    """Mean held-out log predictive density and 90% interval coverage."""
    x, y = make_data(N_TEST, 1)
    th = np.asarray(th).reshape(E, M, 2)
    w = np.asarray(gate_weights(phi, jnp.asarray(x)))                   # (n, E)
    mu = th[None, :, :, 0] + th[None, :, :, 1] * x[:, None, None]        # (n, E, M)
    logp = -0.5 * np.log(2 * np.pi * SIGMA ** 2) - 0.5 * ((y[:, None, None] - mu) / SIGMA) ** 2
    lpd = np.logaddexp.reduce(logp + np.log(w + 1e-300)[:, :, None] - np.log(M), axis=(1, 2))

    rng = np.random.default_rng(0)
    e = (rng.random((len(x), 400))[..., None] > np.cumsum(w, 1)[:, None]).sum(-1).clip(0, E - 1)
    m = rng.integers(0, M, e.shape)
    draws = mu[np.arange(len(x))[:, None], e, m] + SIGMA * rng.normal(size=e.shape)
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=1)
    return lpd.mean(), np.mean((y >= lo) & (y <= hi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", default="1,2,3,5", help="comma-separated numbers of experts")
    ap.add_argument("--n_iter", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thin", type=int, default=500)
    a = ap.parse_args()

    x, y = make_data(N_TRAIN, 0)
    plotter = Plotter(data=(x, y), sample_y=sample_y, gate_weights=gate_weights,
                      true_function=lambda t: t ** 3, log_every=a.thin,
                      outdir=os.path.normpath(os.path.join(HERE, "..", "figs")),
                      defaults=dict(t_range=(-2, 2), save=True, show=False,
                                    show_expert_regions=True, show_theta_scatter=True,
                                    settings=None))
    for E in map(int, a.E.split(",")):
        cfg, data, th, phi, th_traj, phi_traj = run(E, a.n_iter, a.seed, a.thin)
        objective = make_objective_moe(cfg, log_score_term_moe, prior, data, log_lik)
        (obj, _, _), _, _, _ = objective_trace_moe(th_traj, phi_traj, objective,
                                                   max_points=th_traj.shape[0])
        plotter.plot(th_traj, obj, np.asarray(th).reshape(E, M, 2), phi,
                     title=cfg.label, obj_text="score + KL (MoE)")

        pi = np.asarray(gate_weights(phi, data[0]).mean(0))
        lpd, cover = held_out(th, phi, E)
        print(f"E={E}: pi {np.round(pi, 3)}  held-out lpd {lpd:.3f}  "
              f"90% coverage {cover:.3f}", flush=True)
