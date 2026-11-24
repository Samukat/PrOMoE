from typing import NamedTuple, Callable
from dataclasses import dataclass

from matplotlib import pyplot as plt
import os
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

__all__ = [
    "Prior", "ExperimentConfigMoE",
    "gaussian_log_likelihood",
    "gate_weights", "log_mixture_weights",
    "make_moe_score", "log_score_term_moe",
    "make_kernel_terms", "make_step_moe",
    "run_experiment_moe", "run_experiments_moe",
    "make_objective_moe", "objective_trace_moe",
]


class Prior(NamedTuple):
    sample: Callable    # (key, shape, dtype) -> draws
    log_pdf: Callable   # (d,) -> scalar; summed over rows for a (P, d) input


# @dataclass
# class ExperimentConfig:
#     n_iter: int
#     n_particles: int = 32
#     step_size: float = 0.001
#     prior_scale: float = 1.0
#     seed: int = 0
#     lam: int = 1
#     label: str = ""
#     version: str = "standard"

@dataclass
class ExperimentConfigMoE:
    n_iter: int
    E: int = 4                        # number of experts
    M: int = 8                        # particles per expert
    d: int = 2                        # parameter dimension of theta
    step_size: float = 1e-3
    gate_step_size: float | None = None   # falls back to step_size
    gate_dim: int = 1                 # K gate covariates: phi is (E, K + 1)
    gate_burn_in: int = 0             # iterations with the gate frozen
    prior_scale: float = 1.0
    phi_init_scale: float = 0.1       # sd of the normal init on phi
    seed: int = 0
    h: float = 1.0                    # bandwidth for the log r_e term
    lam: float = 1.0
    label: str = ""
    pin_gate: bool = False  # new thign too
    gate_optimizer: str = "sgd"       # "sgd" or "adam" (gate only; particles stay SGLD)
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8


################################################################################
# LOG SCORING - NOT MOE
################################################################################

LOG_2PI = jnp.log(2.0 * jnp.pi)  # precompute for speeeeed


def gaussian_log_likelihood(theta, ty, sigma):
    """Compute the log-likelihood of a datum under a linear-Gaussian model.

    Evaluates log p_theta(y | t) = log N(y; theta[0] + theta[1] * t, sigma^2).

    Args:
        theta: Array of shape (2,), model parameters [intercept, slope].
        ty: Tuple (t_i, y_i) of scalars, the predictor and response.
        sigma: Scalar observation noise standard deviation.

    Returns:
        Scalar log-likelihood log p_theta(y_i | t_i).
    """
    t, y = ty
    resid = y - (theta[0] + theta[1] * t)
    return -0.5 * (LOG_2PI + 2.0 * jnp.log(sigma) + (resid / sigma) ** 2)


# THIS IS FOR TESTING THE NEW TYPE OF GATING
def gaussian_log_likelihood_sigmoid(theta, ty, sigma):
    """log N(y; sigma(theta[0] + theta[1] * t), sigma_noise^2).

    Second-order strongly identifiable: d2mu/dtheta_u dtheta_v is nonzero,
    unlike the linear expert where it vanishes identically.
    """
    t, y = ty
    mu = jax.nn.sigmoid(theta[0] + theta[1] * t)
    resid = y - mu
    return -0.5 * (LOG_2PI + 2.0 * jnp.log(sigma) + (resid / sigma) ** 2)


################################################################################
# GATING
################################################################################
#
# The commented out funcitons are used for debugging and visualisng and are the
# acutal weighting functions before logging and taking the deriviative
#
# def mixture_weights(phi, t, M):
#     """omega_p(i) = w_{e(p)}(t_i) / M, flattened over p = (e, a). -> (n, E*M), rows sum to 1."""
#     w = gate_weights(phi, t)                                  # (n, E)
#     return jnp.repeat(w, M, axis=1) / M                       # (n, E*M)

# def gate_design(phi, data_x): This one is for quadratic gates
#     """(n, 2K + 1) gate design [x_1^2..x_K^2, x_1..x_K, 1].

#     K = (phi.shape[1] - 1) // 2.  A 1-D data_x is the single gate feature
#     (K must be 1).  A 2-D data_x has its first K columns read as the gate
#     covariates, so the likelihood can still see the full covariate row.
#     """
#     K = (phi.shape[1] - 1) // 2
#     feats = data_x[:, None] if data_x.ndim == 1 else data_x[:, :K]
#     return jnp.concatenate(
#         [feats ** 2, feats, jnp.ones_like(feats[:, :1])], axis=1
#     )


def gate_design(phi, data_x):
    """(n, K + 1) gate design [x_1..x_K, 1], K = phi.shape[1] - 1.

    A 1-D data_x is the single gate feature (K must be 1). A 2-D data_x has
    its first K columns read as the gate covariates, so the likelihood can
    still see the full covariate row.
    """
    K = phi.shape[1] - 1
    feats = data_x[:, None] if data_x.ndim == 1 else data_x[:, :K]
    return jnp.concatenate([feats, jnp.ones_like(feats[:, :1])], axis=1)


def gate_weights(phi, t):
    """Softmax gate.  phi: (E, K + 1) [slopes, intercept] -> (n, E).

    z_e(t) = t . phi[e, :K] + phi[e, K];  w_e(t) = softmax_e z_e(t)
    E = 1 gives an all-ones column, i.e. the single-population case.
    t may be (n,) (K = 1) or a 2-D data_x whose first K columns feed the gate.
    """
    t_aug = gate_design(phi, t)                               # (n, K + 1)
    return jax.nn.softmax(t_aug @ phi.T, axis=1)              # (n, E)


def log_mixture_weights(phi, data_x, M):
    # This returns the log w_e(t_i) = log [ softmax(a*data_x + b)]
    # This is for speed up and no issues as 'log_softmax' is a function rather
    # than logging the output of the standard weighting function
    """log omega_p(i) = log w_{e(p)}(t_i) - log M, flattened over p = (e, a). -> (n, E*M)."""
    t_aug = gate_design(phi, data_x)                               # (n, K + 1)
    log_w = jax.nn.log_softmax(t_aug @ phi.T, axis=1)             # (n, E)
    return jnp.repeat(log_w, M, axis=1) - jnp.log(M)              # (n, E*M)


################################################################################
# SCORING FOR MOE
################################################################################


def make_moe_score(base_score_term, M):
    """Wrap a log_omega-taking score term into (thetas, phi, data, ll, lam, key) form.
    - data here is data_x, data_y

    Base score term protocol (shared by the log score and the MMD score):
        base_score_term(particles, data, log_likelihood, lam, log_omega,
                        key=None, M=None) -> scalar
    `key` is a PRNG key for stochastic scores (MMD Monte Carlo draws, data
    minibatches); `M` is particles-per-expert so the score can see the block
    structure of log_omega. The log score ignores both.
    """
    def score(thetas, phi, data, log_likelihood, lam, key=None):
        data_x, _ = data
        return base_score_term(thetas, data, log_likelihood, lam,
                               log_mixture_weights(phi, data_x, M),
                               key=key, M=M)
    return score


def log_score_term_moe(particles, xs, log_likelihood, lam, log_omega=None,
                       key=None, M=None, per_row=False):
    """Log-score term (A) of the PrO objective, with optional gate weights.

    log_omega: Array (n, p) of log mixture weights log omega_j(t_i), rows
        logsumexp to 0. If None, falls back to uniform log(1/p).
    key, M: unused here; part of the shared base-score protocol (see
        make_moe_score) so this and the MMD score are interchangeable.
    per_row: return the (n,) per-row scores instead of their mean. The rows
        are independent here, so the caller can do its own weighting in one
        call (see e1_fit.make_turbine_weighted); mean(per_row) is the default.
    """
    ll = jax.vmap(jax.vmap(log_likelihood, in_axes=(None, 0)),
                  in_axes=(0, None))(particles, xs)                # (p, n)

    if log_omega is None:
        p = particles.shape[0]
        log_pred = logsumexp(ll, axis=0) - jnp.log(p)              # (n,)
    else:
        log_pred = logsumexp(ll + log_omega.T, axis=0)             # (n,)

    s = -lam * log_pred                                            # (n,)
    return s if per_row else jnp.mean(s)


log_score_term_moe.per_row_capable = True


################################################################################
# KERNEL TERMS (shared by the step and the objective trace)
################################################################################

def make_kernel_terms(E, M, h, prior):
    """Build kappa_e and grad_theta log r_e from one shared kernel evaluation.

    Factored out of make_step_moe so the monitored objective and the flow read
    the same kappa. If you change the kernel here, both move together.
    """
    P = E * M
    h2 = h * h
    NEG = -jnp.inf
    OFF = ~jnp.eye(P, dtype=bool)      # leave-one-out, pooled
    OFF_M = ~jnp.eye(M, dtype=bool)    # leave-one-out, within an expert

    def sq_dists(X):
        """(n, n) squared distances via Gram matrix: O(n^2) memory, not O(n^2 d)."""
        n2 = (X * X).sum(-1)
        return jnp.maximum(n2[:, None] + n2[None, :] - 2.0 * (X @ X.T), 0.0)

    def kernel_terms(X, log_pi_p, with_grad=True):
        """X: (P, d) expert-major. log_pi_p: (P,) log pi of each particle's expert.

        Returns (kap, grad_log_r) with kap: (E,) and grad_log_r: (P, d) or None.
        kap_e = mean_{i in e} [log q_MoE(x_i) - log q_0(x_i)].
        """
        d = X.shape[-1]
        logK = -sq_dists(X) / (2.0 * h2)                        # (P, P)

        # kappa's logsumexp and a_pool's normaliser are the same object.
        logits = jnp.where(OFF, logK + log_pi_p[None, :], NEG)  # (P, P)
        lse = logsumexp(logits, axis=1)                         # (P,)

        log_qmoe = lse - jnp.log(M) - 0.5 * d * jnp.log(2.0 * jnp.pi * h2)
        log_q0 = jax.vmap(prior.log_pdf)(X)
        kap = (log_qmoe - log_q0).reshape(E, M).mean(1)         # (E,)

        if not with_grad:
            return kap, None

        a_pool = jnp.exp(logits - lse[:, None])                 # rows sum to 1
        pool_mean = a_pool @ X                                  # (P, d)

        # sum_j (a_own - a_pool)_ij (x_j - x_i)
        #   = A_own X - A_pool X - x_i (rowsum(A_own) - rowsum(A_pool)).
        # M > 1: both row sums are 1, so the x_i terms cancel.
        # M == 1: a_own is identically 0, so the expression collapses to
        #         x_i - (A_pool X)_i.
        if M > 1:
            Xe = X.reshape(E, M, d)
            logK_own = -jax.vmap(sq_dists)(Xe) / (2.0 * h2)     # (E, M, M)
            a_own = jax.nn.softmax(jnp.where(OFF_M, logK_own, NEG), axis=-1)
            own_mean = jnp.einsum("eij,ejd->eid", a_own, Xe).reshape(P, d)
        else:
            own_mean = X

        return kap, (own_mean - pool_mean) / h2

    return kernel_terms


def expert_pi(phi, data_x):
    """(E,) mixing weights pi_e = mean_x w_e(x)."""
    return gate_weights(phi, data_x).mean(0)


################################################################################
# STEP MOE
################################################################################

def make_step_moe(cfg, score_term, prior, data, log_likelihood, lam, step_size,
                  gate_step_size=None, noise_fn=jax.random.normal,
                  gate_burn_in=0):
    """One SGLD step for MoE-PVI: particles by Langevin, gate by plain SGD.

    gate_burn_in: phi is held at its init for this many leading iterations
    (the particles still move, and still feel pi_e from the frozen gate).
    cfg.gate_optimizer == "adam": phi is updated by Adam on the same gate drift,
    with gate_step_size as the Adam learning rate. The moments only start
    accumulating once burn-in ends. The particle update is unchanged.

    The carry is (thetas, phi, i, m, v) with i the step counter and (m, v) the
    Adam moments for phi (unused, and left at zero, under SGD); the per-step
    output is still ((thetas, phi), step_size, 1).
    """

    if gate_step_size is None:
        gate_step_size = step_size

    E, M = cfg.E, cfg.M
    pin_gate = cfg.pin_gate
    use_adam = cfg.gate_optimizer == "adam"
    _n_gate = max(cfg.n_iter - gate_burn_in, 1)
    def _gate_sched(i):
        # TOY_GATE_SCHED=warm: log-ramp 1e-3 -> 1 over the first 20% of gate steps,
        # hold to 60%, then log-decay back to 1e-3 at the end. Otherwise constant 1.
        if os.environ.get("TOY_GATE_SCHED") != "warm":
            return 1.0
        uu = jnp.clip((i - gate_burn_in) / _n_gate, 0.0, 1.0)
        up = 10.0 ** (-3.0 + 3.0 * jnp.minimum(uu / 0.2, 1.0))
        down = 10.0 ** (-3.0 * jnp.clip((uu - 0.6) / 0.4, 0.0, 1.0))
        return up * down
    if cfg.gate_optimizer not in ("sgd", "adam"):
        raise ValueError(f"unknown gate_optimizer {cfg.gate_optimizer!r}")
    b1, b2, eps = cfg.adam_b1, cfg.adam_b2, cfg.adam_eps
    kernel_terms = make_kernel_terms(E, M, cfg.h, prior)

    obj_grad_fn = jax.grad(score_term, argnums=(0, 1))
    prior_log_grad_fn = jax.grad(prior.log_pdf, 0)

    def gate_pi(p):
        return expert_pi(p, data[0])

    def step(carry, key):
        thetas, phi, i, m, v = carry

        # one gate forward pass; the vjp replaces the second forward + grad.
        # pi_e comes back concrete, so it is already detached from phi.
        pi_e, gate_vjp = jax.vjp(gate_pi, phi)                  # (E,)
        pi_p = jnp.repeat(pi_e, M)[:, None]                     # (P, 1)
        # guard exact zeros
        log_pi_p = jnp.log(pi_p.ravel() + 1e-30)

        kap, grad_log_r = kernel_terms(jax.lax.stop_gradient(thetas), log_pi_p)

        # score key is derived from the step key; the noise still uses `key`
        # itself, so runs with a deterministic score are unchanged.
        k_score = jax.random.fold_in(key, 1)
        g_theta, g_phi = obj_grad_fn(thetas, phi, data, log_likelihood, lam,
                                     k_score)
        drift = g_theta - pi_p * (prior_log_grad_fn(thetas) + grad_log_r)

        # d/dphi sum_e kap_e * pi_e(phi), with kap held constant
        gate_drift = g_phi + cfg.M * gate_vjp(kap)[0]
        if pin_gate:
            gate_drift = gate_drift.at[-1].set(0.0)

        noise = jnp.sqrt(2.0 * step_size * pi_p) * noise_fn(key, thetas.shape)
        thetas = thetas - step_size * drift + noise
        # frozen gate during burn-in: the update is scaled to exactly zero
        active = i >= gate_burn_in
        gate_lr = jnp.where(active, gate_step_size * _gate_sched(i), 0.0)
        if use_adam:
            # moments frozen during burn-in; t counts Adam steps taken
            m = jnp.where(active, b1 * m + (1.0 - b1) * gate_drift, m)
            v = jnp.where(active, b2 * v + (1.0 - b2) * gate_drift ** 2, v)
            t = jnp.maximum(i - gate_burn_in + 1, 1).astype(phi.dtype)
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)
            phi = phi - gate_lr * m_hat / (jnp.sqrt(v_hat) + eps)
        else:
            phi = phi - gate_lr * gate_drift

        return (thetas, phi, i + 1, m, v), ((thetas, phi), step_size, 1)

    return step

################################################################################
# MONITORED OBJECTIVE - not for any computation just visuals
################################################################################


# NEED TO DOUBLE CHECK THIS LATER
def make_objective_moe(cfg, base_score_term, prior, data, log_likelihood):
    """The functional the flow actually descends, plus its decomposition.

        F(thetas, phi) = score_term(thetas, phi)
                         + sum_e pi_e(phi) * sum_{i in e} [log q_MoE(x_i) - log q_0(x_i)]
                       = score_term + M * sum_e pi_e(phi) * kappa_e

    Both the pi_e weight and the log r_e channel are present.


    Returns objective(thetas, phi, key=None) -> (total, score, kl).
    `key` only matters for stochastic scores (MMD with the MC backend).
    """
    score_term = make_moe_score(base_score_term, cfg.M)
    kernel_terms = make_kernel_terms(cfg.E, cfg.M, cfg.h, prior)
    data_x = data[0]

    def objective(thetas, phi, key=None):
        score = score_term(thetas, phi, data, log_likelihood, cfg.lam, key)
        pi_e = expert_pi(phi, data_x)                           # (E,)
        log_pi_p = jnp.log(jnp.repeat(pi_e, cfg.M) + 1e-30)     # (P,)
        kap, _ = kernel_terms(thetas, log_pi_p, with_grad=False)
        kl = cfg.M * jnp.sum(pi_e * kap)
        return score + kl, score, kl

    return objective


def objective_trace_moe(theta_traj, phi_traj, objective_fn, max_points=500,
                        seed=0):
    """Evaluate the objective along a thinned trajectory.

    Returns ((total, score, kl), th_sub, phi_sub, stride) with the three traces
    as numpy arrays. `seed` feeds one PRNG key per evaluated point (only used
    by stochastic scores).
    """
    stride = max(1, theta_traj.shape[0] // max_points)
    th_sub, phi_sub = theta_traj[::stride], phi_traj[::stride]
    keys = jax.random.split(jax.random.key(seed), th_sub.shape[0])
    total, score, kl = jax.vmap(objective_fn)(th_sub, phi_sub, keys)
    return ((np.asarray(total), np.asarray(score), np.asarray(kl)),
            th_sub, phi_sub, stride)


################################################################################
# Inference, Experiment
################################################################################


def run_experiment_moe(cfg, base_score_term, prior, data, log_likelihood,
                       dtype=None, theta_init=None):
    """Run the MoE SGLD flow for one config.

    Returns (thetas_final, phi_final), (theta_traj, phi_traj, eff, k), prior.
    theta_traj is (n_iter+1, E*M, d), phi_traj is (n_iter+1, E, 2), both with
    index 0 the init.
    """
    if dtype is None:  # mainly for x32 or x64
        dtype = jnp.asarray(data[0]).dtype

    # Init particles and experts
    key = jax.random.key(cfg.seed)
    key, sub_th, sub_phi = jax.random.split(key, 3)
    keys = jax.random.split(key, cfg.n_iter)

    if theta_init is None:  # TESTING
        thetas_init = prior.sample(sub_th, (cfg.E * cfg.M, cfg.d), dtype)
    else:
        thetas_init = jnp.asarray(theta_init, dtype).reshape(
            cfg.E * cfg.M, cfg.d)          # expert-major: rows 0..M-1 are e=0

    phi_init = cfg.phi_init_scale * jax.random.normal(
        sub_phi, (cfg.E, cfg.gate_dim + 1), dtype)

    if cfg.pin_gate:                      # reference expert: z_{E-1}(t) = 0
        phi_init = phi_init.at[-1].set(0.0)


    # Score -> update step -> run steps
    score_term = make_moe_score(base_score_term, cfg.M)
    step = make_step_moe(cfg, score_term, prior, data, log_likelihood, cfg.lam,
                         cfg.step_size, cfg.gate_step_size,
                         gate_burn_in=cfg.gate_burn_in)

    carry_final, ((theta_traj, phi_traj), eff, k) = jax.lax.scan(
        step, (thetas_init, phi_init, jnp.int32(0),
               jnp.zeros_like(phi_init), jnp.zeros_like(phi_init)), keys)
    state_final = carry_final[:2]

    theta_traj = jnp.concatenate([thetas_init[None], theta_traj])
    phi_traj = jnp.concatenate([phi_init[None], phi_traj])
    # globals().update(locals()) Pretty good debuggin line
    return state_final, (theta_traj, phi_traj, eff, k), prior


def run_experiments_moe(configs, base_score_term, prior, t_train, y_train,
                        log_likelihood, plotter, dtype=None, theta_init=None):
    if isinstance(configs, ExperimentConfigMoE):
        configs = [configs]

    results = {}
    for i, cfg in enumerate(configs):
        label = cfg.label or f"config_{i}"      # empty labels would collide

        (thetas_final, phi_final), (theta_traj, phi_traj, eff, k), _ = \
            run_experiment_moe(cfg, base_score_term, prior,
                               (t_train, y_train), log_likelihood,
                               dtype=dtype, theta_init=theta_init)

        objective_fn = make_objective_moe(
            cfg, base_score_term, prior, (t_train, y_train), log_likelihood)
        (obj, score, kl), th_sub, _, log_every = objective_trace_moe(
            theta_traj, phi_traj, objective_fn)

        results[label] = {
            "thetas_final": thetas_final, "phi_final": phi_final,
            "traj": theta_traj, "phi_traj": phi_traj,
            "prior": prior, "eff": eff, "k": k,
            "obj": obj, "obj_score": score, "obj_kl": kl,
        }

        plotter.log_every = log_every
        plotter.plot(
            th_sub,
            obj,
            thetas_final.reshape(cfg.E, cfg.M, cfg.d),   # expert-major
            phi_final,
            title=label,
            obj_text="score + KL (MoE)",
            show_expert_regions=True, show_theta_scatter=True,
        )

    plt.show()
    return results
