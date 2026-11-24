# MoE-PrO: prediction-centric UQ with a mixture of experts

JAX code for MoE-PrO. Each expert carries its own particle cloud, and a softmax gate
decides which expert explains each input $x$. Particles move by Langevin steps. The gate
is trained by SGD or Adam on the same objective.

```
src/
  pcuq_moe_utils.py    the method: gate, log score, kernel (log r_e) terms, update step, run loop
  pcuq_plot.py         diagnostic figure: particle traces, objective, predictive band, gate, particles
experiments/
  three_segment_cubic.py   thesis Section 4.1: linear experts on y = x^3 (E = 1, 2, 3, 5)
  three_segment_step.py    linear experts on a three-segment step function (~15 s on a CPU)
  sine_quadratic.py        quadratic experts on one period of a sine wave, same softmax-linear gate
figs/                      figures from the experiments are written here
requirements.txt
```

## Install

```
pip install -r requirements.txt
```

## Run the experiments

```
python experiments/three_segment_cubic.py
python experiments/three_segment_step.py
python experiments/sine_quadratic.py
```

Each experiment prints the expert weights $\pi_e$ and writes a figure to `figs/`.

- **`three_segment_cubic.py`:** the three-segment regression in Section 4.1 of the thesis.
  The data follow Kock et al. (2026, Appendix B.1): $x \sim U[-2, 2]$ and
  $y = x^3 + N(0, 0.1)$, with $n = 1000$. Each expert is the misspecified line
  $N(	heta_1 + 	heta_2 x, 0.1)$ with prior $N(0, 5I)$, $\lambda = 1$, $M = 32$ and
  $h = 1$. The run uses step $5 	imes 10^{-5}$ for 200,000 iterations, with the gate
  frozen for the first 5000 and then updated by Adam at $10^{-3}$. It fits
  $E = 1, 2, 3, 5$ in turn (`--E 3` for one), and also prints the held-out log predictive
  density and the 90% coverage on 10,000 test points. With $E = 3$ the gate switches near
  $x = \pm 1.1$, and the outer experts' particles lie along the tangent lines of $x^3$.
  The full run takes several minutes per $E$ on a CPU; `--n_iter 20000` is a quick check.
- **`three_segment_step.py`:** $y$ is $0$ on $[-6, 0)$, $5$ on $[0, 3)$, and $\pm 5$ with equal
  probability on $[3, 6]$, plus $N(0, 0.4^2)$ noise. With $E = 3$ linear experts, the gate
  switches at $x \approx 0$ and $x \approx 3$.
- **`sine_quadratic.py`:** $y = \sin(\pi x) + N(0, 0.2^2)$ on $[-1, 1]$, with experts
  $y \sim N(\theta_0 + \theta_1 x + \theta_2 x^2, 0.2^2)$. Out of $E = 4$, two experts
  survive, near $4x + 4x^2$ for $x < 0$ and $4x - 4x^2$ for $x > 0$, and the gate switches
  at $x \approx 0$. The other two get $\pi_e \approx 0$.

  The example keeps $x$ on $[-1, 1]$. On a wider range the $x^2$ term makes the Langevin
  step unstable at any step size large enough to move the particles. Fitting several
  periods also needs $|\theta_2|$ far outside the prior, so it is better to rescale $x$
  than to widen the prior.

## Using it on your own problem

Copy one of the experiments and edit it. Each experiment puts `src/` on `sys.path`, so it runs from
any directory. You supply four things: data, a prior, a per-datum log-likelihood and a config.

```python
from pcuq_moe_utils import (Prior, ExperimentConfigMoE, gate_weights,
                            gaussian_log_likelihood, log_score_term_moe,
                            run_experiments_moe)
from pcuq_plot import Plotter

# 1. Prior on one expert's parameters theta (dimension d)
prior = Prior(sample=lambda key, shape, dtype: ...,   # draws of shape `shape`
              log_pdf=lambda theta: ...)              # scalar log density

# 2. log p_theta(y_i | x_i) for one datum xy = (x_i, y_i)
def log_lik(theta, xy):
    return gaussian_log_likelihood(theta, xy, sigma=0.4)  # linear-Gaussian expert

# 3. Settings
cfg = ExperimentConfigMoE(n_iter=30_000, E=3, M=16, d=2,
                          step_size=1e-4, gate_step_size=1e-3,
                          gate_optimizer="adam", gate_burn_in=2_000,
                          lam=n * M)                  # thesis lambda = 1

# 4. Run, then plot (pass sample_y to Plotter for the predictive panel)
results = run_experiments_moe(cfg, log_score_term_moe, prior, x, y, log_lik,
                              Plotter(data=(x, y), gate_weights=gate_weights))
```

`results[cfg.label]` holds `thetas_final` with shape $(E \cdot M, d)$ in expert-major
order, `phi_final` (the gate), the full trajectories `traj` / `phi_traj`, and the
objective trace `obj`. You can pass a list of configs to compare settings; each one gets
its own entry and figure.

To fit without plotting, call `run_experiment_moe(cfg, log_score_term_moe, prior, (x, y), log_lik)`.

### Config fields worth knowing

- `E`, `M`: the number of experts and the number of particles per expert.
- `lam`: the weight on the score. The code's `lam` equals $\lambda n M$, where $\lambda$
  is the weight used in the thesis, so `lam = n * M` corresponds to $\lambda = 1$.
- `step_size`: the Langevin step for the particles. If the traces in the figure are flat,
  the step is too small, or the prior is too wide for where the particles start.
- `gate_step_size`, `gate_optimizer`: the gate's learning rate and optimiser
  (`"sgd"` or `"adam"`).
- `gate_burn_in`: how many steps the gate stays frozen at the start while the particles
  settle.
- `h`: the kernel bandwidth in the $\log r_e$ term.
- `gate_dim`: the number of gate covariates $K$. When `x` is 2-D, its first $K$ columns
  feed the gate.

### Swapping the model or the score

- **Model:** any `log_lik(theta, (x_i, y_i))` written in JAX works, and `d` sets the
  dimension of `theta`.
- **Score:** `log_score_term_moe` is the log score. Any function with the same signature
  `(particles, data, log_likelihood, lam, log_omega, key=None, M=None) -> scalar` can
  replace it.

Setting the environment variable `TOY_GATE_SCHED=warm` switches on a warm-up/decay
schedule for the gate's learning rate. With it unset, the rate is constant.
