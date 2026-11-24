"""
Diagnostic panel for MoE-PVI runs.

Usage in a notebook
-------------------
    from pcuq_plot import Plotter

    P = Plotter(
        data=(_t, _y),
        sample_y=sample_y,
        gate_weights=gate_weights,
        true_function=true_function,
        log_every=LOG_EVERY,
    )

    P.plot(traj, obj, thetas, phi, title="single population (E=1)", save=True)

Every experiment setting is stamped under the figure. By default the settings
are read out of the *calling* namespace (your notebook's globals), so the
caption cannot drift from the constants cell that actually ran.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:  # jax is only needed for the predictive panel
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = jnp = None
    _HAS_JAX = False

# Output quality only -- none of these alter how a figure looks.
# fonttype 42 embeds real TrueType glyphs in PDF/PS (selectable, searchable and
# scalable text instead of the default Type-3 outlines); svg.fonttype "none"
# keeps SVG text as text.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

#: Formats a saved figure is written in when `save=True`. "pdf" is vector.
VECTOR_FORMATS = frozenset({"pdf", "svg", "eps", "ps"})


def hi_res_inline(dpi: int = 200, fmt: str = "retina") -> None:
    """Render inline figures at `dpi`, so the image you right-click -> "Save
    image as" out of the notebook is print quality instead of the default
    100 dpi screen grab.

    Call once near the top of the notebook, before any plotting.

    With fmt="retina" the notebook draws the image at half its pixel size, so
    dpi=200 is exactly the current on-screen size at twice the resolution.
    Higher dpi keeps growing the picture on screen (dpi=300 shows it 1.5x
    larger); that is a display effect only -- the figure itself is unchanged.
    A no-op outside IPython.
    """
    try:
        from IPython import get_ipython
        from IPython.core.pylabtools import select_figure_formats
        from matplotlib_inline.config import InlineBackend
    except ImportError:  # pragma: no cover - notebook convenience only
        return
    ip = get_ipython()
    if ip is None:
        return
    kw = {"bbox_inches": "tight", "dpi": int(dpi)}
    cfg = InlineBackend.instance()
    cfg.print_figure_kwargs = kw
    cfg.figure_formats = {fmt}
    # rebuild the formatters now, in case the config singleton has no shell
    select_figure_formats(ip, {fmt}, **kw)


# ---------------------------------------------------------------------------
# settings footer
# ---------------------------------------------------------------------------

#: Names looked up in the calling namespace, in the order they are printed.
#: Append to this list when you add a knob.
DEFAULT_SETTINGS_KEYS: list[str] = [
    # data / model
    "N_DATA", "SIGMA_NOISE", "SIGMA", "LS", "PRIOR_SD",
    # particle system
    "N_PARTICLES", "N_SAMPLES", "S_PARTNER", "N_BATCH",
    # flow
    "SCALE", "LAM_eff", "LAM", "H_EFF", "H",
    # run
    "N_STEPS", "LOG_EVERY", "DTYPE",
]

#: Non-scalar names (backends, step fns, kernels) described via `describe_obj`
#: instead of `_fmt`. Always printed, including under settings_keys="all".
DEFAULT_OBJECT_KEYS: list[str] = ["backend", "kernel", "step", "objective"]

_SCALARISH = (int, float, bool, str, complex, np.number, np.bool_)


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, type):                        # DTYPE = jnp.float32
        return getattr(v, "__name__", str(v))
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return getattr(v, "__name__", str(v))


def _caller_globals(skip: int = 1) -> dict:
    """Globals of the frame `skip` levels above this call."""
    frame = inspect.currentframe()
    try:
        for _ in range(skip + 1):
            if frame is None:
                return {}
            frame = frame.f_back
        return dict(frame.f_globals) if frame is not None else {}
    finally:
        del frame


def _auto_keys(ns: Mapping[str, Any]) -> list[str]:
    """Every scalar-valued, non-private name — used when settings_keys='all'."""
    return [k for k, v in ns.items()
            if not k.startswith("_") and (isinstance(v, _SCALARISH) or v is None)]


def _factory_name(fn: Any) -> str | None:
    """'gaussian_backend' from a closure whose qualname is
    'gaussian_backend.<locals>.sample'."""
    fn = inspect.unwrap(fn) if callable(fn) else fn
    fn = getattr(fn, "__wrapped__", fn)          # jax.jit wrappers
    qn = getattr(fn, "__qualname__", "")
    if "<locals>" in qn:
        return qn.split(".<locals>.")[0].split(".")[-1]
    return None


def describe_obj(o: Any, depth: int = 0) -> str:
    """Best-effort short label for a backend / step fn / kernel object."""
    if o is None:
        return "None"
    if isinstance(o, _SCALARISH) or isinstance(o, type):
        return _fmt(o)

    # 1. an explicit tag always wins
    for attr in ("name", "label", "_label"):
        v = getattr(o, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2. functools.partial -> func(kwargs)
    if isinstance(o, functools.partial):
        args = [describe_obj(a, depth + 1) for a in o.args]
        args += [f"{k}={describe_obj(v, depth + 1)}" for k,
                 v in o.keywords.items()]
        return f"{describe_obj(o.func, depth + 1)}({', '.join(args)})"

    # 3. NamedTuple / dataclass of functions: name it by its factory, and show
    #    any scalar fields (S, S_partner, ...) that came from the constructor
    fields = getattr(o, "_fields", None)
    if fields is None and hasattr(o, "__dataclass_fields__"):
        fields = tuple(o.__dataclass_fields__)
    if fields:
        scalars, factory = [], None
        for f in fields:
            v = getattr(o, f, None)
            if isinstance(v, _SCALARISH) or v is None:
                scalars.append(f"{f}={_fmt(v)}")
            elif factory is None and depth < 2:
                factory = _factory_name(v)
        head = factory or type(o).__name__
        return f"{head}({', '.join(scalars)})" if scalars else head

    # 4. plain closure: factory name + any scalar free variables
    if callable(o):
        head = _factory_name(o) or getattr(o, "__name__", type(o).__name__)
        free = []
        fn = getattr(o, "__wrapped__", o)
        cells = getattr(fn, "__closure__", None) or ()
        names = getattr(getattr(fn, "__code__", None), "co_freevars", ()) or ()
        for nm, cell in zip(names, cells):
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, _SCALARISH):
                free.append(f"{nm}={_fmt(v)}")
            elif depth < 2 and (fac := _factory_name(v)):
                free.append(f"{nm}={fac}")
        return f"{head}({', '.join(free)})" if free else head

    return type(o).__name__


def fn_expr(fn: Callable) -> str:
    """Best-effort one-line rendering of a function's body."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return getattr(fn, "__name__", "?")
    src = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', "",
                 src, flags=re.S)   # drop docstring
    body = [ln.strip() for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith(("def ", "@", "#"))]
    if not body:
        return getattr(fn, "__name__", "?")
    if len(body) == 1 and body[0].startswith("return "):
        return body[0][len("return "):]
    return "; ".join(body)


def _model_line(tag: str, expr: str, sigma: Any) -> str:
    noise = "" if sigma is None else f" + N(0, {_fmt(sigma)}^2)"
    return f"{tag + ':':<14}y = {expr}{noise}"


def true_model_text(true_function: Callable | None = None,
                    sigma_noise: float | None = None,
                    label: str | None = None) -> str:
    """'true model:  y = t**3 + N(0, 0.3^2)'  (data-generating process)"""
    if label is not None:
        expr = label
    elif true_function is not None:
        expr = fn_expr(true_function)
    else:
        return ""
    return _model_line("true model", expr, sigma_noise)


def model_text(mu_theta: Callable | None = None,
               sigma: float | None = None,
               label: str | None = None,
               sample_y: Callable | None = None,
               max_len: int = 150) -> str:
    """'fitted model:  y = theta[0] + theta[1] * (t ** 3) + N(0, 0.3^2)'"""
    if label is not None:
        expr = label
    elif mu_theta is not None:
        expr = fn_expr(mu_theta)
    elif sample_y is not None:
        expr = fn_expr(sample_y)
        sigma = None                      # the draw already shows its own noise
    else:
        return ""
    if len(expr) > max_len:
        expr = expr[:max_len - 1] + "…"
    return _model_line("fitted model", expr, sigma)


def settings_text(namespace: Mapping[str, Any] | None = None,
                  extra: Mapping[str, Any] | None = None,
                  keys: Sequence[str] | str | None = None,
                  object_keys: Sequence[str] | None = None,
                  width: int = 210,
                  header: str = "",
                  timestamp: bool = True) -> str:
    """Wrapped one-string summary of every setting, optionally led by `header`."""
    ns = {} if namespace is None else namespace
    if keys == "all":
        names = _auto_keys(ns)
    else:
        names = DEFAULT_SETTINGS_KEYS if keys is None else list(keys)
    obj_names = DEFAULT_OBJECT_KEYS if object_keys is None else list(
        object_keys)
    items = [f"{k}={describe_obj(ns[k])}" for k in obj_names if k in ns]
    items += [f"{k}={_fmt(ns[k])}" for k in names if k in ns]
    if extra:
        items += [f"{k}={describe_obj(v)}" for k, v in extra.items()]
    if timestamp:
        items.append(time.strftime("%Y-%m-%d %H:%M"))
    body = textwrap.fill("  |  ".join(items), width=width) if items else ""
    return "\n".join(s for s in (header, body) if s)


def _auto_name(title: str, ext: str = "png") -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", title).strip("_").lower() or "run"
    return f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}.{ext}"


# ---------------------------------------------------------------------------
# predictive helpers
# ---------------------------------------------------------------------------

def _draw_predictive(thetas, tg, sample_y, n_draws, seed):
    """(E, M, len(tg), n_draws) draws from every particle."""
    E, M = thetas.shape[0], thetas.shape[1]
    flat = np.asarray(thetas).reshape(E * M, -1)
    if _HAS_JAX:
        keys = jax.random.split(jax.random.key(seed), E * M)
        draws = jax.vmap(sample_y, (0, None, 0, None))(
            jnp.asarray(flat), tg, keys, n_draws)
        return np.asarray(draws).reshape(E, M, len(tg), n_draws)
    out = [np.asarray(sample_y(flat[i], tg, i + seed, n_draws))
           for i in range(E * M)]
    return np.stack(out).reshape(E, M, len(tg), n_draws)


def _weighted_quantiles(draws, w, quantiles):
    """draws (E, M, G, S), gate weights w (G, E) -> (len(quantiles), G)."""
    E, M, G, S = draws.shape
    stacked = np.concatenate(
        [draws[e].transpose(1, 0, 2).reshape(G, -1) for e in range(E)], axis=1)
    wts = np.concatenate(
        [np.repeat(w[:, e][:, None], M * S, axis=1) for e in range(E)], axis=1)
    qs = np.empty((G, len(quantiles)))
    for i in range(G):
        o = np.argsort(stacked[i])
        v, ww = stacked[i][o], wts[i][o]
        qs[i] = np.interp(quantiles, np.cumsum(ww) / ww.sum(), v)
    return qs.T


def _shades(base, n, dark=0.40, light=0.55):
    """`n` tints of a single base colour, dark -> light, so experts stay
    tellable apart while sharing a hue. n <= 1 returns the base untouched."""
    rgb = np.asarray(mcolors.to_rgb(base))
    if n <= 1:
        return [tuple(rgb)]
    out = []
    for t in np.linspace(-dark, light, n):
        c = rgb + (1.0 - rgb) * t if t >= 0 else rgb * (1.0 + t)
        out.append(tuple(np.clip(c, 0.0, 1.0)))
    return out


_EXPERT_CMAP = plt.get_cmap("tab10")


def _plot_expert_regions(ax, phi, gate_weights, t_range, n_grid, E, dtype=None):
    """Gate weights w_e(t) over the covariate range plotted as individual lines."""
    if _HAS_JAX:
        tg = jnp.linspace(*t_range, n_grid, dtype=dtype)
    else:
        tg = np.linspace(*t_range, n_grid)
    tg_np = np.asarray(tg)
    w = np.asarray(gate_weights(phi, tg))  # (G, E)

    for e in range(E):
        ax.plot(
            tg_np,
            w[:, e],
            color=_EXPERT_CMAP(e % 10),
            label=f"expert {e}",
            lw=1.5,
        )

    ax.set_xlim(tg_np.min(), tg_np.max())
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=min(E, 4), fontsize=6, framealpha=0.9)
    ax.set(xlabel="t", ylabel=r"$w_e(t)$", title="expert regions")


def _plot_theta_scatter(ax, thetas, E, M, ix, param_labels):
    """theta-space scatter of final particles, coloured by expert."""
    th = np.asarray(thetas)                                     # (E, M, p)
    # guard len(ix)==1
    i0, i1 = (ix + ix)[:2]
    for e in range(E):
        ax.scatter(th[e, :, i0], th[e, :, i1],
                   color=_EXPERT_CMAP(e % 10), s=22, alpha=0.85,
                   edgecolor="k", linewidth=0.3, label=f"expert {e}")
    lab0 = param_labels[0] if len(
        param_labels) > 0 else rf"$\theta_{{{i0+1}}}$"
    lab1 = param_labels[1] if len(
        param_labels) > 1 else rf"$\theta_{{{i1+1}}}$"
    ax.legend(fontsize=6, framealpha=0.9)
    ax.set(xlabel=lab0, ylabel=lab1, title="particles by expert")
# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def plot_run(traj, obj, thetas, phi=None, *,
             data=None,
             sample_y=None,
             gate_weights=None,
             show_expert_regions=False,
             show_theta_scatter=False,
             true_function=None,
             true_label=None,
             mu_theta=None,
             model_label=None,
             sigma=None,
             sigma_noise=None,
             log_every=1,
             param_ix=(0, 1),
             param_labels=(r"$\theta_1$", r"$\theta_2$"),
             param_titles=("intercept", "slope"),
             t_range=(-2.0, 2.0),
             n_grid=200,
             n_pred_draws=64,
             pred_seed=1,
             quantiles=(0.05, 0.5, 0.95),
             band_label="90%",
             settings="auto",
             settings_keys=None,
             object_keys=None,
             extra_settings=None,
             obj_text=r"$\frac{1}{n^2}\sum_i$MMD$^2_i$",
             footer_width=210,
             title="",
             save=None,
             outdir="figs",
             dpi=300,
             fmt="png",
             rasterize_traj=True,
             show=True):
    """
    Parameters
    ----------
    traj      : (K, E, M, p) or (K, E*M, p) particle trajectory
    obj       : (K,) objective values
    thetas    : (E, M, p) final particles
    phi       : (E, d) gate parameters; ignored when `gate_weights` is None
    data      : (t, y) observed data for the predictive panel, or None
    sample_y  : fn(theta, tg, key, n) -> (len(tg), n). Omit to skip the panel.
    gate_weights : fn(phi, tg) -> (len(tg), E). Omit for equal expert weights.
    settings  : "auto" reads the caller's globals; pass a dict for explicit
                control, or None for no footer.
    settings_keys : list of names to print, or "all" for every scalar in scope.
    save      : True (auto filename in `outdir`), a path string, or None.
    fmt       : extension(s) written when `save is True` -- "png", "pdf", or a
                sequence such as ("png", "pdf") for one file each. An explicit
                `save="...ext"` path wins.
    dpi       : raster resolution (300 = print quality).
    rasterize_traj : draw the dense particle traces as pixels inside vector
                output, so a PDF stays a few MB instead of hundreds. No effect
                on PNG, and none on the axes, text or legends, which stay
                vector.

    Returns (fig, path).
    """
    if settings == "auto":
        settings = _caller_globals(1)

    traj = np.asarray(traj)
    obj = np.asarray(obj)
    thetas_np = np.asarray(thetas)
    E, M = thetas_np.shape[0], thetas_np.shape[1]

    its = np.arange(traj.shape[0]) * log_every
    # (K, E*M, p)
    tr = traj.reshape(traj.shape[0], -1, traj.shape[-1])
    p = tr.shape[-1]
    # (K, E, M, p) whenever the particle -> expert assignment is recoverable
    if traj.ndim == 4:
        tr_e = traj
    elif tr.shape[1] == E * M:
        tr_e = tr.reshape(tr.shape[0], E, M, p)
    else:                                   # unknown grouping: one block
        tr_e = tr[:, None]
    ix = [i for i in param_ix if i < p]

    if isinstance(settings, Mapping):
        if sigma_noise is None:
            sigma_noise = settings.get("SIGMA_NOISE")
        if sigma is None:
            sigma = settings.get("SIGMA")
        if mu_theta is None:
            mu_theta = settings.get("mu_theta")

    # --- footer first: its height determines the figure height ---------------
    info = ""
    if settings is not None:
        header = "\n".join(s for s in (
            true_model_text(true_function, sigma_noise, true_label),
            model_text(mu_theta, sigma, model_label, sample_y),
        ) if s)
        info = settings_text(
            settings,
            extra={"E": E, "M": M, "steps_logged": traj.shape[0],
                   **(extra_settings or {})},
            keys=settings_keys,
            object_keys=object_keys,
            width=footer_width,
            header=header,
        )
    n_lines = info.count("\n") + 1 if info else 0
    foot_in = 0.22 * n_lines + 0.15 if info else 0.0
    fig_h = 3.6 + foot_in

    show_pred = sample_y is not None
    show_regions = show_expert_regions and (
        gate_weights is not None) and (phi is not None)
    show_scatter = show_theta_scatter and E > 1
    ncols = len(ix) + 1 + int(show_pred) + \
        int(show_regions) + int(show_scatter)
    fig, ax = plt.subplots(1, ncols, figsize=(
        4.5 * ncols, fig_h), squeeze=False)
    ax = ax[0]

    # --- particle trajectories ----------------------------------------------
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    n_grp = tr_e.shape[1]
    for k, i in enumerate(ix):
        shades = _shades(colors[k % len(colors)], n_grp)
        for e in range(n_grp):
            ax[k].plot(its, tr_e[:, e, :, i], lw=0.7,
                       color=shades[e], alpha=0.6)
        if n_grp > 1:
            handles = [plt.Line2D([], [], color=shades[e], lw=1.5)
                       for e in range(n_grp)]
            ax[k].legend(handles, [f"expert {e}" for e in range(n_grp)],
                         fontsize=6, framealpha=0.9,
                         ncol=min(n_grp, 4), loc="upper right")
        lab = param_labels[k] if k < len(
            param_labels) else rf"$\theta_{{{i+1}}}$"
        ttl = param_titles[k] if k < len(param_titles) else f"param {i}"
        ax[k].set(xlabel="iteration", ylabel=lab, title=ttl)

    # --- objective -----------------------------------------------------------
    j = len(ix)
    ax[j].plot(its[:len(obj)], obj, lw=1, color="k")
    ax[j].set(xlabel="iteration", ylabel=obj_text,
              title="objective (-lambda H(Q) absent)", yscale="log")

    # --- posterior predictive of the mixture ---------------------------------
    if show_pred:
        a = ax[j + 1]
        dtype = settings.get("DTYPE") if isinstance(
            settings, Mapping) else None
        if _HAS_JAX:
            tg = jnp.linspace(*t_range, n_grid, dtype=dtype)
        else:
            tg = np.linspace(*t_range, n_grid)
        tg_np = np.asarray(tg)

        if gate_weights is not None and phi is not None:
            w = np.asarray(gate_weights(phi, tg))                  # (G, E)
        else:
            w = np.full((n_grid, E), 1.0 / E)

        draws = _draw_predictive(thetas, tg, sample_y, n_pred_draws, pred_seed)
        q = _weighted_quantiles(draws, w, list(quantiles))

        if data is not None:
            a.scatter(np.asarray(data[0]), np.asarray(data[1]),
                      s=4, alpha=0.2, color="tab:red", label="data")
        if len(q) >= 3:
            a.fill_between(tg_np, q[0], q[-1], alpha=0.15, color="grey",
                           lw=0, label=band_label)
        if len(q) % 2 == 1:
            a.plot(tg_np, q[len(q) // 2], color="k", label="median")
        if true_function is not None:
            a.plot(tg_np, np.asarray(true_function(tg_np)),
                   "k--", lw=1, label="true")
        a.legend(loc="upper left", fontsize=8, frameon=False)
        a.set(xlabel="t", ylabel="y", title="predictive")

        # --- expert regions -------------------------------------------------------
        col = j + 1 + int(show_pred)
        if show_regions:
            dtype = settings.get("DTYPE") if isinstance(
                settings, Mapping) else None
            _plot_expert_regions(
                ax[col], phi, gate_weights, t_range, n_grid, E, dtype)
            col += 1

        # --- particles by expert ---------------------------------------------------
        if show_scatter:
            _plot_theta_scatter(ax[col], thetas, E, M, ix, param_labels)

    # --- footer --------------------------------------------------------------
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, foot_in / fig_h, 1, 0.94 if title else 1.0])
    if info:
        fig.text(0.006, 0.006, info, ha="left", va="bottom",
                 fontsize=7, family="monospace", color="0.25")

    path = None
    if save:
        exts = [fmt] if isinstance(fmt, str) else list(fmt)
        if isinstance(save, str):
            paths = [save]
        else:
            stem = _auto_name(title, "")
            paths = [os.path.join(outdir, stem + e) for e in exts]
        if rasterize_traj and any(
                os.path.splitext(q)[1].lstrip(".").lower() in VECTOR_FORMATS
                for q in paths):
            for k in range(len(ix)):
                for line in ax[k].lines:
                    line.set_rasterized(True)
                ax[k].set_rasterization_zorder(None)
        for q in paths:
            os.makedirs(os.path.dirname(q) or ".", exist_ok=True)
            fig.savefig(q, dpi=dpi, bbox_inches="tight")
            print("saved ->", q)
        path = paths[0]

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, path


def plot_eff(cfg, eff, sub=20, step_size=None):
    if step_size == None:
        step_size = cfg.step_size

    it = np.arange(eff[::sub].shape[0])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(it, eff[::sub], lw=0.6)
    ax.axhline(step_size, ls="--", c="k", lw=0.8, label="step_size")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("eff step")
    ax.legend()
    plt.tight_layout()

    print(
        f"clipped on {(eff < 0.99 * step_size).mean():.1%} of iterations")
    print(f"median eff: {np.median(eff):.2e}  (nominal {step_size:.2e})")
    print(f"total integrated time: {eff.sum():.3f}")
    plt.show()

# ---------------------------------------------------------------------------
# bind the fixed pieces once, then call .plot(...) per experiment
# ---------------------------------------------------------------------------


@dataclass
class Plotter:
    data: tuple | None = None
    sample_y: Callable | None = None
    gate_weights: Callable | None = None
    true_function: Callable | None = None
    true_label: str | None = None
    mu_theta: Callable | None = None
    model_label: str | None = None
    log_every: int = 1
    outdir: str = "figs"
    defaults: dict = field(default_factory=dict)

    def plot(self, traj, obj, thetas, phi=None, **kw):
        opts = dict(
            data=self.data,
            sample_y=self.sample_y,
            gate_weights=self.gate_weights,
            true_function=self.true_function,
            true_label=self.true_label,
            mu_theta=self.mu_theta,
            model_label=self.model_label,
            log_every=self.log_every,
            outdir=self.outdir,
            settings=_caller_globals(1),      # the notebook, not this method
        )
        opts.update(self.defaults)
        opts.update(kw)
        return plot_run(traj, obj, thetas, phi, **opts)

    __call__ = plot
