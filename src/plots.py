"""Shared matplotlib style and the standard chapter figures.

Palette follows the dataviz reference instance (light surface). Color follows the
entity (run_type), assigned in fixed order, never by rank.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fits import PowerLawFit

SURFACE = "#fcfcfb"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"

# Fixed slot per run_type. Final runs are the story, so they get slot 1 and a star.
RUN_COLORS = {
    "final": "#2a78d6",     # slot 1 blue
    "ladder": "#eb6834",    # slot 2 orange
    "ablation": "#1baf7a",  # slot 3 aqua
    "midtrain": "#eda100",  # slot 4 yellow
    "aborted": MUTED,
    "unknown": MUTED,
}
RUN_MARKERS = {"final": "*", "ladder": "o", "ablation": "o", "midtrain": "s", "aborted": "x", "unknown": "o"}
RUN_LABELS = {"final": "Final run", "ladder": "Ladder run", "ablation": "Ablation",
              "midtrain": "Mid-training", "aborted": "Aborted", "unknown": "Unknown"}
# Fixed slots for projects when several releases are overlaid.
PROJECT_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _empty(ax, msg="No data yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def scaling_curve(df: pd.DataFrame, y: str = "loss", fit: PowerLawFit | None = None,
                  ax=None, title: str | None = None, ylabel: str | None = None, label_finals: bool = True):
    """Log compute vs. loss/benchmark. Experiments as points, finals as stars, fit + extrapolation."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = df.dropna(subset=["flops", y])
    if d.empty:
        _empty(ax); return ax
    for rt in ["ladder", "ablation", "midtrain", "aborted", "unknown", "final"]:
        s = d[d["run_type"] == rt]
        if s.empty:
            continue
        final = rt == "final"
        ax.scatter(s["flops"], s[y], s=220 if final else 55, marker=RUN_MARKERS[rt],
                   c=RUN_COLORS[rt], edgecolors=SURFACE, linewidths=2 if final else 1.5,
                   label=RUN_LABELS[rt], zorder=4 if final else 3)
        if final and label_finals:
            for _, r in s.iterrows():
                ax.annotate(r["run_id"], (r["flops"], r[y]), xytext=(8, 6), textcoords="offset points",
                            fontsize=9, color=INK2)
    if fit is not None:
        ex = d[~d["is_final"]]
        lo, hi = ex["flops"].min(), ex["flops"].max()
        top = max(d["flops"].max(), hi) * 3
        xs_in = np.geomspace(lo, hi, 100); xs_out = np.geomspace(hi, top, 100)
        ax.plot(xs_in, fit.predict(xs_in), color=INK2, lw=2, label=f"Fit on experiments: {fit.label()}", zorder=2)
        ax.plot(xs_out, fit.predict(xs_out), color=INK2, lw=2, ls=(0, (4, 3)), label="Extrapolation", zorder=2)
        band = np.exp(fit.rmse_log)
        ax.fill_between(xs_out, fit.predict(xs_out) / band, fit.predict(xs_out) * band, color=INK2, alpha=0.10, lw=0)
    ax.set_xscale("log")
    if y == "loss":
        ax.set_yscale("log")
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel(ylabel or ("Loss" if y == "loss" else y))
    ax.set_title(title or "Scaling curve: experiments and final runs")
    ax.legend(loc="upper right")
    return ax


def compute_split(df: pd.DataFrame, ax=None, title: str | None = None, value: str = "flops"):
    """One horizontal stacked bar of compute share by run_type, with a 2px surface gap."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.8))
    g = df.dropna(subset=[value]).groupby("run_type", observed=True)[value].sum()
    g = g[g > 0]
    if g.empty:
        _empty(ax); return ax
    order = [rt for rt in ["final", "midtrain", "ladder", "ablation", "aborted", "unknown"] if rt in g.index]
    total = g.sum(); left = 0.0
    for rt in order:
        w = g[rt] / total
        ax.barh(0, w, left=left, color=RUN_COLORS[rt], height=0.5, edgecolor=SURFACE, linewidth=2, label=RUN_LABELS[rt])
        if w > 0.07:
            ax.text(left + w / 2, 0, f"{w:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if rt in ("final", "aborted", "unknown") else INK)
        left += w
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False); ax.set_xticks([])
    unit = "FLOPs" if value == "flops" else value.replace("_", " ")
    n_exp = int((~df["is_final"]).sum()); n_fin = int(df["is_final"].sum())
    ax.set_title(title or f"Share of logged {unit}: {n_exp} experimental runs vs {n_fin} final")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1)
    return ax


def run_size_hist(df: pd.DataFrame, ax=None):
    """Distribution of experimental run sizes (log FLOPs), finals marked as rug ticks."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    ex = df[~df["is_final"]].dropna(subset=["flops"]); fin = df[df["is_final"]].dropna(subset=["flops"])
    if ex.empty and fin.empty:
        _empty(ax); return ax
    if not ex.empty:
        bins = np.arange(np.floor(np.log10(ex["flops"].min())), np.ceil(np.log10(df["flops"].max())) + 1, 0.5)
        ax.hist(np.log10(ex["flops"]), bins=bins, color=RUN_COLORS["ladder"], edgecolor=SURFACE, linewidth=2,
                label="Experimental runs")
    for _, r in fin.iterrows():
        ax.axvline(np.log10(r["flops"]), color=RUN_COLORS["final"], lw=2)
    if not fin.empty:
        ax.plot([], [], color=RUN_COLORS["final"], lw=2, label="Final run(s)")
    ax.set_xlabel("log10 training compute (FLOPs)"); ax.set_ylabel("Runs")
    ax.set_title("Size distribution of experimental runs"); ax.legend()
    return ax


def timeline(df: pd.DataFrame, ax=None, title: str | None = None):
    """Runs over calendar time, sized by compute."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    d = df.dropna(subset=["date_start", "flops"])
    if d.empty:
        _empty(ax, "No dated runs yet"); return ax
    size = 40 + 260 * (np.log10(d["flops"]) - np.log10(d["flops"].min())) / max(
        np.log10(d["flops"].max()) - np.log10(d["flops"].min()), 1e-9)
    for rt in ["ladder", "ablation", "midtrain", "aborted", "unknown", "final"]:
        s = d[d["run_type"] == rt]
        if s.empty:
            continue
        ax.scatter(s["date_start"], np.log10(s["flops"]), s=size[s.index], marker=RUN_MARKERS[rt],
                   c=RUN_COLORS[rt], edgecolors=SURFACE, linewidths=1.5, label=RUN_LABELS[rt], zorder=3)
    ax.set_ylabel("log10 FLOPs"); ax.set_title(title or "Runs over time (marker size ∝ log compute)")
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend(loc="upper left")
    return ax


def overlay_projects(dfs: dict[str, pd.DataFrame], fits: dict[str, PowerLawFit | None], y: str = "loss", ax=None,
                     title: str | None = None):
    """Several releases of one lab (or several labs) on one axis. Color follows the project, fixed order."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for i, (name, df) in enumerate(dfs.items()):
        c = PROJECT_SLOTS[i % len(PROJECT_SLOTS)]
        d = df.dropna(subset=["flops", y])
        if d.empty:
            continue
        any_data = True
        ex, fin = d[~d["is_final"]], d[d["is_final"]]
        ax.scatter(ex["flops"], ex[y], s=45, c=c, edgecolors=SURFACE, linewidths=1.5, label=name, zorder=3)
        ax.scatter(fin["flops"], fin[y], s=220, marker="*", c=c, edgecolors=SURFACE, linewidths=2, zorder=4)
        f = fits.get(name)
        if f is not None:
            xs = np.geomspace(d["flops"].min(), d["flops"].max(), 100)
            ax.plot(xs, f.predict(xs), color=c, lw=2, zorder=2)
    if not any_data:
        _empty(ax); return ax
    ax.set_xscale("log")
    if y == "loss":
        ax.set_yscale("log")
    ax.set_xlabel("Training compute (FLOPs)"); ax.set_ylabel("Loss" if y == "loss" else y)
    ax.set_title(title or "Across releases (stars = final runs)"); ax.legend()
    return ax


def ladder_map(df: pd.DataFrame, ax=None, title: str | None = None):
    """Parameters vs tokens (log-log) for every run: where the ladder sits relative to the final runs.
    Useful when a lab publishes run sizes but not losses. Iso-FLOP guide lines at 1e20 ... 1e25."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    d = df.dropna(subset=["params_active", "tokens"])
    if d.empty:
        _empty(ax); return ax
    for rt in ["ladder", "ablation", "midtrain", "aborted", "unknown", "final"]:
        s = d[d["run_type"] == rt]
        if s.empty:
            continue
        final = rt == "final"
        ax.scatter(s["params_active"], s["tokens"], s=220 if final else 55, marker=RUN_MARKERS[rt],
                   c=RUN_COLORS[rt], edgecolors=SURFACE, linewidths=2 if final else 1.5, label=RUN_LABELS[rt], zorder=4 if final else 3)
        if final:
            for _, r in s.iterrows():
                ax.annotate(r["run_id"], (r["params_active"], r["tokens"]), xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK2)
    nmin, nmax = d["params_active"].min() / 2, d["params_active"].max() * 2
    ns = np.geomspace(nmin, nmax, 50)
    for C in [1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25]:
        ds = C / (6 * ns)
        if ds.max() < d["tokens"].min() / 3 or ds.min() > d["tokens"].max() * 3:
            continue
        ax.plot(ns, ds, color=GRID, lw=1, zorder=1)
        ax.annotate(f"{C:.0e} FLOPs", (ns[-1], ds[-1]), fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Active parameters"); ax.set_ylabel("Training tokens")
    ax.set_title(title or "Run sizes: parameters vs tokens (grey lines: equal compute)")
    ax.legend(loc="upper left")
    return ax


def stated_split(parts: dict[str, float], ax=None, title: str | None = None, unit: str = "GPU-hours"):
    """The lab's own stated split (e.g. development vs final GPU-hours) as one stacked bar."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.8))
    total = sum(parts.values()); left = 0.0
    colors = [RUN_COLORS["final"], RUN_COLORS["ladder"], RUN_COLORS["ablation"], RUN_COLORS["midtrain"]]
    for (k, v), c in zip(parts.items(), colors):
        w = v / total
        ax.barh(0, w, left=left, color=c, height=0.5, edgecolor=SURFACE, linewidth=2, label=f"{k}: {v:,.0f} {unit}")
        if w > 0.07:
            ax.text(left + w / 2, 0, f"{w:.0%}", ha="center", va="center", fontsize=9, color="white" if c == RUN_COLORS["final"] else INK)
        left += w
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.set_title(title or f"Lab-stated split of {unit}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return ax
