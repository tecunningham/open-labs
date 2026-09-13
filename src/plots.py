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
        if final and len(s) <= 6:
            for _, r in s.iterrows():
                ax.annotate(r["run_id"], (r["params_active"], r["tokens"]), xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK2)
    xlo, xhi = d["params_active"].min() / 3, d["params_active"].max() * 3
    ylo, yhi = d["tokens"].min() / 3, d["tokens"].max() * 3
    ns = np.geomspace(xlo, xhi, 50)
    for C in [1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26]:
        ds = C / (6 * ns)
        inside = (ds > ylo) & (ds < yhi)
        if inside.sum() < 2:
            continue
        ax.plot(ns[inside], ds[inside], color=GRID, lw=1, zorder=1)
        ax.annotate(f"{C:.0e}", (ns[inside][-1], ds[inside][-1]), fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Active parameters"); ax.set_ylabel("Training tokens")
    ax.set_title(title or "Run sizes: parameters vs tokens (grey lines: equal FLOPs)")
    ax.legend(loc="upper left")
    return ax


def loss_trajectory(segs, ax=None, title: str | None = None, ylabel: str = "Loss",
                    min_tokens: float = 0.0, label_ends: bool = True):
    """A long run drawn as one trajectory: eval loss against cumulative training tokens, one line
    per phase or side branch, coloured by run_type. `segs` are src.losses.Segment objects. Points
    before `min_tokens` are dropped so the warm-up spike does not set the y-range."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    seen = set(); drawn = False; xmax = 0.0
    for s in segs:
        keep = s.tokens >= min_tokens
        if keep.sum() == 0:
            continue
        drawn = True
        x, y = s.tokens[keep], s.loss[keep]
        trunk = s.run_type in ("final", "midtrain")
        ax.plot(x, y, color=RUN_COLORS[s.run_type], lw=2.2 if trunk else 1.6, alpha=1 if trunk else 0.9,
                label=RUN_LABELS[s.run_type] if s.run_type not in seen else None, zorder=3 if trunk else 2,
                solid_capstyle="round")
        seen.add(s.run_type)
        if s.released:
            ax.scatter([x[-1]], [y[-1]], s=240, marker="*", c=RUN_COLORS["final"], edgecolors=SURFACE,
                       linewidths=2, zorder=5, label="Released checkpoint")
        if label_ends and s.label:
            box = dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85)
            if trunk:   # phase name above the middle of the phase, on the median loss so spikes do not move it
                mid = np.searchsorted(x, (x[0] + x[-1]) / 2)
                ax.annotate(s.label, (x[min(mid, len(x) - 1)], float(np.median(y))), xytext=(0, 9),
                            textcoords="offset points", fontsize=8, color=INK2, ha="center", va="bottom", bbox=box)
            else:       # side branch: name just past its last point, above the line
                ax.annotate(s.label, (x[-1], y[-1]), xytext=(5, 6), textcoords="offset points", fontsize=8,
                            color=MUTED, ha="left", va="bottom", bbox=box)
        xmax = max(xmax, float(x[-1]))
    if not drawn:
        _empty(ax); return ax
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v / 1e12:g}T"))
    ax.set_xlabel("Cumulative training tokens"); ax.set_ylabel(ylabel)
    ax.set_title(title or "Loss over the whole run (phases and side branches)")
    ax.set_xlim(0, xmax * 1.12)
    ax.legend(loc="upper right")
    return ax


def loss_curves(curves, x: str = "flops", ax=None, title: str | None = None, ylabel: str = "Loss",
                min_tokens: float = 0.0, label_ends: bool = True):
    """Several whole pretraining runs on one axis: eval loss against cumulative tokens (`x="tokens"`)
    or cumulative compute 6ND (`x="flops"`), log x. `curves` are src.losses.Curve objects. Final
    runs solid in project colours, abandoned trials dashed, ladder runs thin and grey."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    slot = 0
    ends: list[tuple[float, float]] = []      # (log10 x, adjusted y) of placed end-labels, for repelling
    y_span = max(np.nanmax(np.concatenate([c.loss[c.tokens >= min_tokens] for c in curves if (c.tokens >= min_tokens).sum() >= 2])) -
                 np.nanmin(np.concatenate([c.loss[c.tokens >= min_tokens] for c in curves if (c.tokens >= min_tokens).sum() >= 2])), 1e-6)
    step_y = 0.026 * y_span                  # about one 8pt label height on a 5in axis
    for c in curves:
        keep = c.tokens >= min_tokens
        if keep.sum() < 2:
            continue
        xv = (c.flops if x == "flops" else c.tokens)[keep]; yv = c.loss[keep]
        if c.kind == "ladder":
            color, lw, ls, z = MUTED, 1.2, "-", 2
        elif c.kind == "aborted":
            color, lw, ls, z = PROJECT_SLOTS[(slot := slot + 1) % len(PROJECT_SLOTS)], 1.6, (0, (4, 2)), 3
        elif c.kind == "delphi":
            color, lw, ls, z = INK2, 1.4, "-", 3
        elif c.kind == "moe":
            color, lw, ls, z = PROJECT_SLOTS[(slot := slot + 1) % len(PROJECT_SLOTS)], 2.2, (0, (1, 1.5)), 4
        else:
            color, lw, ls, z = PROJECT_SLOTS[(slot := slot + 1) % len(PROJECT_SLOTS)], 2.2, "-", 4
        ax.plot(xv, yv, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
        if label_ends:
            lx, y0 = np.log10(xv[-1]), float(yv[-1])
            free = lambda yy: not any(abs(lx - px) < 0.9 and abs(yy - py) < step_y for px, py in ends)
            ly = next(y0 + k * step_y for k in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5] if free(y0 + k * step_y))
            ends.append((lx, ly))
            ax.annotate(c.name, (xv[-1], yv[-1]), xytext=(5, (ly - yv[-1]) / step_y * 8), textcoords="offset points",
                        fontsize=8, color=color if c.kind != "ladder" else INK2, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative training compute (FLOPs, 6ND)" if x == "flops" else "Cumulative training tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ("Pretraining runs on one compute axis" if x == "flops" else "Pretraining runs by tokens"))
    ax.margins(x=0.18)
    from matplotlib.lines import Line2D
    kinds = {c.kind for c in curves}
    handles = [Line2D([], [], color=INK2, lw=2.2, label="Released dense run (phases stitched)"),
               Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 2)), label="Abandoned trial"),
               Line2D([], [], color=MUTED, lw=1.2, label="Ladder run (210B tokens each)")]
    if "delphi" in kinds:
        handles.append(Line2D([], [], color=INK2, lw=1.4, label="Delphi held-out target (compute-optimal)"))
    if "moe" in kinds:
        handles.append(Line2D([], [], color=INK2, lw=2.2, ls=(0, (1, 1.5)), label="MoE run (compute on active params)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    return ax


def capability_timeline(points: pd.DataFrame, ax=None, title: str | None = None, ylabel: str = "Score",
                        ref_lines: dict[str, float] | None = None, max_radius_pt: float = 22.0,
                        min_radius_pt: float = 3.0, fmax: float | None = None, notes: str | None = None):
    """Released or evaluated checkpoints over time: x = date, y = a benchmark score, circle area
    proportional to cumulative pretraining compute (`flops`), with a ring for post-training compute
    (`post_flops`) where known. Columns: date, label, value, flops, post_flops (optional),
    kind ('final', 'midtrain', 'aborted', ...). Reference lines mark other labs' models."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    d = points.dropna(subset=["date", "value"]).copy()
    fmax = fmax or float(d["flops"].max())
    s_of = lambda f: max(np.pi * max_radius_pt ** 2 * f / fmax, np.pi * min_radius_pt ** 2)
    r_of = lambda f: np.sqrt(s_of(f) / np.pi)
    for _, r in d.iterrows():
        kind = r.get("kind", "final")
        ax.scatter([r["date"]], [r["value"]], s=s_of(r["flops"]), c=RUN_COLORS.get(kind, RUN_COLORS["final"]),
                   alpha=0.6, edgecolors=SURFACE, linewidths=1, zorder=3)
        post = r.get("post_flops", np.nan)
        if pd.notna(post) and post > 0:
            # Ring whose extra area is the post-training compute, at the same scale as the disk. It is
            # drawn only when it would be distinguishable from the disk (at least 2% wider).
            disk = s_of(r["flops"]); outer = disk + np.pi * max_radius_pt ** 2 * post / fmax
            if outer / disk >= 1.04:
                ax.scatter([r["date"]], [r["value"]], s=outer, facecolors="none",
                           edgecolors=RUN_COLORS["ladder"], linewidths=1.5, zorder=2)
        ax.annotate(r["label"], (r["date"], r["value"]), xytext=(r_of(r["flops"]) + 4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, ha="left", va="center")
    for name, v in (ref_lines or {}).items():
        ax.axhline(v, color=GRID, lw=1.5, zorder=1)
        ax.annotate(name, (ax.get_xlim()[0], v), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=MUTED)
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_ylabel(ylabel); ax.set_title(title or "Capability over time (circle area = pretraining compute)")
    ax.margins(x=0.25, y=0.3)
    if notes:
        ax.text(0.99, 0.02, notes, transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="right", va="bottom")
    return ax


LAB_LABELS = {"meta": "Meta Llama", "deepseek": "DeepSeek", "ai2-olmo": "Ai2 OLMo", "marin": "Marin",
              "prime-intellect": "Prime Intellect", "eleutherai": "EleutherAI Pythia", "cerebras": "Cerebras-GPT"}
LAB_ORDER = ["meta", "deepseek", "ai2-olmo", "marin", "prime-intellect", "eleutherai", "cerebras"]

# One H100-hour at 40% of 1e15 FLOP/s = 1.44e18 FLOPs; at $2/hour that is $1.4M per 1e24 FLOPs.
FLOPS_PER_GPU_HOUR = 1.44e18
USD_PER_GPU_HOUR = 2.0


def release_timeline(df: pd.DataFrame, ax=None, title: str | None = None, max_radius_pt: float = 30.0,
                     min_radius_pt: float = 1.6, ref_flops=(1e23, 1e24, 1e25)):
    """Every released model as a circle on a lab row against release date, area proportional to
    training compute. Filled = pretrained from scratch (pretraining compute); hollow orange =
    post-training-only release (post-training compute); dashed grey = planned; x = compute not
    stated. Models a lab released on the same day are drawn as concentric circles with one label."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    labs = [l for l in LAB_ORDER if l in set(df["lab"])] + sorted(set(df["lab"]) - set(LAB_ORDER))
    y_of = {lab: len(labs) - 1 - i for i, lab in enumerate(labs)}
    fmax = float(df["flops"].max())
    s_of = lambda f: max(np.pi * max_radius_pt ** 2 * f / fmax, np.pi * min_radius_pt ** 2)   # points^2
    r_of = lambda f: np.sqrt(s_of(f) / np.pi)

    x_lo = pd.Timestamp("2022-11-01")
    x_hi = max(pd.Timestamp("2027-03-01"), df["release_date"].max() + pd.Timedelta(days=120))
    ax.set_xlim(x_lo, x_hi)
    for y in y_of.values():
        ax.axhline(y, color=GRID, lw=1, zorder=0)

    # Label placement: alternate above/below the row; within a side, step outward to the first
    # level whose already-placed labels do not overlap this one. Widths are estimated from the
    # character count and the axes width in points.
    fontsize = 7.5
    ax_w_pt = ax.figure.get_figwidth() * 72 * ax.get_position().width
    days_per_pt = (x_hi - x_lo).days / ax_w_pt
    placed: dict[tuple[str, int, int], list[tuple[float, float]]] = {}   # (lab, side, level) -> [(x0, x1)] in day numbers
    import matplotlib.dates as mdates
    flip = {lab: 0 for lab in labs}
    for (lab, date), g in df.sort_values("release_date").groupby(["lab", "release_date"], sort=False):
        y = y_of[lab]
        g = g.sort_values("flops", ascending=False, na_position="last")
        kinds = set(g["kind"])
        for _, r in g.iterrows():
            if pd.isna(r["flops"]):
                ax.scatter([date], [y], s=40, marker="x", c=MUTED, linewidths=1.5, zorder=4)
                continue
            s = s_of(r["flops"])
            if r["kind"] == "posttrain":
                ax.scatter([date], [y], s=s, facecolors="none", edgecolors=RUN_COLORS["ladder"], linewidths=1.6, zorder=3)
            elif r["kind"] == "planned":
                ax.scatter([date], [y], s=s, facecolors="none", edgecolors=MUTED, linewidths=1.4, linestyle=(0, (3, 2)), zorder=3)
            else:
                ax.scatter([date], [y], s=s, c=RUN_COLORS["final"], alpha=0.55, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        fam = g["family"].iloc[0]
        variants = [v for v in g["variant"] if v][::-1]          # smallest first reads better
        label = (fam + " " + " / ".join(variants)).strip()
        if g["flops"].isna().all():
            label += " (compute not stated)"
        rmax = r_of(g["flops"].max()) if g["flops"].notna().any() else 4
        side = 1 if flip[lab] % 2 == 0 else -1; flip[lab] += 1
        if g["flops"].notna().any() and g["flops"].max() >= fmax:
            side = 1        # the largest circle spills into the row below; keep its label above
        half_w_days = 0.5 * len(label) * 0.52 * fontsize * days_per_pt
        xc = mdates.date2num(date); span = (xc - half_w_days, xc + half_w_days)
        level = 0
        while any(not (span[1] < a or span[0] > b) for a, b in placed.get((lab, side, level), [])):
            level += 1
        placed.setdefault((lab, side, level), []).append(span)
        dy = side * (rmax + 3 + level * (fontsize + 2))
        ax.annotate(label, (date, y), xytext=(0, dy), textcoords="offset points", ha="center",
                    va="bottom" if side > 0 else "top", fontsize=fontsize,
                    color=INK2 if "pretrain" in kinds or "planned" in kinds else RUN_COLORS["ladder"], zorder=5)

    ax.set_yticks(list(y_of.values())); ax.set_yticklabels([LAB_LABELS.get(l, l) for l in labs], fontsize=9)
    ax.set_ylim(-0.55, len(labs) + 0.05)
    loc = mdates.YearLocator(); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", which="major", color=GRID)
    ax.set_title(title or "Releases by lab, circle area proportional to training compute")

    # Marker legend (below left) and size key (below right, an inset outside the axes), so neither
    # covers a lab row. Key circles use the same point scale as the main axes.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=RUN_COLORS["final"], alpha=0.55, label="Pretrained from scratch (pretraining compute)"),
               Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="none", markeredgecolor=RUN_COLORS["ladder"], markeredgewidth=1.6, label="Post-training-only release (post-training compute)"),
               Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="none", markeredgecolor=MUTED, label="Planned / in progress"),
               Line2D([], [], marker="x", ls="", ms=6, color=MUTED, label="Compute not stated")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.06), fontsize=7.5, ncol=1)
    key = ax.inset_axes([0.50, -0.30, 0.50, 0.26])
    key.set_axis_off(); key.set_xlim(0, 1); key.set_ylim(0, 1)
    key.text(0.02, 0.97, f"Circle area = training compute. Dollars: accelerator time at \\${USD_PER_GPU_HOUR:.0f} per H100-hour, 40% MFU.",
             fontsize=7, color=MUTED, va="top")
    yk = 0.10
    for f in ref_flops:
        r = r_of(f)
        yk += r / 300
        key.scatter([0.14], [yk], s=s_of(f), c=RUN_COLORS["final"], alpha=0.55, edgecolors=SURFACE, linewidths=0.8, clip_on=False)
        usd = f / FLOPS_PER_GPU_HOUR * USD_PER_GPU_HOUR
        key.text(0.28, yk, f"{f:.0e} FLOPs   (about \\${usd / 1e6:,.1f}M)", fontsize=7, color=MUTED, va="center")
        yk += r / 300 + 0.10
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


def paired_deltas(pairs: list[tuple[str, float, float]], base_label: str, new_label: str, ax=None, title: str | None = None):
    """Benchmark before vs after post-training as paired dots with a connector: the RL analogue of the final-run star."""
    style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.6 * len(pairs) + 1.5))
    ys = np.arange(len(pairs))[::-1]
    for y, (name, b, n) in zip(ys, pairs):
        ax.plot([b, n], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([b], [y], s=70, c=MUTED, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=base_label if y == ys[0] else None)
        ax.scatter([n], [y], s=90, c=RUN_COLORS["final"], edgecolors=SURFACE, linewidths=1.5, zorder=4, label=new_label if y == ys[0] else None)
        ax.annotate(f"{n - b:+.1f}", (max(b, n), y), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([p[0] for p in pairs])
    ax.set_xlabel("Score"); ax.set_title(title or f"{new_label} vs {base_label}")
    ax.legend(loc="lower right")
    return ax
