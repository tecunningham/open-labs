"""Thin helpers so each chapter .qmd stays short and identical in structure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "_quarto.yml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fits, load, losses, plots  # noqa: E402


def fmt_flops(x):
    return "" if pd.isna(x) else f"{x:.2e}"


def key_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Compact table of every run with provenance and confidence."""
    cols = ["project", "run_id", "run_type", "stage", "params_total", "tokens", "flops", "flops_method",
            "gpu_hours", "loss", "benchmark_name", "benchmark_value", "confidence", "source_url"]
    t = df[cols].copy()
    for c in ["params_total", "tokens", "flops"]:
        t[c] = t[c].map(fmt_flops)
    t["source_url"] = t["source_url"].map(lambda u: f"[src]({u})" if u else "")
    return t


def standard_figures(lab: str, project: str | None = None, y: str = "loss", irreducible: bool = False,
                     split_by: str = "flops", fit_experiments: bool = True):
    """Emit the chapter's standard figure set and return (df, fit, residuals).

    `fit_experiments=False` draws the points without a fitted law. Use it when the rows that carry
    a loss are continuation phases (midtraining, cooldowns) whose `flops` is per-phase compute, not
    the cumulative compute the loss reflects; a power law through those points means nothing."""
    df = load.runs(lab, project)
    fit = fits.fit_experiments(df, y=y, irreducible=irreducible) if fit_experiments else None
    res = fits.final_residuals(df, fit, y=y)

    if df[y].notna().sum() >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        plots.scaling_curve(df, y=y, fit=fit, ax=ax)
        plt.show()
    else:
        display(Markdown(f"_No `{y}` values recorded for these runs, so there is no loss-vs-compute curve yet. "
                         "The run-size map below shows where experiments sit relative to the final runs._"))
    fig, ax = plt.subplots(figsize=(8, 5))
    plots.ladder_map(df, ax=ax)
    plt.show()

    if not res.empty:
        display(Markdown("**Final runs against the fit on experiments** (log residual < 0 means the final run "
                         "beat the extrapolation; elasticity is d log L / d log C at the final run):"))
        display(res.round(4))

    fig, ax = plt.subplots(figsize=(8, 1.8))
    plots.compute_split(df, ax=ax, value=split_by)
    plt.show()
    display(Markdown(f"Experimental share of logged FLOPs: **{load.experiment_share(df):.0%}** "
                     "(a lower bound: unlogged and failed runs are missing)."))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    plots.run_size_hist(df, ax=ax)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    plots.timeline(df, ax=ax)
    plt.show()
    return df, fit, res


def experiments_table(lab: str):
    e = load.experiments(lab)
    if e.empty:
        display(Markdown("_No published scaling experiments recorded yet._")); return e
    e = e.copy(); e["source_url"] = e["source_url"].map(lambda u: f"[src]({u})" if u else "")
    display(e.drop(columns=["lab"]))
    return e


def claims_table(lab: str):
    c = load.claims(lab)
    if c.empty:
        display(Markdown("_No verbatim lab claims recorded yet._")); return c
    c = c.copy(); c["source_url"] = c["source_url"].map(lambda u: f"[src]({u})" if u else "")
    display(c.drop(columns=["lab"]))
    return c
