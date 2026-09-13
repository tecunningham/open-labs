"""Load the curated CSVs with light typing and derived columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RUN_TYPES = ["ladder", "ablation", "final", "midtrain", "aborted", "unknown"]
STAGES = ["pretraining", "midtraining", "post-training/RL"]
CONFIDENCE = ["reported", "derived", "guess"]

NUMERIC = ["params_total", "params_active", "tokens", "flops", "gpu_hours",
           "cost_usd_est", "loss", "benchmark_value"]


def runs(lab: str | None = None, project: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "runs.csv", dtype=str, keep_default_na=False)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    for c in ["date_start", "date_end"]:
        df[c] = pd.to_datetime(df[c].replace("", np.nan), errors="coerce")
    # Active params default to total; FLOPs default to 6ND when missing.
    df["params_active"] = df["params_active"].fillna(df["params_total"])
    est = 6 * df["params_active"] * df["tokens"]
    missing = df["flops"].isna() & est.notna()
    df.loc[missing, "flops"] = est[missing]
    df.loc[missing, "flops_method"] = "6ND"
    df["run_type"] = pd.Categorical(df["run_type"], RUN_TYPES)
    df["is_final"] = df["run_type"] == "final"
    if lab:
        df = df[df["lab"] == lab]
    if project:
        df = df[df["project"] == project]
    return df.reset_index(drop=True)


def experiments(lab: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "experiments.csv", dtype=str, keep_default_na=False)
    return df[df["lab"] == lab].reset_index(drop=True) if lab else df


def labs(lab: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "labs.csv", dtype=str, keep_default_na=False)
    return df[df["lab"] == lab].reset_index(drop=True) if lab else df


def claims(lab: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(DATA / "claims.csv", dtype=str, keep_default_na=False)
    return df[df["lab"] == lab].reset_index(drop=True) if lab else df


def releases(lab: str | None = None) -> pd.DataFrame:
    """One row per released model (or model group) with a release date and a training-compute
    estimate on one FLOPs scale; `flops_basis` says how each number was obtained. Used for the
    cross-lab release timeline in the preface."""
    df = pd.read_csv(DATA / "releases.csv", dtype=str, keep_default_na=False)
    df["flops"] = pd.to_numeric(df["flops"].replace("", np.nan), errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df[df["lab"] == lab].reset_index(drop=True) if lab else df


def compute_split(df: pd.DataFrame) -> pd.DataFrame:
    """FLOPs, GPU-hours and run counts by run_type. Experiments = everything not final."""
    g = (df.groupby("run_type", observed=True)
           .agg(runs=("run_id", "count"), flops=("flops", "sum"), gpu_hours=("gpu_hours", "sum"))
           .reset_index())
    total = g["flops"].sum()
    g["flops_share"] = g["flops"] / total if total else np.nan
    return g


EXPERIMENT_TYPES = ["ladder", "ablation", "aborted", "midtrain"]


def experiment_share(df: pd.DataFrame) -> float:
    """Share of known FLOPs in ladder/ablation/aborted/midtrain runs. Rows typed `unknown` (e.g. an
    unreleased run still training) are excluded from both numerator and denominator.
    Lower bound: unlogged runs are missing."""
    d = df[df["run_type"] != "unknown"]
    total = d["flops"].sum()
    return float(d.loc[d["run_type"].isin(EXPERIMENT_TYPES), "flops"].sum() / total) if total else float("nan")
