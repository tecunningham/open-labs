"""Validate and merge agent-collected CSV fragments into data/*.csv.

Usage: python -m src.merge <dir> [<dir> ...]   (each dir holds runs.csv, experiments.csv, labs.csv, claims.csv, notes.md)
Rows are validated against the schema, de-duplicated on natural keys, and appended. Invalid rows are reported and skipped.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

from .load import CONFIDENCE, DATA, RUN_TYPES, STAGES

KEYS = {
    "runs.csv": ["lab", "project", "run_id", "benchmark_name"],
    "experiments.csv": ["lab", "project", "intervention", "source_url"],
    "labs.csv": ["lab", "project"],
    "claims.csv": ["lab", "project", "claim", "source_url"],
}
NUMERIC_RUN_COLS = ["params_total", "params_active", "tokens", "flops", "gpu_hours", "cost_usd_est", "loss", "benchmark_value"]
URL = re.compile(r"^https?://\S+$")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)


def validate(name: str, df: pd.DataFrame, schema_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    errs = []
    extra, missing = set(df.columns) - set(schema_cols), set(schema_cols) - set(df.columns)
    if missing:
        errs.append(f"{name}: missing columns {sorted(missing)}"); return df.iloc[0:0], errs
    if extra:
        errs.append(f"{name}: dropping unknown columns {sorted(extra)}"); df = df[schema_cols]
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].str.strip()
    keep = pd.Series(True, index=df.index)
    if name == "runs.csv":
        bad = ~df["run_type"].isin(RUN_TYPES); errs += [f"runs: bad run_type {r!r} in {i}" for i, r in df.loc[bad, "run_type"].items()]; keep &= ~bad
        bad = ~df["stage"].isin(STAGES); errs += [f"runs: bad stage {r!r} in {i}" for i, r in df.loc[bad, "stage"].items()]; keep &= ~bad
        bad = ~df["confidence"].isin(CONFIDENCE); errs += [f"runs: bad confidence {r!r} in {i}" for i, r in df.loc[bad, "confidence"].items()]; keep &= ~bad
        for c in NUMERIC_RUN_COLS:
            v = pd.to_numeric(df[c].replace("", None), errors="coerce")
            bad = (df[c] != "") & v.isna(); errs += [f"runs: non-numeric {c}={r!r} in {i}" for i, r in df.loc[bad, c].items()]; keep &= ~bad
    if "source_url" in df.columns:
        # Allow "url; url; ..." lists: validate the first URL only.
        first = df["source_url"].str.split(";").str[0].str.strip()
        bad = (df["source_url"] != "") & ~first.str.match(URL)
        errs += [f"{name}: bad source_url {r!r}" for r in df.loc[bad, "source_url"]]; keep &= ~bad
        nourl = df["source_url"] == ""
        if nourl.any():
            errs.append(f"{name}: {int(nourl.sum())} rows without source_url (kept, flagged)")
    if "raw_data_available" in df.columns:
        enum = ["logs", "figure_only", "numbers_in_text", ""]
        bad = ~df["raw_data_available"].isin(enum)
        for i in df.index[bad]:
            txt = df.at[i, "raw_data_available"]; low = txt.lower()
            cls = "logs" if any(k in low for k in ("wandb", "json", "csv", "checkpoint", "yes")) else (
                  "numbers_in_text" if "partial" in low or "table" in low else "figure_only")
            df.at[i, "notes"] = (df.at[i, "notes"] + " | " if df.at[i, "notes"] else "") + f"raw data: {txt}"
            df.at[i, "raw_data_available"] = cls
            errs.append(f"experiments: raw_data_available free text -> {cls!r} (kept text in notes)")
    return df[keep], errs


def merge_dir(d: Path) -> None:
    for name, keys in KEYS.items():
        src = d / name
        if not src.exists():
            print(f"  {name}: absent"); continue
        target = DATA / name
        schema = list(_read(target).columns)
        new, errs = validate(name, _read(src), schema)
        for e in errs:
            print("  !", e)
        cur = _read(target)
        before = len(cur)
        merged = pd.concat([cur, new], ignore_index=True).drop_duplicates(subset=keys, keep="last")
        merged.to_csv(target, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"  {name}: +{len(merged) - before} rows ({len(new)} candidate, {len(merged)} total)")
    notes = d / "notes.md"
    if notes.exists():
        out = DATA / "notes" / f"{d.name}.md"; out.parent.mkdir(exist_ok=True)
        out.write_text(notes.read_text()); print(f"  notes -> {out.relative_to(DATA.parent)}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"== {arg}"); merge_dir(Path(arg))
