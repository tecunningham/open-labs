"""AI R&D benchmark scores reported in closed-lab model and system cards.

`data/ai_rd_benchmarks.csv` is long-form: one row per (lab, model, benchmark, subtask, condition).
`wide()` pivots one lab into the table the appendix shows: rows are benchmarks, columns are
successive frontier models in card-date order, cells are the reported score text with a
provenance mark.
"""
from __future__ import annotations

import pandas as pd

from src.load import DATA

LABS = {"gdm": "Google DeepMind", "openai": "OpenAI", "anthropic": "Anthropic"}

# How the number reached this table. `reported` = read from the card itself; `snippet` = read from
# a search-result excerpt of the card or of a write-up quoting it; `memory` = recalled without a
# source check. Marks are appended to the cell.
CONFIDENCE_MARK = {"reported": "", "snippet": "†", "memory": "‡"}

# Row order in the wide tables: AI R&D suites, then research-replication and competition suites,
# then agentic-coding benchmarks, then the lab's own threshold determination.
FAMILY_ORDER = ["re_bench", "ai_rd_suite1", "ai_rd_suite2", "ml_rd_internal", "internal_research_debugging",
                "openai_proof_qa", "mle_bench", "paperbench", "openai_prs", "swe_lancer", "re_interviews",
                "agentic_tasks", "ai_rd_uplift", "metr_external", "swe_bench_verified", "swe_bench_pro",
                "terminal_bench", "agentic_coding_internal", "other", "ml_rd_determination"]


def load(lab: str | None = None, frontier_only: bool = False) -> pd.DataFrame:
    """`frontier_only` keeps the cards flagged `frontier == yes`: the lab's most capable released
    model at the card date, or a card that moved the lab's frontier. Smaller siblings (Sonnet 4,
    Haiku 4.5, Codex addenda, Gemini Flash-Lite) stay in the CSV."""
    df = pd.read_csv(DATA / "ai_rd_benchmarks.csv", dtype=str, keep_default_na=False)
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    df["card_date"] = pd.to_datetime(df["card_date"], errors="coerce")
    if lab:
        df = df[df["lab"] == lab]
    if frontier_only:
        df = df[df["frontier"] == "yes"]
    return df.reset_index(drop=True)


def _cell(rows: pd.DataFrame) -> str:
    parts = []
    for _, r in rows.sort_values("conditions").iterrows():
        s = r["score_text"] or r["score"]
        if not s:
            continue
        if r["conditions"]:
            s = f"{s} ({r['conditions']})"
        parts.append(s + CONFIDENCE_MARK.get(r["confidence"], ""))
    return "<br>".join(parts)


def models(df: pd.DataFrame) -> list[str]:
    """Models in card-date order, then by the order they first appear in the file."""
    order = (df.assign(_i=range(len(df)))
               .groupby("model", sort=False)
               .agg(d=("card_date", "min"), i=("_i", "min"))
               .sort_values(["d", "i"]))
    return list(order.index)


def wide(df: pd.DataFrame, families: list[str] | None = None) -> pd.DataFrame:
    """Rows: benchmark (and subtask); columns: models in card order; cells: reported scores."""
    if families:
        df = df[df["family"].isin(families)]
    df = df.copy()
    df["row"] = df.apply(lambda r: f"{r['benchmark']}: {r['subtask']}" if r["subtask"] else r["benchmark"], axis=1)
    # Rows grouped by family (AI R&D suites first, coding benchmarks after, the lab's threshold
    # determination last), then by first appearance in the file (the CSV is written in card order).
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    df["_fam"] = df["family"].map(lambda f: rank.get(f, len(FAMILY_ORDER) - 1))
    df["_pos"] = range(len(df))
    row_order = list(df.sort_values(["_fam", "_pos"]).drop_duplicates("row")["row"])
    cols = models(df)
    out = pd.DataFrame("", index=row_order, columns=cols)
    for (row, model), g in df.groupby(["row", "model"], sort=False):
        out.loc[row, model] = _cell(g)
    out.index.name = "Benchmark"
    return out


def cards(df: pd.DataFrame) -> pd.DataFrame:
    """One line per model: card date, title, URL."""
    t = (df.groupby("model", sort=False)
           .agg(card_date=("card_date", "min"), card_title=("card_title", "first"), card_url=("card_url", "first"))
           .reset_index())
    t = t.sort_values("card_date")
    t["card_date"] = t["card_date"].dt.strftime("%Y-%m")
    return t.reset_index(drop=True)


def to_markdown(t: pd.DataFrame) -> str:
    """GitHub-style table without the `tabulate` dependency; header cells wrap the model names."""
    cols = [t.index.name or ""] + list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        cells = [str(idx)] + [str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- time series
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from src import plots


def numeric_series(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a numeric score, one series key per benchmark (+ subtask)."""
    d = df[df["score_num"].notna() & df["card_date"].notna()].copy()

    def key(r):
        s = f"{r['benchmark']}: {r['subtask']}" if r["subtask"] else r["benchmark"]
        # Terminal-Bench changed version several times; each version is its own series.
        m = re.search(r"\bv\d(?:\.\d)?", r["conditions"]) if r["family"] == "terminal_bench" else None
        if m:
            s += f" {m.group(0)}"
        return s
    d["series"] = d.apply(key, axis=1)
    # A benchmark reported in two different units (percent uplift vs a multiple) is two series.
    d["_unit"] = d["metric"].map(_unit)
    d["series"] = d["series"] + d.groupby("series")["_unit"].transform(
        lambda u: u.map(lambda v: "" if u.nunique() == 1 else f" [{v}]"))
    return d


def _unit(metric: str) -> str:
    m = metric.lower()
    if "%" in m or "percent" in m:
        return "percent"
    if "0 to 1" in m:
        return "fraction"
    if "speedup" in m or "multiple" in m or "acceleration" in m or m.endswith(" x"):
        return "multiple"
    return "score"


def _short(model: str) -> str:
    for p in ("Claude ", "Gemini ", "OpenAI "):
        model = model.replace(p, "")
    return model.replace(" (Oct 2024)", " Oct24").replace(" Thinking", "")


def timeseries(df: pd.DataFrame, title: str | None = None, ncols: int = 3, min_points: int = 2):
    """Small multiples: one panel per benchmark series with >= min_points numeric scores.

    Within a panel the lowest score per model (the base setting, e.g. no parallel compute) is
    joined by a line; other conditions for the same model are hollow markers. Points from
    rows not yet checked against the card (snippet or memory) are drawn with a lighter fill.
    Speedup-style series spanning more than 20x use a log y-axis."""
    plots.style()
    d = numeric_series(df)
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    keep = [s for s, g in d.groupby("series") if g["model"].nunique() >= min_points]
    order = (d[d["series"].isin(keep)].groupby("series")
               .agg(fam=("family", lambda x: rank.get(x.iloc[0], 99)), t=("card_date", "min"))
               .sort_values(["fam", "t"]).index.tolist())
    if not order:
        fig, ax = plt.subplots(figsize=(8, 2)); plots._empty(ax, "No series with two or more numeric scores"); return fig
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False, sharex=True)
    x0, x1 = d["card_date"].min(), d["card_date"].max()
    pad = pd.Timedelta(days=45)
    for ax, s in zip(axes.flat, order):
        g = d[d["series"] == s].sort_values("card_date")
        base = g.loc[g.groupby("model")["score_num"].idxmin()].sort_values("card_date")
        extra = g.drop(base.index)
        c = plots.RUN_COLORS["final"]
        ax.plot(base["card_date"], base["score_num"], color=c, lw=1.5, zorder=2)
        verified = base["confidence"] == "reported"
        ax.scatter(base["card_date"][verified], base["score_num"][verified], s=28, c=c, zorder=3,
                   edgecolors=plots.SURFACE, linewidths=1)
        ax.scatter(base["card_date"][~verified], base["score_num"][~verified], s=28, c="#a9c8ee", zorder=3,
                   edgecolors=c, linewidths=1)
        if not extra.empty:
            ax.scatter(extra["card_date"], extra["score_num"], s=26, facecolors="none", edgecolors=c, linewidths=1.2, zorder=3)
        for i, (_, r) in enumerate(base.iterrows()):
            up = i % 2 == 0
            ax.annotate(_short(r["model"]), (r["card_date"], r["score_num"]), xytext=(0, 6 if up else -6),
                        textcoords="offset points", ha="center", va="bottom" if up else "top",
                        fontsize=6.5, color=plots.INK2)
        ptitle = re.sub(r" \[.*\]$", "", s)
        ax.set_title("\n".join(textwrap.wrap(ptitle, 40)[:2]), fontsize=8.5)
        metric = g["metric"].iloc[0]
        ax.set_ylabel(textwrap.fill(metric, 22), fontsize=7)
        lo, hi = base["score_num"].min(), base["score_num"].max()
        if lo > 0 and hi / lo > 20:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0 if lo >= 0 else None)
            ax.margins(y=0.25)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.tick_params(labelsize=7.5)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
