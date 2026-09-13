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

# How the number reached this table. `reported` = read from the card itself (or the METR report the
# card cites); `announcement` = read from the lab's launch post, not the card; `snippet` = read from
# a search-result excerpt of the card or of a write-up quoting it; `memory` = recalled without a
# source check. Marks are appended to the cell.
CONFIDENCE_MARK = {"reported": "", "announcement": "*", "snippet": "†", "memory": "‡"}

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
    # An internal task rescaled between cards is split at the known break (see SCALE_BREAKS).
    for series, (cut, lo_label, hi_label) in SCALE_BREAKS.items():
        m = d["series"] == series
        d.loc[m, "series"] = d.loc[m, "score_num"].map(lambda v: f"{series} ({hi_label if v >= cut else lo_label})")
    # A benchmark reported in two different units (percent uplift vs a multiple) is two series.
    d["_unit"] = d["metric"].map(_unit)
    d["series"] = d["series"] + d.groupby("series")["_unit"].transform(
        lambda u: u.map(lambda v: "" if u.nunique() == 1 else f" [{v}]"))
    return d


# Series whose scores are not comparable across a card boundary: (cutoff, label below, label at or above).
# Anthropic's quadruped task was re-normalised between Sonnet 4.5 (threshold 1.0) and Opus 4.5 (threshold 12).
SCALE_BREAKS = {
    "Internal AI Research Evaluation Suite 1: Quadruped RL": (5, "threshold 1 scale", "threshold 12 scale"),
}


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


def timeseries(df: pd.DataFrame, title: str | None = None, ncols: int = 3, min_points: int = 2, lab: str | None = None):
    """Small multiples: one panel per benchmark series with >= min_points numeric scores.

    Within a panel the lowest score per model (the base setting, e.g. no parallel compute) is
    joined by a line; other conditions for the same model are hollow markers. Points from
    rows not yet checked against the card (snippet or memory) are drawn with a lighter fill.
    Speedup-style series spanning more than 20x use a log y-axis."""
    plots.style()
    d = numeric_series(df)
    lab = lab or (df["lab"].iloc[0] if len(df) else None)
    meta = series_meta()
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
    deps = [meta_for(meta, lab, s) for s in order]
    deps = [m["deprecated_date"] for m in deps if m is not None and pd.notna(m["deprecated_date"])]
    if deps:
        x1 = max([x1] + deps)
    pad = pd.Timedelta(days=45)
    for ax, s in zip(axes.flat, order):
        g = d[d["series"] == s].sort_values("card_date")
        base = g.loc[g.groupby("model")["score_num"].idxmin()].sort_values("card_date")
        extra = g.drop(base.index)
        c = plots.RUN_COLORS["final"]
        m = meta_for(meta, lab, s)
        ax.plot(base["card_date"], base["score_num"], color=c, lw=1.5, zorder=2)
        verified = base["confidence"].isin(["reported", "announcement"])
        ax.scatter(base["card_date"][verified], base["score_num"][verified], s=28, c=c, zorder=3,
                   edgecolors=plots.SURFACE, linewidths=1)
        ax.scatter(base["card_date"][~verified], base["score_num"][~verified], s=28, c="#a9c8ee", zorder=3,
                   edgecolors=c, linewidths=1)
        if not extra.empty:
            ax.scatter(extra["card_date"], extra["score_num"], s=26, facecolors="none", edgecolors=c, linewidths=1.2, zorder=3)
        _retire_mark(ax, m, base["card_date"].iloc[-1], base["score_num"].iloc[-1], c)
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
        refs = [v for v in ([m["human_ref"], m["threshold"]] if m is not None else []) if pd.notna(v)]
        if m is not None and pd.notna(m["ceiling"]):
            # Bounded metric: show the whole range so saturation is visible.
            ax.set_ylim(0, m["ceiling"] * 1.06)
            ax.axhline(m["ceiling"], color=plots.INK2, lw=0.8, zorder=1)
        elif lo > 0 and hi / lo > 20:
            ax.set_yscale("log")
            if refs:
                ax.set_ylim(min(lo, *refs) / 1.6, max(hi, *refs) * 1.9)
        else:
            top = max([hi] + refs) * 1.3
            ax.set_ylim(0 if lo >= 0 else None, top)
        ax.set_xlim(x0 - pad, x1 + pad)
        _ref_lines(ax, m, x0, x1)
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


# ----------------------------------------------------------------------------- ceilings and overview
CATEGORY_COLORS = {"research": "#2a78d6", "swe": "#eb6834", "terminal": "#1baf7a", "knowledge": "#eda100"}
CATEGORY_LABELS = {"research": "AI research tasks", "swe": "Software engineering", "terminal": "Terminal and agentic",
                   "knowledge": "Knowledge"}
HUMAN_REF_COLOR = "#c8322b"
_SHORT = [("Internal AI Research Evaluation Suite 1: ", "Suite 1 "), ("Internal AI Research Evaluation Suite 2", "Suite 2"),
          ("OpenAI Research Engineer interviews: ", "RE interview "), ("MLE-bench: ", ""), ("PaperBench: ", ""),
          ("SWE-bench Verified: ", "SWE-V "), ("SWE-bench Verified", "SWE-bench Verified"), (" (RSP checkpoint)", ""),
          ("GRB (GDM internal research engineering benchmark)", "GRB (internal)"), ("SWE-Lancer IC SWE Diamond", "SWE-Lancer Diamond"),
          ("Agentic tasks (autonomy suite)", "Agentic tasks"), ("Internal agentic coding evaluation", "Internal agentic coding"),
          ("FrontierBench v0.1 (Terminal-Bench successor)", "FrontierBench v0.1"), ("Internal Research Debugging Eval", "Research debugging"),
          (" (75 competitions, AIDE)", " full (pass@10)"), ("MLE-Bench Revised", "MLE-bench revised")]


def series_meta() -> pd.DataFrame:
    m = pd.read_csv(DATA / "ai_rd_benchmark_series.csv", dtype=str, keep_default_na=False)
    for c in ["ceiling", "human_ref", "threshold"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["deprecated_date"] = pd.to_datetime(m["deprecated_date"], errors="coerce")
    return m.set_index(["lab", "series"])


def _retire_mark(ax, m, last_x, last_y, color):
    """Black cross where the lab stopped reporting the series, joined to its last value."""
    if m is None or pd.isna(m.get("deprecated_date", pd.NaT)):
        return None
    xd = m["deprecated_date"]
    if xd > last_x:
        ax.plot([last_x, xd], [last_y, last_y], color=color, lw=0.8, ls=(0, (2, 2)), alpha=0.7, zorder=2)
    ax.scatter([xd], [last_y], marker="x", s=44, c=plots.INK, linewidths=1.4, zorder=5)
    return xd


def meta_for(meta: pd.DataFrame, lab: str, series: str):
    """Exact series match, else the series without its version / scale / unit suffix."""
    for key in (series, re.sub(r" (v\d(?:\.\d)?|\[.*\]|\(.*scale\))$", "", series).strip(),
                re.sub(r" (v\d(?:\.\d)?|\[.*\])$", "", re.sub(r" \(.*scale\)", "", series)).strip()):
        if (lab, key) in meta.index:
            return meta.loc[(lab, key)]
    return None


def short_name(series: str) -> str:
    for a, b in _SHORT:
        series = series.replace(a, b)
    return re.sub(r" \[.*\]$", "", series)


def _ref_lines(ax, m, x0, x1):
    """Human reference (red dotted) and rule-out threshold (grey dashed) with right-hand labels."""
    if m is None:
        return
    if pd.notna(m["human_ref"]):
        ax.axhline(m["human_ref"], color=HUMAN_REF_COLOR, ls=(0, (1.5, 2.5)), lw=1.2, zorder=1)
        ax.annotate(m["human_ref_label"], (0.01, m["human_ref"]), xycoords=("axes fraction", "data"), xytext=(0, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=6.3, color=HUMAN_REF_COLOR)
    if pd.notna(m["threshold"]):
        ax.axhline(m["threshold"], color=plots.MUTED, ls=(0, (4, 3)), lw=1, zorder=1)
        ax.annotate(m["threshold_label"], (0.01, m["threshold"]), xycoords=("axes fraction", "data"), xytext=(0, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=6.3, color=plots.INK2)


def overview(df: pd.DataFrame, lab: str, title: str | None = None, min_points: int = 2):
    """Every bounded series of one lab on a single 0 to 100 percent axis, as percent of its ceiling.

    Colour is the benchmark category (fixed slots); identity is the label at each line's last
    point. The ceiling is the top of the axis. Lower-is-better and unbounded series are excluded
    (see `overview` in data/ai_rd_benchmark_series.csv)."""
    plots.style()
    meta = series_meta()
    d = numeric_series(df)
    rows = []
    for s, g in d.groupby("series"):
        m = meta_for(meta, lab, s)
        if m is None or m["overview"] != "yes" or pd.isna(m["ceiling"]) or m["direction"] != "higher":
            continue
        base = g.loc[g.groupby("model")["score_num"].idxmin()].sort_values("card_date")
        if base["model"].nunique() < min_points:
            continue
        base = base.assign(pct=base["score_num"] / m["ceiling"] * 100, category=m["category"], series=s)
        rows.append((base, m))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    if not rows:
        plots._empty(ax, "No bounded series with two or more points"); return fig
    allb = pd.concat([b for b, _ in rows])
    x0, x1 = allb["card_date"].min(), allb["card_date"].max()
    retire_dates = [m["deprecated_date"] for _, m in rows if pd.notna(m["deprecated_date"])]
    x1 = max([x1] + retire_dates)
    span = (x1 - x0).days or 1
    ax.axhline(100, color=plots.INK2, lw=1, zorder=1)
    ax.annotate("ceiling", (x0, 100), xytext=(0, 3), textcoords="offset points", fontsize=7, color=plots.INK2)
    ends = []
    for b, m in sorted(rows, key=lambda bm: bm[0]["card_date"].min()):
        c = CATEGORY_COLORS[b["category"].iloc[0]]
        ax.plot(b["card_date"], b["pct"], color=c, lw=1.6, alpha=0.9, zorder=2)
        ax.scatter(b["card_date"], b["pct"], s=16, c=c, edgecolors=plots.SURFACE, linewidths=0.8, zorder=3)
        last = b.iloc[-1]
        xd = _retire_mark(ax, m, last["card_date"], last["pct"], c)
        ends.append((max(last["card_date"], xd) if xd is not None else last["card_date"], last["pct"],
                     short_name(b["series"].iloc[0]), c))
    # Right-hand labels, nudged apart so they do not overlap.
    ends.sort(key=lambda e: e[1])
    ys = [e[1] for e in ends]
    gap = 3.6
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    over = ys[-1] - 104 if ys and ys[-1] > 104 else 0
    ys = [y - over for y in ys]
    xend = x1 + pd.Timedelta(days=45)
    xlab = x1 + pd.Timedelta(days=int(span * 0.07))
    for (xd, y, name, c), yl in zip(ends, ys):
        ax.plot([xd, xlab], [y, yl], color=c, lw=0.6, alpha=0.6, zorder=1, clip_on=False)
        ax.annotate(name, (xlab, yl), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.8,
                    color=plots.INK2, annotation_clip=False)
    ax.set_ylim(0, 108)
    ax.set_xlim(x0 - pd.Timedelta(days=30), xend)
    fig.subplots_adjust(right=0.78)
    ax.set_ylabel("Score as percent of the benchmark ceiling")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    handles = [plt.Line2D([], [], color=CATEGORY_COLORS[k], lw=2, label=CATEGORY_LABELS[k])
               for k in CATEGORY_COLORS if k in set(allb["category"])]
    if retire_dates:
        handles.append(plt.Line2D([], [], color=plots.INK, marker="x", ls="none", markersize=7, markeredgewidth=1.4,
                                  label="retired or replaced (card stops reporting it)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, title=None)
    ax.set_title(title or f"{LABS.get(lab, lab)}: every bounded AI R&D evaluation, as percent of its ceiling")
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    return fig
