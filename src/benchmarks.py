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
