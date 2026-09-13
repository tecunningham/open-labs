"""Tidy Marin's evalchemy evaluation runs of fine-tuned Marin bases into one CSV.

Marin logs every evalchemy evaluation as its own W&B run in `marin-community/marin`, named like
`evalchemy-exp3956k-marin-8b-base-finetuned-ot4-10k_pt2-math-kimi-k2pt5-32768tokens-step394-AIME24-avg10seeds`,
with the benchmark mean and std in the run summary. This script pulls every such run
(anonymously, see fetch_wandb.py) and writes data/posttraining/marin_sft_scaling.csv with one
row per (base, teacher, examples, repeat, checkpoint, benchmark).

    python -m src.fetch_evalchemy
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from .fetch_wandb import ROOT, _j, gql

OUT = ROOT / "data" / "posttraining" / "marin_sft_scaling.csv"
PROJECT = ("marin-community", "marin")
NAME_FILTER = {"displayName": {"$regex": "^evalchemy-.*marin-(8b|32b)-base"}}
PARAMS = {"8b": 8.03e9, "32b": 3.2e10}

# A lighter query than fetch_wandb.RUNS_Q: no historyKeys, which is large for thousands of runs.
Q = """
query Runs($entity: String!, $project: String!, $cursor: String, $per: Int!, $filters: JSONString, $order: String) {
  project(name: $project, entityName: $entity) {
    runCount(filters: $filters)
    runs(first: $per, after: $cursor, filters: $filters, order: $order) {
      edges { node { name displayName state createdAt summaryMetrics } }
      pageInfo { hasNextPage endCursor }
    }
  }
}"""

PAT = re.compile(r"^evalchemy-(?P<exp>exp\d+[a-z0-9]*)-marin-(?P<size>8b|32b)-base-finetuned-ot4-(?P<n>\d+[kK]?)"
                 r"(?:_pt(?P<rep>\d))?-math-(?P<teacher>.+?)-(?P<maxtok>\d+)tokens-step(?P<step>\d+)-(?P<bench>[A-Za-z0-9]+)"
                 r"-avg(?P<seeds>\d+)seeds$")


def fetch() -> list[dict]:
    entity, project = PROJECT
    cursor, recs = None, []
    while True:
        d = gql(Q, {"entity": entity, "project": project, "cursor": cursor, "per": 100,
                    "filters": json.dumps(NAME_FILTER), "order": "-created_at"})
        p = d["project"]
        for e in p["runs"]["edges"]:
            n = e["node"]; sm = _j(n["summaryMetrics"])
            nums = {k: v for k, v in sm.items() if isinstance(v, (int, float)) and not k.startswith("_")}
            recs.append({"name": n["displayName"], "id": n["name"], "created": n["createdAt"], "state": n["state"], "metrics": nums})
        pi = p["runs"]["pageInfo"]
        if not pi["hasNextPage"]:
            break
        cursor = pi["endCursor"]
    print(f"{len(recs)} evalchemy runs on Marin bases (project reports {p['runCount']})")
    return recs


def tidy(recs: list[dict]) -> pd.DataFrame:
    rows, skipped = [], Counter()
    for x in recs:
        m = PAT.match(x["name"])
        if not m:
            skipped["per-seed run or other name"] += 1; continue
        g = m.groupdict(); b = g["bench"].lower()
        mean, std = x["metrics"].get(f"{b}/correct_mean"), x["metrics"].get(f"{b}/correct_std")
        if mean is None:
            skipped["no mean metric"] += 1; continue
        n = g["n"].lower(); n = int(float(n[:-1]) * 1000) if n.endswith("k") else int(n)
        rows.append({"base": f"marin-{g['size']}-base", "params": PARAMS[g["size"]], "exp": g["exp"], "teacher": g["teacher"],
                     "n_examples": n, "repeat": int(g["rep"] or 1), "max_tokens": int(g["maxtok"]), "step": int(g["step"]),
                     "benchmark": g["bench"], "mean": round(mean, 4), "std": round(std, 4) if std is not None else None,
                     "n_seeds": int(g["seeds"]), "wandb_run": x["id"], "created": x["created"][:10], "state": x["state"]})
    print("skipped:", dict(skipped))
    return pd.DataFrame(rows).sort_values(["base", "teacher", "n_examples", "repeat", "step", "benchmark"])


if __name__ == "__main__":
    df = tidy(fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT.relative_to(ROOT)}")
