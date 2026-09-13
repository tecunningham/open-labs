"""Pull loss curves from public W&B projects into data/losses/<lab>/<run>.csv.

Talks to W&B's GraphQL endpoint directly with `requests`, **anonymously**. Public projects are
readable without any key, and in the Sept 2026 collection session a personal API key was in fact
*refused* for every public project ("the provided API key cannot access this resource") while the
same queries succeeded unauthenticated. So: do not set WANDB_API_KEY for this script, and it needs
no `wandb` package (PyPI was blocked in that session anyway).

    python -m src.fetch_wandb                 # all labs
    python -m src.fetch_wandb ai2-olmo        # one lab
    python -m src.fetch_wandb --list ai2-llm/Olmo-3-1025-7B [--max 100] [--filter '{"name":{"$regex":"..."}}']
    python -m src.fetch_wandb --summary       # print final-step losses for every fetched CSV
    python -m src.fetch_wandb --apply         # write those losses into data/runs.csv (loss, loss_eval_set, notes)

Output: one CSV per target (`_step` + the loss keys that were logged, downsampled to ~SAMPLES points
per segment, restart segments concatenated and sorted) and a `.meta.json` beside it with the run
URLs, batch-size-in-tokens (when recoverable) and the final-step loss of each key.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "losses"
URL = "https://api.wandb.ai/graphql"
SAMPLES = 1500
MAX_ROWS = 3000  # per curve after concatenating segments; eval-key rows are always kept


@dataclass
class Target:
    """One curve = one runs.csv row. `runs` are W&B run ids/names in `project`, concatenated in
    order (restart segments). `keys` are the history keys to pull. `tokens_per_step` lets a
    reader turn `_step` into tokens; None when the batch schedule varied."""
    run_id: str                 # runs.csv run_id this curve belongs to
    project: str                # entity/project
    runs: list[str]
    keys: list[str]
    tokens_per_step: float | None = None
    note: str = ""
    aliases: list[str] = field(default_factory=list)  # other runs.csv run_ids that share this curve


OLMO3_KEYS = ["train/CE loss", "eval/lm/c4_en-validation/CE loss", "eval/lm/pile-validation/CE loss",
              "eval/lm/dolma_common-crawl-validation/CE loss"]
OLMO3_STAGE2_KEYS = ["train/CE loss"]
OLMO2_KEYS = ["train/CrossEntropyLoss"]
PYTHIA_KEYS = ["validation/lm_loss", "train/lm_loss"]
MARIN_KEYS = ["train/loss", "eval/paloma/c4_en/loss", "eval/loss", "eval/macro_loss"]

# Ai2: the stage-1 runs are logged as dozens of restart segments in one W&B group; we pull the
# whole group (see GROUPS below) rather than listing segment ids by hand.
GROUPS = {
    "olmo3-7B-stage1 (internal: OLMo25)": ("ai2-llm/Olmo-3-1025-7B", "Olmo-3-1025-7B-stage-1"),
    "olmo3-32B-stage1 (internal: stego32-highlr-filter3)": ("ai2-llm/Olmo-3-1125-32B", "Olmo-3-1125-32B-stage-1"),
    "olmo2-7B-stage1": ("ai2-llm/OLMo-2-1124-7B", "OLMo-2-1124-7B-stage-1"),
    "olmo2-13B-stage1": ("ai2-llm/OLMo-2-1124-13B", "OLMo-2-1124-13B-stage-1"),
}

TARGETS: dict[str, list[Target]] = {
    "ai2-olmo": [
        # Olmo 3 (OLMo-core logs `train/CE loss` every step and in-loop LM evals on c4_en, pile, ...).
        # 7B stage 1: 4,194,304 tok/step (global_batch_size in the run config); 32B: 8,388,608.
        Target("olmo3-7B-stage1 (internal: OLMo25)", "ai2-llm/Olmo-3-1025-7B", [], OLMO3_KEYS, 4_194_304,
               "39 restart segments, group Olmo-3-1025-7B-stage-1; ends at step 1,413,814 = 5.93T tokens"),
        Target("olmo3-7B-stage2-midtrain", "ai2-llm/Olmo-3-1025-7B", ["j9nypgrc", "zxv811e1", "ycefe7k4"], OLMO3_STAGE2_KEYS, None,
               "anneal-round5-100B (3 segments, step counter restarts); the released 100B midtrain"),
        Target("olmo3-7B-stage3-longcontext", "ai2-llm/Olmo-3-1025-7B", ["tx9cibv8"], OLMO3_STAGE2_KEYS, None,
               "64k long-context stage, 50B tokens"),
        Target("olmo3-32B-stage1 (internal: stego32-highlr-filter3)", "ai2-llm/Olmo-3-1125-32B", [], OLMO3_KEYS, 8_388_608,
               "69 restart segments, group Olmo-3-1125-32B-stage-1; ends at step 678,999 = 5.70T tokens "
               "(midtraining branched at 656,000 = 5.50T)"),
        Target("olmo3-32B-stage2-midtrain-ingredients-1+2", "ai2-llm/Olmo-3-1125-32B", ["ufsc2yts", "0gquyaos"], OLMO3_STAGE2_KEYS, 4_194_304,
               "two 100B midtraining ingredients (run-2, run-4), souped; stored as two segments with the same step range"),
        Target("olmo3-32B-stage3-longcontext", "ai2-llm/Olmo-3-1125-32B", ["u0waaxom"], OLMO3_STAGE2_KEYS, 8_388_608,
               "long-context run-3, 100B tokens"),
        # OLMo 2 (old OLMo trainer: `train/CrossEntropyLoss`, no in-loop LM eval logged to W&B).
        # 7B: 1024 seq x 4096 = 4,194,304 tok/step; 13B: 2048 x 4096 = 8,388,608.
        Target("olmo2-7B-stage1", "ai2-llm/OLMo-2-1124-7B", [], OLMO2_KEYS, 4_194_304,
               "11 restart segments; ends at step 928,646 = 3.9T tokens"),
        Target("olmo2-7B-stage2-soup", "ai2-llm/OLMo-2-1124-7B", ["7xkf4smi", "m9lacg4r", "ims1w5bt"], OLMO2_KEYS, 4_194_304,
               "3 x 50B anneal ingredients (souped), steps 928,646 -> 940,577"),
        Target("olmo2-13B-stage1", "ai2-llm/OLMo-2-1124-13B", [], OLMO2_KEYS, 8_388_608,
               "75 restart segments; ends at step 596,057 = 5.0T tokens"),
        Target("olmo2-13B-stage2-soup", "ai2-llm/OLMo-2-1124-13B", ["x5bz8z33", "4vhawieh", "5icuybbp", "iuxd2avq"], OLMO2_KEYS, 8_388_608,
               "3 x 100B + 1 x 300B anneal ingredients (souped)"),
    ],
    # Pythia: `eleutherai/pythia` has 10k runs; these are the finished 143k-step runs whose config
    # (layers, hidden, batch 1024 x 2048, Pile vs deduped Pile path) matches the released model.
    # 2M tokens/step. The repo README's own mapping is "rough and partial"; it agrees on 2.8b,
    # 2.8b-deduped and 1b. No finished deduped run matching 1b, 6.9b or 12b was found.
    "eleutherai": [
        Target("pythia-70m", "eleutherai/pythia", ["32t0zbcs"], PYTHIA_KEYS, 2_097_152, "group 'v2 70M'"),
        Target("pythia-70m-deduped", "eleutherai/pythia", ["2vcd53l4"], PYTHIA_KEYS, 2_097_152, "group 'v2-70M-deduped'"),
        Target("pythia-160m", "eleutherai/pythia", ["3mvtbwii"], PYTHIA_KEYS, 2_097_152, "group 'v2 160M'"),
        Target("pythia-160m-deduped", "eleutherai/pythia", ["38uk24gn"], PYTHIA_KEYS, 2_097_152, "group 'v2 160M deduped'"),
        Target("pythia-410m", "eleutherai/pythia", ["12j05401"], PYTHIA_KEYS, 2_097_152, "group 'v2-410m'"),
        Target("pythia-410m-deduped", "eleutherai/pythia", ["2pmqy4bd"], PYTHIA_KEYS, 2_097_152, "group 'v2-410m-deduped'"),
        Target("pythia-1b", "eleutherai/pythia", ["2i9stqg2"], PYTHIA_KEYS, 2_097_152, "group '800M Pythia' (README mapping)"),
        Target("pythia-1.4b", "eleutherai/pythia", ["vd5ogsc6"], PYTHIA_KEYS, 2_097_152, "group 'v2 1.4B'"),
        Target("pythia-1.4b-deduped", "eleutherai/pythia", ["2dca9aar"], PYTHIA_KEYS, 2_097_152, "group 'v2 1.4B deduped'"),
        Target("pythia-2.8b", "eleutherai/pythia", ["12vuw5ef"], PYTHIA_KEYS, 2_097_152, "group '2.7B New' (README mapping)"),
        Target("pythia-2.8b-deduped", "eleutherai/pythia", ["4aqjesl8"], PYTHIA_KEYS, 2_097_152, "group '2.7B Deduped New' (README mapping)"),
        Target("pythia-6.9b", "eleutherai/pythia", ["vi7laank"], PYTHIA_KEYS, 2_097_152, "group '6.7B Decay' (32L x 4096, Pile)"),
        Target("pythia-12b", "eleutherai/pythia", ["kg5ni1dl"], PYTHIA_KEYS, 2_097_152, "group 'v2-12B' (36L x 5120, Pile)"),
    ],
    # Marin: everything is in the public `marin-community/marin` project (the 8B retrospective's
    # `stanford-mercury/marin` links are private). Run *names* are stable; display names are not.
    # Levanter logs `train/loss` and Paloma eval losses; `eval/paloma/c4_en/loss` is the one the
    # retrospectives plot. 8B batch schedule: 1024 seq x 4096 to step 660k, 3072 to 1.32M, then 4096
    # (tokens_per_step left None). 32B: 8192 seq x 4096 = 33,554,432 tok/step from step 21k on.
    "marin": [
        Target("exp600 phase1 Kestrel (DCLM, WSD-S)", "marin-community/marin", ["llama-8b-tootsie-0.001-19ad63"], MARIN_KEYS, None,
               "steps 0-660,600 at 1024x4096 = 2.77T"),
        Target("exp600 phase2 Ocelot (DCLM, WSD+EMA)", "marin-community/marin", ["llama-8b-tootsie-phase2"], MARIN_KEYS, None,
               "steps 660k-742k at 3072x4096"),
        Target("exp600 phase3 Jellyfish (first cooldown, 'monumental-jellyfish')", "marin-community/marin", ["llama-8b-tootsie-phase3"], MARIN_KEYS, None,
               "steps 742k-820k"),
        Target("exp600 dessert runs (math + FLAN patches off Jellyfish)", "marin-community/marin", ["llama-8b-tootsie-dessert"], MARIN_KEYS, None,
               "steps 820k-837k"),
        Target("exp600 phase4 Phoenix (rewarm, Nemotron-CC)", "marin-community/marin", ["llama-8b-tootsie-adept-phoenix"], MARIN_KEYS, None,
               "steps 820k-1,321k at 3072x4096 = 6.3T; W&B state 'crashed' but ran to its planned end"),
        Target("exp898 Raccoon (deep cooldown 1.7e-4->1.7e-5 from Jellyfish)", "marin-community/marin", ["tootsie-8b-soft-raccoon-3"], MARIN_KEYS, None,
               "third attempt (soft-raccoon-3) is the one that finished"),
        Target("exp916 Spoonbill (cooldown to 3e-5 + Tulu 0.3% + FLAN 1%, z-loss 1e-4 fix)", "marin-community/marin", ["tootsie-8b-focused-spoonbill-zloss"], MARIN_KEYS, None,
               "finished variant with the z-loss fix; 'tootsie-8b-hypnotic-spoonbill-2' is the report's run"),
        Target("exp977 phase5 Starling (second cooldown, 'sensible-starling')", "marin-community/marin", ["tootsie-8b-sensible-starling"], MARIN_KEYS, None,
               "steps 1.32M-1.40M at 4096x4096"),
        Target("marin-8b-base-deeper-starling (exp600_tootsie, aggregate)", "marin-community/marin", ["tootsie-8b-deeper-starling"], MARIN_KEYS, None,
               "released base = end of this run (step 1,419,999)", aliases=["exp600 Deeper Starling dessert (final released base)"]),
        Target("exp860 Tootsie 13B", "marin-community/marin", ["llama-13b-tootsie-ema-mk2"], MARIN_KEYS, None,
               "40L x 5120; longest of the mk1-mk3 attempts (367k steps); all crashed/abandoned"),
        Target("exp861 Tootsie 24B", "marin-community/marin", ["llama-22b-tootsie-ema-mk5"], MARIN_KEYS, None,
               "W&B calls it 22b (56L x 6144); longest attempt, 326k steps"),
        Target("exp750 Tootsie 70B", "marin-community/marin", ["llama-real-70b-tootsie"], MARIN_KEYS, None,
               "80L x 8192; 277k steps, abandoned"),
        Target("exp1295_32b phase1 (Llama-3-style 32B, spiky)", "marin-community/marin", ["llama-32b-tootsie-2"], MARIN_KEYS, 33_554_432,
               "display name 'Marin 32B run1'; steps 0-80,654"),
        Target("exp1390_32b_necro (optimizer-state rebuild restart)", "marin-community/marin", ["marin-32b-necro-2"], MARIN_KEYS, 33_554_432, ""),
        Target("exp1380_muon32b (Muon optimizer swap at 80k)", "marin-community/marin", ["marin-32b-muon-4"], MARIN_KEYS, 33_554_432,
               "4th Muon attempt, 86k steps"),
        Target("exp1395_qwen3_32b phase3 (QK-Norm switch, warm-start from 80k)", "marin-community/marin", ["marin-32b-qwen"], MARIN_KEYS, 33_554_432,
               "display name 'Marin 32B+qk-norm+warmstart'; steps 80k-170k"),
        Target("exp1529_32b_bison_cooldown (first 32B cooldown)", "marin-community/marin", ["tootsie-32b-cooldown-bison-adamc"], MARIN_KEYS, 33_554_432,
               "display name 'Marin 32B Bison Cooldown'; steps 160k-192k"),
        Target("marin-32b-base Mantis (aggregate final artifact)", "marin-community/marin", ["tootsie-32b-cooldown-mantis-adamc-v2"], MARIN_KEYS, 33_554_432,
               "display name 'Marin 32B Mantis Cooldown'; released base = end of this run (step 191,999)",
               aliases=["exp1529_32b_mantis_cooldown (released cooldown)"]),
        Target("tootsie-scaling-512 (Dec 2024 ladder, 16L, 210B tokens)", "marin-community/marin", ["tootsie-scaling-512-81c36c"], MARIN_KEYS, 4194304,
               "hidden 512, 16 layers, 50k steps; one of five ladder widths"),
        Target("tootsie-scaling-768 (Dec 2024 ladder, 16L, 210B tokens)", "marin-community/marin", ["tootsie-scaling-768-d17a90"], MARIN_KEYS, 4194304,
               "hidden 768, 16 layers, 50k steps; one of five ladder widths"),
        Target("tootsie-scaling-1024 (Dec 2024 ladder, 16L, 210B tokens)", "marin-community/marin", ["tootsie-scaling-1024-b45766"], MARIN_KEYS, 4194304,
               "hidden 1024, 16 layers, 50k steps; one of five ladder widths"),
        Target("tootsie-scaling-1536 (Dec 2024 ladder, 16L, 210B tokens)", "marin-community/marin", ["tootsie-scaling-1536-350a3a"], MARIN_KEYS, 4194304,
               "hidden 1536, 16 layers, 50k steps; one of five ladder widths"),
        Target("tootsie-scaling-2048 (Dec 2024 ladder, 16L, 210B tokens)", "marin-community/marin", ["tootsie-scaling-2048-1ed392"], MARIN_KEYS, 4194304,
               "hidden 2048, 16 layers, 50k steps; one of five ladder widths"),
    ],
}


# ----------------------------------------------------------------------------- GraphQL plumbing
def gql(query: str, variables: dict) -> dict:
    r = requests.post(URL, json={"query": query, "variables": variables}, timeout=180)
    r.raise_for_status()
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(j["errors"])
    return j["data"]


def _j(v):
    return json.loads(v) if isinstance(v, str) else (v or {})


RUNS_Q = """
query Runs($entity: String!, $project: String!, $cursor: String, $per: Int!, $filters: JSONString, $order: String) {
  project(name: $project, entityName: $entity) {
    runCount(filters: $filters)
    runs(first: $per, after: $cursor, filters: $filters, order: $order) {
      edges { node { name displayName state group createdAt heartbeatAt summaryMetrics historyKeys } }
      pageInfo { hasNextPage endCursor }
    }
  }
}"""

RUN_Q = """
query Run($entity: String!, $project: String!, $run: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) { name displayName state group createdAt heartbeatAt summaryMetrics config sampledHistory(specs: $specs) }
  }
}"""


def list_runs(path: str, filters: dict | None = None, maxn: int = 100, order: str = "-created_at") -> list[dict]:
    entity, project = path.split("/")
    cursor, out = None, []
    while len(out) < maxn:
        d = gql(RUNS_Q, {"entity": entity, "project": project, "cursor": cursor, "per": min(100, maxn - len(out)),
                         "filters": json.dumps(filters) if filters else None, "order": order})
        p = d["project"]
        if p is None:
            print(f"{path}: project not found / not public"); return []
        out += [e["node"] for e in p["runs"]["edges"]]
        pi = p["runs"]["pageInfo"]
        if not pi["hasNextPage"]:
            break
        cursor = pi["endCursor"]
    return out


def group_runs(path: str, group: str) -> list[str]:
    """Run names in a W&B group, oldest first (that is segment order for Ai2's restarts)."""
    runs = list_runs(path, {"group": group}, maxn=1000, order="+created_at")
    return [r["name"] for r in runs]


def fetch_run(path: str, run: str, keys: list[str], samples: int = SAMPLES) -> tuple[dict, list[dict]]:
    """sampledHistory returns only steps where *all* keys of a spec are present, so ask for one spec
    per key and merge on `_step` (eval keys are logged far more sparsely than train loss)."""
    entity, project = path.split("/")
    specs = [json.dumps({"keys": ["_step", k], "samples": samples}) for k in keys]
    d = gql(RUN_Q, {"entity": entity, "project": project, "run": run, "specs": specs})
    r = d["project"]["run"]
    if r is None:
        raise KeyError(f"{path}/{run} not found")
    merged: dict[float, dict] = {}
    for k, series in zip(keys, r["sampledHistory"] or []):
        for h in series or []:
            if h.get(k) is None or h.get("_step") is None:
                continue
            merged.setdefault(float(h["_step"]), {"_step": h["_step"]})[k] = h[k]
    return r, [merged[s] for s in sorted(merged)]


# ----------------------------------------------------------------------------- commands
def cmd_list(path: str, maxn: int, filters: dict | None) -> None:
    runs = list_runs(path, filters, maxn)
    print(f"{path}: showing {len(runs)}")
    for n in runs:
        s = _j(n["summaryMetrics"]); hk = _j(n["historyKeys"]).get("keys", {})
        losskeys = sorted(k for k in set(s) | set(hk) if "loss" in k.lower() and "downstream" not in k)
        print(f"  {n['name']:22} {n['displayName'][:60]!r:62} {n['state']:9} step={s.get('_step')} "
              f"group={n['group']!r} {n['createdAt'][:10]}..{(n['heartbeatAt'] or '')[:10]}")
        print(f"      loss keys: {losskeys[:10]}")


def fetch_target(lab: str, t: Target) -> dict | None:
    runs = t.runs
    if not runs:
        runs = group_runs(*GROUPS[t.run_id])
    segments, meta_runs = [], []
    for rid in runs:
        try:
            r, rows = fetch_run(t.project, rid, t.keys)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {t.project}/{rid}: {e}"); continue
        s = _j(r["summaryMetrics"])
        meta_runs.append({"run": rid, "display_name": r["displayName"], "state": r["state"],
                          "url": f"https://wandb.ai/{t.project}/runs/{rid}", "created": r["createdAt"],
                          "last_heartbeat": r["heartbeatAt"], "final_step": s.get("_step"),
                          "summary": {k: s.get(k) for k in t.keys if k in s}, "rows": len(rows)})
        for h in rows:
            h["_segment"] = rid
        segments.append(rows)
    rows = sorted((h for seg in segments for h in seg), key=lambda h: (float(h["_step"]), h["_segment"]))
    if not rows:
        print(f"  - {t.run_id}: none of {t.keys} logged"); return None
    present = [k for k in t.keys if any(h.get(k) is not None for h in rows)]
    rows = thin(rows, present)
    out = OUT / lab / (slug(t.run_id) + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["_step"] + present + ["_segment"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    final = {k: next((h[k] for h in reversed(rows) if h.get(k) is not None), None) for k in present}
    final_step = max(float(h["_step"]) for h in rows)
    meta = {"run_id": t.run_id, "aliases": t.aliases, "project": t.project, "keys": present,
            "tokens_per_step": t.tokens_per_step, "final_step": final_step,
            "final_tokens": t.tokens_per_step * final_step if t.tokens_per_step else None,
            "final_loss": final, "note": t.note, "segments": meta_runs, "rows": len(rows),
            "source": "W&B GraphQL, anonymous, sampledHistory"}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  + {t.run_id}: {len(rows)} rows from {len(meta_runs)} segment(s) -> {out.relative_to(ROOT)}  final={final}")
    return meta


def slug(run_id: str) -> str:
    """Filesystem-safe name for a runs.csv run_id."""
    import re
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", run_id).strip("_")


def thin(rows: list[dict], keys: list[str]) -> list[dict]:
    """Downsample to ~MAX_ROWS evenly over the index, keeping every row that carries an eval key
    (those are sparse) and the first/last row."""
    if len(rows) <= MAX_ROWS:
        return rows
    evalkeys = [k for k in keys if not k.startswith("train")]
    keep = {0, len(rows) - 1}
    keep.update(range(0, len(rows), max(1, len(rows) // MAX_ROWS)))
    keep.update(i for i, h in enumerate(rows) if any(h.get(k) is not None for k in evalkeys))
    return [rows[i] for i in sorted(keep)]


def cmd_fetch(labs: list[str]) -> None:
    for lab in labs:
        print(f"== {lab}")
        for t in TARGETS[lab]:
            fetch_target(lab, t)


# Which key becomes the runs.csv `loss`, per lab, and how to describe it. Train losses are
# smoothed (median of the last SMOOTH sampled points) because a single final batch is noisy;
# eval losses are taken as logged.
SMOOTH = 20
LOSS_COLUMN = {
    "ai2-olmo": (["train/CE loss", "train/CrossEntropyLoss"], True,
                 "train CE loss, nats/token (W&B; median of last 20 sampled points)"),
    "eleutherai": (["validation/lm_loss"], False, "Pile validation loss, nats/token (W&B validation/lm_loss)"),
    "marin": (["eval/paloma/c4_en/loss"], False, "Paloma c4_en eval loss, nats/token (W&B eval/paloma/c4_en/loss)"),
}


def final_loss(lab: str, csv_path: Path, meta: dict) -> tuple[float | None, str | None, str]:
    keys, smooth, label = LOSS_COLUMN[lab]
    key = next((k for k in keys if k in meta["keys"]), None)
    if key is None:
        return None, None, label
    with csv_path.open() as f:
        vals = [float(r[key]) for r in csv.DictReader(f) if r.get(key) not in (None, "")]
    if not vals:
        return None, None, label
    if smooth and len(vals) >= SMOOTH:
        tail = sorted(vals[-SMOOTH:]); v = tail[len(tail) // 2]
    else:
        v = vals[-1]
    return v, key, label


def cmd_apply() -> None:
    """Write final-step losses into data/runs.csv for every fetched curve (and its aliases)."""
    runs_csv = ROOT / "data" / "runs.csv"
    with runs_csv.open(newline="") as f:
        rd = csv.DictReader(f); rows = list(rd); fields = rd.fieldnames
    by_id: dict[str, list[dict]] = {}
    for r in rows:  # a run_id can appear on several rows (one per benchmark); write the loss to all
        by_id.setdefault(r["run_id"], []).append(r)
    n = 0
    for meta_path in sorted(OUT.glob("*/*.meta.json")):
        lab = meta_path.parent.name; m = json.loads(meta_path.read_text())
        v, key, label = final_loss(lab, meta_path.with_suffix("").with_suffix(".csv"), m)
        if v is None:
            print(f"  ? {lab}/{m['run_id']}: no {LOSS_COLUMN[lab][0]} in curve"); continue
        extra = ""
        c4 = m["final_loss"].get("eval/lm/c4_en-validation/CE loss")
        if c4 is not None:
            extra = f" c4_en eval CE {c4:.4f}."
        urls = "; ".join(seg["url"] for seg in m["segments"][-1:])
        for rid in [m["run_id"]] + m.get("aliases", []):
            hits = by_id.get(rid)
            if not hits:
                print(f"  ! {lab}/{rid}: not in runs.csv"); continue
            for r in hits:
                r["loss"] = f"{v:.4f}"; r["loss_eval_set"] = label  # row `confidence` is left alone: it
                # describes the whole row; the loss's own provenance is the W&B tag in `notes`.
                tag = f"Loss from W&B ({key}, final step {m['final_step']:.0f}, {urls}).{extra}"
                if "Loss from W&B" in r["notes"]:
                    import re
                    r["notes"] = re.sub(r"Loss from W&B \([^)]*\)\)?\.( c4_en eval CE [0-9.]+\.)?", tag, r["notes"])
                else:
                    r["notes"] = (r["notes"] + " " if r["notes"] else "") + tag
                n += 1
    with runs_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"updated loss on {n} rows of data/runs.csv")


def cmd_summary() -> None:
    for meta in sorted(OUT.glob("*/*.meta.json")):
        m = json.loads(meta.read_text())
        toks = f"{m['final_tokens']:.3g}" if m.get("final_tokens") else "?"
        print(f"{meta.parent.name:12} {m['run_id']:55} step={m['final_step']:>10.0f} tokens={toks:>8} {m['final_loss']}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        path, rest = args[1], args[2:]
        maxn, filters = 100, None
        while rest:
            if rest[0] == "--max": maxn, rest = int(rest[1]), rest[2:]
            elif rest[0] == "--filter": filters, rest = json.loads(rest[1]), rest[2:]
            else: sys.exit(f"unknown option {rest[0]}")
        cmd_list(path, maxn, filters)
    elif args[:1] == ["--summary"]:
        cmd_summary()
    elif args[:1] == ["--apply"]:
        cmd_apply()
    else:
        cmd_fetch(args or [lab for lab in TARGETS if TARGETS[lab]])
