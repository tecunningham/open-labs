"""Read the per-run loss curves in data/losses/ and put them on a cumulative-token axis.

`runs.csv` gives each continuation phase of a long run its *own* tokens and FLOPs, so a loss
plotted against that row's compute is misleading: the loss at the end of phase 4 reflects every
token since step 0. The helpers here convert W&B step numbers into cumulative training tokens
using the run's batch-size schedule, so a whole multi-phase run can be drawn as one trajectory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOSSES = ROOT / "data" / "losses"


def slug(run_id: str) -> str:
    """Filesystem-safe name for a runs.csv run_id (same rule as src.fetch_wandb)."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", run_id).strip("_")


def curve(lab: str, run_id: str) -> pd.DataFrame:
    """The fetched loss history for one runs.csv row, with its .meta.json in `df.attrs`."""
    base = LOSSES / lab / slug(run_id)   # slugs may contain dots, so do not use with_suffix
    df = pd.read_csv(str(base) + ".csv")
    df.attrs = json.loads(Path(str(base) + ".meta.json").read_text())
    return df


# --------------------------------------------------------------------------- step -> tokens

# A schedule is a list of (first_step, tokens_per_step): each entry holds from its first step up
# to the next entry's first step. Steps are the trainer's global step, which Levanter keeps
# counting across phase restarts, so one schedule covers a whole multi-phase run.
Schedule = list[tuple[int, int]]

# Marin 8B "Tootsie": 4096-token sequences; 1024 per step on the v5e slices (Kestrel), 3072 from the
# move to the v4-2048 (Ocelot through Phoenix), 4096 for the Starling cooldowns. Reproduces the
# model card's phase totals: 2.7T / 3.78T / 4.78T / 11.1T / 12.4T / 12.7T. The 3072 -> 4096 change
# is placed at step 1,320,000 (Phoenix ran to 1,321,398 at 3072, Starling starts at 1,320,061), a
# difference of about 0.02T.
MARIN_8B: Schedule = [(0, 1024 * 4096), (660_600, 3072 * 4096), (1_320_000, 4096 * 4096)]
# Marin 32B "Mantis": 8192 x 4096 throughout (the model card's 2.679T / 2.684T / 1.074T phases are
# 80k / 80k / 32k steps at this size).
MARIN_32B: Schedule = [(0, 8192 * 4096)]


def cumulative_tokens(step, schedule: Schedule) -> np.ndarray:
    """Tokens seen by global `step` under a piecewise-constant batch schedule."""
    s = np.asarray(step, dtype=float)
    out = np.zeros_like(s)
    for i, (start, tps) in enumerate(schedule):
        end = schedule[i + 1][0] if i + 1 < len(schedule) else np.inf
        out += tps * np.clip(np.minimum(s, end) - start, 0, None)
    return out


# --------------------------------------------------------------------------- trajectories

@dataclass
class Segment:
    """One drawn piece of a trajectory: a phase, cooldown or side branch of a long run."""
    run_id: str
    label: str
    run_type: str
    tokens: np.ndarray
    loss: np.ndarray
    released: bool = False   # the released checkpoint is this segment's last point

    @property
    def trunk(self) -> bool:
        """Final and mid-training phases form the released model's own path; the rest are branches."""
        return self.run_type in ("final", "midtrain")


def segments(lab: str, spec: list[tuple[str, str, str]], schedule: Schedule, key: str,
             released: str | None = None) -> list[Segment]:
    """Build Segments from (run_id, short label, run_type) triples, keeping only points where `key`
    was logged (eval losses are sparse; train loss is on the training mix and not comparable)."""
    out = []
    for run_id, label, run_type in spec:
        c = curve(lab, run_id).dropna(subset=[key]).sort_values("_step")
        out.append(Segment(run_id, label, run_type, cumulative_tokens(c["_step"], schedule), c[key].to_numpy(),
                           released=(run_id == released)))
    return out


def trunk_curve(segs: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    """The released model's own path (final + midtrain segments) as one token-sorted curve."""
    t = [s for s in segs if s.trunk]
    x = np.concatenate([s.tokens for s in t]); y = np.concatenate([s.loss for s in t])
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


def loss_at(segs: list[Segment], tokens) -> np.ndarray:
    """Trunk loss interpolated at the given cumulative token counts."""
    x, y = trunk_curve(segs)
    return np.interp(np.asarray(tokens, dtype=float), x, y)


def phase_table(segs: list[Segment], min_tokens: float = 0.0) -> pd.DataFrame:
    """Start/end tokens and loss per segment; the start is the first point at or after `min_tokens`
    so the warm-up value at step 0 does not appear as a phase's starting loss."""
    rows = []
    for s in segs:
        k = s.tokens >= min_tokens
        if k.sum() == 0:
            continue
        x, y = s.tokens[k], s.loss[k]
        rows.append({"phase": s.label or s.run_id, "run_type": s.run_type,
                     "tokens_start_T": x[0] / 1e12, "tokens_end_T": x[-1] / 1e12,
                     "loss_start": y[0], "loss_end": y[-1], "delta": y[-1] - y[0]})
    return pd.DataFrame(rows)


# Marin's phase structure, as runs.csv names the rows. Order is drawing order; the trunk phases
# come first so side branches sit on top of them.
MARIN_8B_SPEC = [
    ("exp600 phase1 Kestrel (DCLM, WSD-S)", "Kestrel", "final"),
    ("exp600 phase2 Ocelot (DCLM, WSD+EMA)", "Ocelot", "final"),
    ("exp600 phase3 Jellyfish (first cooldown, 'monumental-jellyfish')", "Jellyfish (cooldown 1)", "midtrain"),
    ("exp600 phase4 Phoenix (rewarm, Nemotron-CC)", "Phoenix (rewarm)", "final"),
    ("exp977 phase5 Starling (second cooldown, 'sensible-starling')", "Starling (cooldown 2)", "midtrain"),
    ("marin-8b-base-deeper-starling (exp600_tootsie, aggregate)", "Deeper Starling", "final"),
    # Three short cooldown branches off the end of Jellyfish; one shared label (blank = unlabelled).
    ("exp600 dessert runs (math + FLAN patches off Jellyfish)", "dessert, Raccoon, Spoonbill (off Jellyfish)", "ablation"),
    ("exp898 Raccoon (deep cooldown 1.7e-4->1.7e-5 from Jellyfish)", "", "ablation"),
    ("exp916 Spoonbill (cooldown to 3e-5 + Tulu 0.3% + FLAN 1%, z-loss 1e-4 fix)", "", "ablation"),
]
MARIN_8B_RELEASED = "marin-8b-base-deeper-starling (exp600_tootsie, aggregate)"

MARIN_32B_SPEC = [
    ("exp1295_32b phase1 (Llama-3-style 32B, spiky)", "Phase 1 (Llama-3-style)", "final"),
    ("exp1395_qwen3_32b phase3 (QK-Norm switch, warm-start from 80k)", "Phase 3 (QK-Norm)", "final"),
    ("marin-32b-base Mantis (aggregate final artifact)", "Mantis cooldown", "midtrain"),
    ("exp1390_32b_necro (optimizer-state rebuild restart)", "", "aborted"),
    ("exp1380_muon32b (Muon optimizer swap at 80k)", "Muon and necro rescues (discarded)", "aborted"),
    ("exp1529_32b_bison_cooldown (first 32B cooldown)", "Bison (discarded)", "aborted"),
]
MARIN_32B_RELEASED = "marin-32b-base Mantis (aggregate final artifact)"
MARIN_KEY = "eval/paloma/c4_en/loss"


def marin_8b() -> list[Segment]:
    return segments("marin", MARIN_8B_SPEC, MARIN_8B, MARIN_KEY, released=MARIN_8B_RELEASED)


def marin_32b() -> list[Segment]:
    return segments("marin", MARIN_32B_SPEC, MARIN_32B, MARIN_KEY, released=MARIN_32B_RELEASED)


# --------------------------------------------------------------------------- whole runs on one axis

@dataclass
class Curve:
    """One pretraining run (possibly several phases stitched together) as a single loss curve,
    with the parameter count needed to put it on a compute axis (6 * params * tokens)."""
    name: str
    kind: str          # 'ladder', 'final', 'aborted'
    params: float
    tokens: np.ndarray
    loss: np.ndarray

    @property
    def flops(self) -> np.ndarray:
        return 6.0 * self.params * self.tokens


def stitch(lab: str, run_ids: list[str], schedule: Schedule, key: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate several runs.csv rows (phases of one run) into one token-sorted curve. `key` may
    be a list of alternative column names (a metric renamed between segments); the first non-null
    value per row is used."""
    keys = [key] if isinstance(key, str) else key
    xs, ys = [], []
    for rid in run_ids:
        c = curve(lab, rid)
        present = [k for k in keys if k in c.columns]
        if not present:
            continue
        v = c[present].bfill(axis=1).iloc[:, 0]
        c = c.assign(_v=v).dropna(subset=["_v"]).sort_values("_step")
        xs.append(cumulative_tokens(c["_step"], schedule)); ys.append(c["_v"].to_numpy())
    x = np.concatenate(xs); y = np.concatenate(ys)
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


# The abandoned Tootsie trials ramped their batch size and the exact switch steps are not
# public, so their step -> token conversion uses the average tokens per step implied by the W&B
# `throughput/total_tokens` counter at the last logged step (data/notes/marin.md, item 9).
_AVG_TPS = {"13B": 2.271e12 / 367_148, "24B": 2.431e12 / 326_496, "70B": 1.542e12 / 277_173}

MARIN_PRETRAINING = [
    # (name, kind, params, run_ids in trunk order, schedule)
    *[(f"ladder d{w}", "ladder", p, [f"tootsie-scaling-{w} (Dec 2024 ladder, 16L, 210B tokens)"], [(0, 4_194_304)])
      for w, p in [(512, 2.49e8), (768, 4.61e8), (1024, 7.32e8), (1536, 1.45e9), (2048, 2.34e9)]],
    ("Marin 8B (Tootsie)", "final", 8.03e9,
     [s[0] for s in MARIN_8B_SPEC if s[2] in ("final", "midtrain")], MARIN_8B),
    ("13B trial", "aborted", 1.3e10, ["exp860 Tootsie 13B"], [(0, _AVG_TPS["13B"])]),
    ("24B trial", "aborted", 2.4e10, ["exp861 Tootsie 24B"], [(0, _AVG_TPS["24B"])]),
    ("70B trial", "aborted", 7.0e10, ["exp750 Tootsie 70B"], [(0, _AVG_TPS["70B"])]),
    ("Marin 32B (Mantis)", "final", 3.2e10,
     ["exp1295_32b phase1 (Llama-3-style 32B, spiky)", "exp1395_qwen3_32b phase3 (QK-Norm switch, warm-start from 80k)",
      "marin-32b-base Mantis (aggregate final artifact)"], MARIN_32B),
    # 2026: Delphi's held-out dense targets (compute-optimal, AdamH recipe) and the MoE runs. MoE
    # compute uses active parameters (about 2B of 67B for Snowball; 23B of 535B for the hero run).
    ("Delphi 1e21 (3.4B)", "delphi", 3.383e9, ["Delphi optimal 1e21 (3 seeds: 0, 42, 62746)"], [(0, 2_097_152)]),
    ("Delphi 1e22 (9.7B)", "delphi", 9.715e9, ["delphi-1e22-9.7Bparams-160Btokens (3 seeds)"], [(0, 4_194_304)]),
    ("Delphi 1e23 (25B)", "delphi", 2.496e10, ["Delphi optimal 1e23 (25B params, 600B tokens)"], [(0, 8_388_608)]),
    ("Snowball 67B-A2B", "moe", 2.0e9, ["Snowball 67B-A2B pretrain+midtrain on 10T tokens (#6044)"],
     [(0, 4096 * 8192), (15_288, 8192 * 8192)]),
    ("535B-A23B hero (running)", "moe", 2.3e10, ["[Hero Run] 535B-A23B on 18.75T tokens (#8435)"], [(0, 46_137_344)]),
]
# The hero run's later segments re-log the evaluation from step 0 under `eval_dropless/...`
# (routing without capacity-factor token dropping), which spans the whole run; the first segment's
# `eval/...` values are a different evaluation mode and are not mixed in.
MARIN_CURVE_KEYS = {"535B-A23B hero (running)": ["eval_dropless/paloma/c4_en-llama3/loss"]}


# Start dates of the runs above (W&B run creation), for efficiency-over-time views.
MARIN_START = {"ladder d512": "2024-12-16", "ladder d768": "2024-12-16", "ladder d1024": "2024-12-16", "ladder d1536": "2024-12-16",
               "ladder d2048": "2024-12-16", "Marin 8B (Tootsie)": "2024-11-28", "13B trial": "2025-02-14", "24B trial": "2025-02-14",
               "70B trial": "2025-01-31", "Marin 32B (Mantis)": "2025-04-24", "Delphi 1e21 (3.4B)": "2026-03-04",
               "Delphi 1e22 (9.7B)": "2026-03-04", "Delphi 1e23 (25B)": "2026-03-04", "Snowball 67B-A2B": "2026-06-27",
               "535B-A23B hero (running)": "2026-08-20"}


def compute_to_reach(c: Curve, target: float, min_tokens: float = 2e10) -> float:
    """Cumulative 6ND compute at which a run's loss first falls to `target`, interpolated; NaN if
    it never does. Points before `min_tokens` are ignored so warm-up noise does not count."""
    k = c.tokens >= min_tokens
    x, y = c.flops[k], c.loss[k]
    below = np.where(y <= target)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(x[0])
    f = (y[i - 1] - target) / (y[i - 1] - y[i])
    return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1]))))


def marin_pretraining_curves(include_2026: bool = True) -> list[Curve]:
    """Every Marin pretraining run with a public c4_en curve, as one curve each. The 32B trunk
    stops at the cooldown branch point (phase 3 ran 10k steps past it); the 13B curve starts at
    step 280k because only its longest restart segment was fetched. Runs whose curve has not been
    fetched are skipped."""
    out = []
    for name, kind, params, run_ids, schedule in MARIN_PRETRAINING:
        if not include_2026 and kind in ("delphi", "moe"):
            continue
        try:
            x, y = stitch("marin", run_ids, schedule, MARIN_CURVE_KEYS.get(name, MARIN_KEY))
        except FileNotFoundError:
            continue
        if name.startswith("Marin 32B"):
            cut = cumulative_tokens(160_000, schedule)   # drop phase-3 points past the cooldown fork
            keep = (x <= cut) | (x >= cumulative_tokens(160_002, schedule) - 1)
            x, y = x[keep], y[keep]
        out.append(Curve(name, kind, params, x, y))
    return out
