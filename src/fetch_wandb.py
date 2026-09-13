"""Pull loss curves from public W&B projects into data/losses/<lab>/<run>.csv.

Needs a W&B API key (any free account works; public projects are readable by anyone logged in):
    export WANDB_API_KEY=...        # or `wandb login`
    python -m src.fetch_wandb                 # all targets below
    python -m src.fetch_wandb ai2-olmo        # one lab
    python -m src.fetch_wandb --list ai2-llm/Olmo-3-1025-7B   # discover run names and metric keys

Never commit the key. The CSVs are small (one row per logged step, downsampled to `samples`).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "losses"

# lab -> list of (entity/project, run filter or None, metric keys). Metric names differ per lab;
# use --list to discover them, then pin the right ones here.
TARGETS = {
    "ai2-olmo": [
        ("ai2-llm/Olmo-3-1025-7B", None, ["train/CE loss", "train/loss", "eval/c4_en-validation/CrossEntropyLoss"]),
        ("ai2-llm/Olmo-3-1125-32B", None, ["train/CE loss", "train/loss", "eval/c4_en-validation/CrossEntropyLoss"]),
        ("ai2-llm/OLMo-2-1124-7B", None, ["train/CE loss", "train/loss"]),
        ("ai2-llm/OLMo-2-1124-13B", None, ["train/CE loss", "train/loss"]),
    ],
    "eleutherai": [
        ("eleutherai/pythia", None, ["train/lm_loss", "validation/lm_loss", "train_loss", "val_loss"]),
    ],
    "marin": [
        # Marin's W&B links are per experiment issue; fill entity/project from an issue's W&B URL.
        ("marin-community/marin", None, ["train/loss", "eval/loss", "eval/paloma/c4_en/loss"]),
    ],
}


def api():
    import wandb
    if not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
        sys.exit("Set WANDB_API_KEY (any free W&B account) or run `wandb login` first.")
    return wandb.Api(timeout=60)


def list_project(path: str) -> None:
    a = api()
    runs = list(a.runs(path))
    print(f"{path}: {len(runs)} runs")
    for r in runs[:50]:
        keys = sorted(k for k in r.summary.keys() if "loss" in k.lower())[:8]
        print(f"  {r.id}  {r.name!r:50}  state={r.state}  steps={r.summary.get('_step')}  loss keys={keys}")


def fetch(lab: str, samples: int = 2000) -> None:
    a = api()
    for project, name_filter, keys in TARGETS[lab]:
        try:
            runs = list(a.runs(project, filters={"display_name": {"$regex": name_filter}} if name_filter else None))
        except Exception as e:  # noqa: BLE001
            print(f"  ! {project}: {e}"); continue
        print(f"{project}: {len(runs)} runs")
        for r in runs:
            present = [k for k in keys if k in r.summary or k in getattr(r, "history_keys", {}).get("keys", {})]
            hist = r.history(keys=["_step"] + (present or keys), samples=samples, pandas=False)
            rows = [h for h in hist if any(h.get(k) is not None for k in keys)]
            if not rows:
                print(f"  - {r.name}: none of {keys} logged; use --list to find the right key"); continue
            out = OUT / lab / f"{r.name.replace('/', '_')}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
            cols = ["_step"] + [k for k in keys if any(k in h for h in rows)]
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
            meta = {"run_id": r.id, "name": r.name, "project": project, "url": r.url, "created": str(r.created_at),
                    "config_tokens_per_batch": r.config.get("global_train_batch_size") or r.config.get("train_batch_size")}
            (out.with_suffix(".meta.json")).write_text(__import__("json").dumps(meta, indent=1))
            print(f"  + {r.name}: {len(rows)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        list_project(args[1])
    else:
        for lab in (args or TARGETS):
            fetch(lab)
