# Next session

State as of 2026-09-13 (branch `claude/wandb-accessibility-check-kfemxy`):

- **Done:** final-run loss curves for OLMo 2/3, Pythia and Marin pulled from W&B into
  `data/losses/` and written into `data/runs.csv` (41 curves, 43 rows). `src/fetch_wandb.py`
  talks to W&B's GraphQL API **anonymously** with `requests`; a personal `WANDB_API_KEY` was
  refused for every public project, and PyPI was blocked, so neither the key nor the `wandb`
  package is needed. Chapters 10, 11, 12 and 90 were updated for the new numbers; `data/losses/README.md`
  has the metric choices and run mapping.
- **Rendered:** after the environment was switched to full network access, `quarto render`
  completed for all nine chapters (Quarto 1.7.32, pandas 3.0, scipy 1.17). The Pythia panel shows
  a fitted curve, the Marin ladder panel shows the five ladder points and the two released models,
  and the OLMo and Marin standard panels show final points without a law.

Environment setup: `.claude/hooks/session-start.sh` (registered in `.claude/settings.json`) installs
Quarto from GitHub and `requirements.txt` from PyPI at session start. In the 2026-09-13
environment PyPI was denied by the network policy, so the pip step printed a warning; allow
`pypi.org` and `files.pythonhosted.org` (and keep `github.com`) in the environment's network
settings and the next session will have a working `quarto render`.

Publishing: the book is deployed to GitHub Pages (https://tecunningham.github.io/open-labs/) by
`.github/workflows/publish.yml` on each push to the default branch; the one-time repo setting is
Settings -> Pages -> Source = "GitHub Actions". Check the Actions tab if the site is stale.

Rendering note: `_quarto.yml` sets `freeze: auto`, which re-executes a chapter only when its
`.qmd` changes. After editing the CSVs, delete `_freeze/` (or the chapter's subfolder) before
`quarto render`, or the figures will show stale data.

Remaining steps, in order:

1. (done) `quarto render` passes; eyeball `_book/` after any data change.
2. Still-missing losses: OLMo 2 32B is on comet.ml (`ai2/olmo-2-0325-32b`, REST API needs a
   free Comet key); the OLMo ladder W&B project is private, so ladder losses would need the
   `allenai/OLMo-ladder` checkpoints evaluated directly; Marin's Dec 2024 `tootsie-scaling-*`
   ladder (five sizes, in `marin-community/marin`) is public and could be added as ladder rows.
3. Marin phase rows carry per-phase `tokens`/`flops`; a loss-vs-cumulative-tokens figure built
   from the CSVs in `data/losses/marin/` would be more honest than the compute axis for them.
4. Verification pass done 2026-09-13 for all six notes files (item-by-item outcomes with
   sources are in each file). Refuted and corrected: DeepSeek V3/V3.2 author counts (199, 263),
   Olmo 3 base MMLU (66.9 / 76.2) and Think-32B MATH/MMLU (96.1 / 85.4), Delphi grid size (81),
   Snowball hardware (TPU v4-2048), 535B device count (704 GB200), OpenDiLoCo 1.1B run (44k
   steps), INTELLECT-3.1 (public model exists). Still open: Cerebras paper-vs-model-card Pile
   loss discrepancy (card returns 401 via proxy); OLMo 2 7B/13B GPU-hours; Prime Intellect
   headcount and funding; Marin 8B phase dates (private W&B project); the INTELLECT-3 SFT token
   conflict (217B in Table 1 vs ~50B from steps x tokens/step).
5. Hosts still blocked from this environment: raw.githubusercontent.com, api.github.com, PyPI,
   ai.meta.com, primeintellect.ai, cerebras.ai, x.com, web.archive.org, marin.readthedocs.io,
   storage.googleapis.com. `git clone` from github.com and huggingface.co API/raw work.
6. Never commit a key; none is needed for the fetcher.
