# Next session

State as of 2026-09-13 (branch `claude/wandb-accessibility-check-kfemxy`):

- **Done:** final-run loss curves for OLMo 2/3, Pythia and Marin pulled from W&B into
  `data/losses/` and written into `data/runs.csv` (41 curves, 43 rows). `src/fetch_wandb.py`
  talks to W&B's GraphQL API **anonymously** with `requests`; a personal `WANDB_API_KEY` was
  refused for every public project, and PyPI was blocked, so neither the key nor the `wandb`
  package is needed. Chapters 10, 11, 12 and 90 were updated for the new numbers; `data/losses/README.md`
  has the metric choices and run mapping.
- **Not verified here:** the Quarto render. This container had no `quarto`, `pandas` or `scipy`,
  so the edited chapters (a new Pythia loss figure and overlay in `10-reference-suites.qmd`; the
  `fit_experiments=False` switch added to `src/chapter.py` and used in `11-marin.qmd` and
  `12-olmo.qmd`) are unrun. Do this first: `quarto render`, and check that the Pythia panel shows
  a fitted curve and that the OLMo and Marin panels show final-run points without a law.

Remaining steps, in order:

1. `quarto render` and fix anything the new chapter code breaks.
2. Still-missing losses: OLMo 2 32B is on comet.ml (`ai2/olmo-2-0325-32b`, REST API needs a
   free Comet key); the OLMo ladder W&B project is private, so ladder losses would need the
   `allenai/OLMo-ladder` checkpoints evaluated directly; Marin's Dec 2024 `tootsie-scaling-*`
   ladder (five sizes, in `marin-community/marin`) is public and could be added as ladder rows.
3. Marin phase rows carry per-phase `tokens`/`flops`; a loss-vs-cumulative-tokens figure built
   from the CSVs in `data/losses/marin/` would be more honest than the compute axis for them.
4. Work through the "Verification status" lists in `data/notes/*.md` against arXiv (reachable in
   this environment), flipping `confidence` from `guess` to `reported` where confirmed.
5. Commit the CSVs. Never commit a key; none is needed for the fetcher.
