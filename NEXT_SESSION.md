# Next session

State as of 2026-09-13 (default branch `main`; work branch `claude/festive-dirac-j624jg`, kept
level with `main`):

- **Published.** The book is live at https://tecunningham.github.io/open-labs/. The default
  branch was renamed to `main`; `.github/workflows/publish.yml` renders and deploys on every push
  to it, and Settings -> Pages -> Source is "GitHub Actions" (the first deploy 404'd because it was
  still "Deploy from a branch"). Check the Actions tab if the site looks stale; a run takes about
  two minutes.
- **Losses.** Final-run loss curves for OLMo 2/3, Pythia and Marin are in `data/losses/` and
  `data/runs.csv` (see `data/losses/README.md`). `src/fetch_wandb.py` talks to W&B's GraphQL API
  anonymously with `requests`; no key or `wandb` package is needed, and a personal key is refused
  for public projects.
- **Marin trajectories.** `src/losses.py` converts W&B steps to cumulative tokens with the batch
  schedule (8B: 1024 -> 3072 -> 4096 sequences of 4096 at steps 0 / 660,600 / 1,320,000; 32B:
  8192 throughout), reproducing the model cards' phase totals. `plots.loss_trajectory` draws the
  phases and side branches as one curve; the Marin chapter has the figure, a per-phase table and
  a matched-token comparison of the abandoned 13B/24B/70B trials against the 8B trunk. The Dec
  2024 `tootsie-scaling-*` ladder is in `runs.csv` with losses and fitted in the chapter.
- **Data-availability table.** The preface (`index.qmd`) grades every lab on five things (total
  spend incl. people, pretraining curves, post-training curves, multiple generations, broad
  benchmarks, pre- vs post-training attribution) with evidence bullets; grades were compiled from
  `data/notes/*.md` on 2026-09-13. ECI (Epoch Capabilities Index) membership was checked only for
  Llama 3.1 405B and DeepSeek V2/V3/R1 via Epoch's own write-ups; epoch.ai itself was blocked.
  `styles.css` (wired in `_quarto.yml`) holds the `.availability` table style.
- **AI R&D benchmarks appendix** (`chapters/95-ai-rd-benchmarks.qmd`, `src/benchmarks.py`,
  `data/ai_rd_benchmarks.csv`, notes in `data/notes/ai-rd-benchmarks.md`). One benchmark-by-model
  table, one saturation overview (all bounded evals on a 0 to 100 percent axis) and one
  small-multiples time-series figure per closed lab (GDM, OpenAI, Anthropic); ceilings and human
  reference lines come from `data/ai_rd_benchmark_series.csv`; columns are the frontier
  cards through Sep 2026 (Gemini 2.5 Pro to 3.8 Flash, o1-preview to GPT-6 Astra, Claude 3.5 Sonnet
  to Fable 5.1). 396 rows; 371 read from the cards themselves after the network was opened
  mid-session, the rest are launch-post or blog figures no card contains. The notes file records
  every correction made against the cards and the cross-card inconsistencies (later cards re-run
  earlier models with different results). Open items: per-task RE-Bench scores for Gemini 2.5 Pro,
  2.5 Deep Think and 3 Pro are figure-only; GPT-5.6 and GPT-6 Astra report most self-improvement
  evals as curves without printed values; Anthropic's Sonnet 4 and Opus 4.1 launch-time
  Terminal-bench numbers are memory-only.
- **Rendered.** `quarto render` passes for all chapters (Quarto 1.7.32, pandas 3.0, scipy 1.17).

Environment setup: `.claude/hooks/session-start.sh` (registered in `.claude/settings.json`) installs
Quarto from GitHub and `requirements.txt` from PyPI at session start; both worked in the
2026-09-13 environment.

Rendering note: `_quarto.yml` sets `freeze: auto`, which re-executes a chapter only when its
`.qmd` changes. After editing the CSVs, delete `_freeze/` (or the chapter's subfolder) before
`quarto render`, or the figures will show stale data. The CI render starts clean every time.

Remaining steps, in rough order:

1. **DeepSeek** is still "in progress" in the README; check what chapter 14 lacks and close it out.
2. **Next lab chapter.** The README's proposed order had SmolLM3 and LLM360 K2 before DeepSeek and
   Llama; LLM360 K2 (W&B logs, 140 checkpoints, training-incident record) fits the fetcher and
   the new trajectory figure directly.
3. **Still-missing losses:** OLMo 2 32B is on comet.ml (`ai2/olmo-2-0325-32b`; the REST API needs
   a free Comet key); the OLMo ladder W&B project is private, so ladder losses would need the
   `allenai/OLMo-ladder` checkpoints evaluated directly.
4. **Trajectories for OLMo.** The OLMo 2/3 stage rows have the same per-phase problem as Marin;
   a schedule in `src/losses.py` (tokens per step are in the `.meta.json` files) would give the
   OLMo chapter the same figure.
5. Open verification items (details in `data/notes/*.md`): Cerebras paper-vs-model-card Pile loss
   discrepancy; OLMo 2 7B/13B GPU-hours; Prime Intellect headcount and funding; Marin 8B phase
   dates (private W&B project); the INTELLECT-3 SFT token conflict (217B in Table 1 vs ~50B from
   steps x tokens/step).
6. Hosts that were blocked from the collection environment: raw.githubusercontent.com,
   api.github.com, ai.meta.com, primeintellect.ai, cerebras.ai, x.com, web.archive.org,
   marin.readthedocs.io, storage.googleapis.com, and github.io (so the published site cannot be
   fetched from inside a session). `git clone` from github.com, PyPI, W&B and the huggingface.co
   API/raw work.
7. Never commit a key; none is needed for the fetcher.
