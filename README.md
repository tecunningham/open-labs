# Open Labs: scaling curves, experiments, and final runs

**Status (2026-09-13):** scope agreed; Quarto book scaffolded and rendering; data and chapters in
for Marin, Ai2 OLMo, Meta Llama, Prime Intellect (RL pilot), and the Pythia / Cerebras-GPT
reference suites; DeepSeek in progress. Render with `quarto render`; merge new data with
`python -m src.merge <dir>`. Agent working notes with per-claim verification lists are in
`data/notes/`.

**Environment caveat.** The collection environment's network policy blocked arxiv.org,
huggingface.co, most lab blogs, W&B and Semantic Scholar. Paper numbers were taken from GitHub
mirrors, tech-report PDFs on other hosts, and search snippets, and every such row is flagged in
its `notes`. A follow-up session (2026-09-13, W&B and arXiv allowed) pulled the final-run loss
curves for OLMo, Marin and Pythia from W&B into `data/losses/` and `runs.csv` (see
`data/losses/README.md`; the W&B API is used anonymously because a personal key was refused for
public projects). Still missing: OLMo 2 32B (comet.ml), the OLMo ladder (private W&B project),
and losses for the ablation rows.

## 1. What this is

A data-driven report on how *open* AI labs turn experiments into final training runs. Each
lab gets a chapter with the same set of figures and tables, built from the lab's own
published training records (tech reports, W&B logs, GitHub issues, blog ablations). A final
chapter compares across labs.

The core object in every chapter is the **scaling curve with experiments and final runs**:
training compute on the x-axis (log), loss or a benchmark on the y-axis, every experimental
run plotted as a point, the released model(s) plotted as the final run(s), and a power-law fit
through the experiments extrapolated out to where the final run landed.

## 2. Why (link to the elasticity / RSI paper)

The elasticity paper ("The Economics of Recursive Self-Improvement", Section 3) needs three
things that frontier labs will not share but open labs already publish, in pieces:

| Paper primitive | What open-lab training records give us |
|---|---|
| Share of experimental compute, S_E (You 2025: ~90% experiments / 10% final at OpenAI; Morrison et al. 2026 for OLMo 3) | Directly: sum compute over ladder/ablation runs vs. the final run, per lab and per release |
| Growth in algorithmic efficiency, g_A, and whether it is scale-biased (Gundlach et al. 2025) | Curve shift between successive releases of the same lab (OLMo 1→2→3, SmolLM 1→2→3, DeepSeek V2→V3), measured at several scales |
| Returns to research effort, and labor vs. compute spend shares | Team size and project duration (published for most open labs) against total compute |

Nate's point in #core-capabilities-stream (2026-07-24) applies here too: spend numbers on runs
carry maybe half the information of a full scaling curve at a tenth of the effort. For open
labs we can often get both, so the report should always show the cheap number (compute split)
even where the full curve is thin.

Two further things only the full curve gives:

- **How informative experiments are.** The residual between the extrapolated fit and the
  actual final run tells us how well small-scale experiments predicted the big run. GPT-4's
  tech report is the canonical closed-lab version of this figure; we can produce it for open labs.
- **Marginal return to compute at the frontier ("terminal elasticity").** The local slope of the
  curve at the final run.

## 3. Questions each chapter answers

1. How much compute went into experiments versus the final run(s)? Counts, sizes, and total
   FLOPs / GPU-hours / dollars for each.
2. What did the experimental ladder look like, and where did the final run land relative to the
   extrapolation from experiments?
3. What is the local slope (elasticity of loss or benchmark with respect to compute) at the
   final run?
4. Across the lab's releases, how far did the curve shift, and did it shift more at large scale
   than small?
5. What were the experiments *for* (data mix, architecture, optimizer, hyperparameters, schedule),
   and how many were aborted or failed?
6. How big was the team and how long did the project take, from first experiment to release?

## 4. Which labs, and in what order

"Open lab" here means: releases weights *and* enough about the training process to reconstruct
at least the final-run compute and some experimental runs. Tiered by how much of the
experiments→final-run record is public.

**Tier A: experiment logs are public.** These are the chapters that can be done properly.

| Lab / project | What is public | Why it matters for us |
|---|---|---|
| Marin (Stanford CRFM) | Every experiment is a GitHub issue with W&B links; Marin 8B and 32B final runs | The only lab whose entire experiment record is browsable. Best pilot chapter. |
| Ai2 OLMo (OLMo 2, OLMo 3, Olmo Hybrid) | Data, code, recipes, W&B logs, thousands of intermediate checkpoints; model-ladder paper (Bhagia et al. 2024); Olmo Hybrid (2026) fits scaling laws to predict how token-efficiency changes with size | Best overall source today. Already cited in the paper via Morrison et al. 2026 for the experiment/final compute ratio. Multiple generations for the curve-shift question. Tülu covers post-training. |
| LLM360 (Amber, CrystalCoder, K2-65B) | Exact data sequence per checkpoint, W&B and system logs, 140 intermediate checkpoints for K2, records of training incidents (loss spikes and fixes) | Closest thing to a lab notebook for one large run. Forensic view of the final run rather than of the experiment ladder. |
| EleutherAI Pythia (and GPT-NeoX) | 70M to 12B, same 300B tokens in the same order, 154 checkpoints per model; models, data, and code released | Cleanest controlled size/compute scaling dataset. No experiments-vs-final split, so it is a reference curve, not a lab chapter. |
| Cerebras-GPT | Seven model sizes trained as a deliberately compute-optimal family; fitted scaling law published with weights and methodology | Literal Chinchilla-style curve on a public dataset. Cheap chapter; pairs with Pythia. |
| Prime Intellect (INTELLECT-1, INTELLECT-2) | INTELLECT-2: code, data, reward trajectories and ablations from a 32B asynchronous RL run, including failures (gradient instability, length-reward effects, filtering); agentic-RL environment scaling | Best open record of post-training / RL experiments. Anchors the RL part of the book. |
| Hugging Face SmolLM (2, 3) | Ablation writeups, training configs, FineWeb/FineMath data ablations; 125M model trained at several token budgets to pick how far past Chinchilla-optimal to go; SmolLM3 = 3B on 11T tokens, 384 H100s for 24 days | Very detailed data- and token-budget ablations at small scale. |
| Apertus (Swiss AI Initiative: ETH, EPFL, CSCS) | 8B and 70B, full data recipe, tech report | Public-compute lab; likely publishes GPU-hours cleanly. |
| BigScience BLOOM | Training chronicles and engineering logs, architecture and hyperparameter experiments, 1.08M A100-hours | Historical (2022) but unusually candid about failed runs. |
| modded-nanogpt speedrun | Every record is a logged run | A miniature lab with a complete experiment record. Ties to METR's own NanoGPT / expenditure-horizon work. |

**Tier B: open weights plus a tech report with scaling-law experiments, but no raw logs.**
Chapters here will lean on the report's figures and stated GPU-hours.

DeepSeek (DeepSeek LLM scaling-law paper; V3: 2.788M H800-hours, small-scale ablations for
MTP and FP8), Meta Llama 3 (IsoFLOP experiments from 6e18 to 1e22 FLOPs to pick 405B / 15.6T
tokens; 39.3M GPU-hours), MiniCPM ("wind tunnel" experiments, WSD schedule), Moonshot Kimi K2
(Muon scaling experiments), Qwen 2.5 / 3 (thin on pretraining, but the GSPO work publishes RL reward
and performance against training compute on a shared cold-start model), NVIDIA Nemotron,
TII Falcon, European public-compute models (EuroLLM, Salamandra, Teuken, Poro).

**Tier C: closed labs with one published experiments→final figure.** Reference points only,
not chapters: GPT-4 tech report (loss predicted from runs at 1/1,000 to 1/10,000 of compute),
Chinchilla (400+ runs), GPT-3, Anthropic scaling-laws papers.

**Two parts, not one.** The sources split cleanly by training stage, so the book should too:

- *Part I, pretraining scaling:* Pythia and Cerebras-GPT as clean reference curves, then the lab
  chapters (Marin, OLMo, SmolLM, LLM360, Apertus, DeepSeek, Llama 3, BLOOM, modded-nanogpt).
- *Part II, post-training / RL scaling:* OLMo / Tülu, Prime Intellect, Qwen (GSPO), DeepSeek R1.
  Same chapter template, but the x-axis is RL compute or environment steps and the y-axis is
  reward or a benchmark, and "final run" means the released post-trained model.

**Proposed order.** Pilot with Marin and OLMo 3 (richest records, and they exercise every
figure). Pythia and Cerebras-GPT next because they are nearly free and give the reference
curves. Then SmolLM3, LLM360 K2, DeepSeek V3, Llama 3, Apertus, modded-nanogpt. Prime Intellect
opens Part II. Decide on the rest of Tier B after seeing how thin the reports are.

## 5. Data model

Hand-curated CSVs with a source URL on every row. No scraping pipeline in the first pass.

`data/runs.csv`, one row per training run:

| column | notes |
|---|---|
| lab, project, run_id | project = release family (e.g. `olmo3`) |
| run_type | `ladder`, `ablation`, `final`, `midtrain`, `aborted`, `unknown` |
| params_total, params_active | active for MoE |
| tokens | |
| flops | reported, else 6·N_active·D, flagged as estimate |
| gpu_hours, hardware | as reported |
| cost_usd_est | our estimate, with the price assumption recorded |
| date_start, date_end | when known |
| loss, loss_eval_set | which held-out set; not comparable across labs |
| benchmarks (wide or long) | for cross-lab comparison |
| purpose | free text: what the experiment tested |
| source_url, confidence | `reported` / `derived` / `guess` |

`data/experiments.csv`, one row per *published scaling experiment or figure* (a ladder, an
ablation series, an RL curve), separate from individual runs:

| column | notes |
|---|---|
| lab, project, date | |
| stage | `pretraining`, `midtraining`, `post-training/RL` |
| intervention | what was varied: size, tokens, data mix, optimizer, schedule, RL algorithm |
| model_sizes, compute_range | as reported |
| metric | loss, benchmark, reward |
| fitted_exponent | the lab's own fit if given, ours otherwise, flagged |
| raw_data_available | `logs`, `figure_only`, `numbers_in_text` |
| source_url | |

This is the "matrix of every publicly available scaling curve" and is probably the single most
reusable output of the project.

`data/labs.csv`: headcount, project start and release dates, total compute, funding type,
sources. `data/claims.csv`: verbatim quantitative statements from the lab ("experiments were
~X% of compute") with citation, so stated and reconstructed numbers can be compared.

## 6. Chapter template

Every lab chapter has the same figures so the cross-lab chapter can overlay them.

1. **Scaling curve.** Log compute vs. loss (and a second panel for a benchmark). Experiment
   runs coloured by `run_type`; final run(s) marked distinctly; power-law fit through
   experiments with an extrapolation band; the final run's residual annotated.
2. **Compute budget.** Experiments vs. final as a stacked bar, plus a histogram of experimental
   run sizes and the run count.
3. **Timeline.** Runs over calendar time, sized by compute, showing iteration cadence and the
   gap between the last experiment and the final run.
4. **Across releases** (where the lab has several generations). Curves overlaid; implied
   compute multiplier per year at two or three fixed scales.
5. **Key-numbers table** with provenance and confidence per cell.
6. **Narrative**: what the lab said they learned from experiments, and what we can and cannot
   reconstruct.

Cross-lab chapter: all final runs and ladders on one chart; experiment share by lab; terminal
slope vs. scale; prediction residual vs. experiment share; the three paper primitives from
Section 2 tabulated per lab.

## 7. Known methodological problems

- **Loss is not comparable across labs** (tokenizer, data, eval set). Within-lab curves use
  loss; cross-lab uses benchmarks or per-lab normalisation. Say which every time.
- **What counts as an experiment.** Ladder runs, ablations, aborted runs, and mid-training /
  annealing restarts all differ. Post-training and RL experiments are a separate budget; the
  first pass is pretraining only, with a flag for where a lab's number bundles them.
- **Survivorship.** Public logs under-report failed and aborted runs. The compute split is
  therefore a lower bound on experiments unless the lab states a total.
- **Which run is "final".** Families release several sizes (1B / 7B / 32B). Treat each released
  size as a final run; ladder runs that were not released are experiments.
- **Compute accounting.** 6ND vs. reported GPU-hours can differ by 2x once MFU, MoE, and
  restarts are included. Store both; plot the reported one when available.
- **Dollars.** Cloud list prices vs. owned hardware. Record the assumption per row.
- **Secondhand claims.** The lab descriptions in Section 4 come from memory and from a ChatGPT
  summary Tom supplied (2026-09-13). Every factual claim there (checkpoint counts, GPU-hours,
  "first scaling law on a public dataset") gets verified against the primary source before it
  appears in a chapter.

## 8. Deliverable and repo layout

A Quarto book (Python for data and plots), rendered to HTML and PDF, with the curated CSVs in
the repo so the figures are reproducible and the numbers are citeable.

```
open-labs/
  README.md              this scope, then the book's front matter
  _quarto.yml
  data/                  runs.csv, labs.csv, claims.csv, sources.md
  src/                   load.py, fits.py (power-law fits and slopes), plots.py (shared style)
  chapters/
    00-intro.qmd
    01-marin.qmd  02-olmo.qmd  03-smollm.qmd  04-deepseek.qmd  05-llama3.qmd ...
    90-cross-lab.qmd
    99-methods.qmd
```

## 9. Phases

| Phase | Work | Rough effort |
|---|---|---|
| 0 | This scope | done |
| 1 | Schema, plotting code, and two pilot chapters (Marin, OLMo 3) end to end | ~1 week |
| 2 | Six more chapters (SmolLM3, DeepSeek V3, Llama 3, LLM360 K2, Apertus, modded-nanogpt) | ~2 weeks |
| 3 | Cross-lab chapter; map results back onto the paper's S_E, g_A, and returns-to-research terms | ~3 days |
| 4 | Send chapters to Ai2, Marin, and Hugging Face for correction; fold in replies | ongoing |

Effort figures are guesses and assume one person part-time.

## 10. Decisions (Tom, 2026-09-13)

1. **y-axis:** both. Loss for within-lab curves, benchmarks alongside; cross-lab uses benchmarks.
2. **Scope of "experiments":** Prime Intellect's RL chapter runs in the pilot so the template is
   tested on post-training data early.
3. **Tier B depth:** stop after DeepSeek and Llama 3 unless a report is unusually detailed.
4. **Closed-lab reference figures** (GPT-4, Chinchilla): include as an appendix.
5. **Audience:** internal METR reference. Chapters can carry working notes and rough estimates,
   with confidence flags rather than polished hedging.
6. **Tooling:** Quarto book.
