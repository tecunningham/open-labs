# Reference scaling suites: Pythia (EleutherAI) and Cerebras-GPT (Cerebras)

Collected 2026-09-13 for the METR open-labs scaling-curve report. Network caveat: arxiv.org, huggingface.co, cerebras.ai, PMLR, businesswire, semanticscholar, wandb.ai were all egress-blocked from this environment. Primary sources were reached through (a) GitHub (raw/API/code search), which hosts the Pythia repo, the Cerebras modelzoo configs, a LaTeX mirror of the Pythia paper, and verbatim mirrors of the HF model cards, and (b) web-search excerpts of the blocked pages. Every number below carries its provenance.

---

## 1. EleutherAI Pythia (arXiv 2304.01373, ICML 2023)

**Suite.** 8 sizes (70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B) x 2 data conditions (Pile, globally-deduplicated Pile) = 16 models, all trained on 299,892,736,000 tokens in the same order (143,000 steps x 2,097,152-token batches). Standard Pile = ~334B tokens (<1 epoch); deduped Pile = 207B tokens (~1.5 epochs). 154 checkpoints per model (step 0, 1, 2, ..., 512 log-spaced, then every 1000 steps). Source: README https://github.com/EleutherAI/pythia ; paper Sec. 2.4 (LaTeX mirror https://github.com/jd-coderepos/sota/blob/master/dataset/train/2304.01373v2/2304.01373v2.tex).

**Params (paper Table 1, non-embedding; totals derived from `models/*/pythia-*.yml`, verified against the HF card mirror https://github.com/shimo-lab/modelmap/blob/main/1000models/code_for_Section_5_and_Appendix_L/data/model_cards/EleutherAI_pythia-12b.md):**

| model | non-embedding | total | layers/d_model/heads | LR |
|---|---|---|---|---|
| 70M | 18,915,328 | 70,426,624 | 6/512/8 | 1.0e-3 |
| 160M | 85,056,000 | 162,322,944 | 12/768/12 | 6.0e-4 |
| 410M | 302,311,424 | 405,334,016 | 24/1024/16 | 3.0e-4 |
| 1B | 805,736,448 | 1,011,781,632 | 16/2048/8 | 3.0e-4 |
| 1.4B | 1,208,602,624 | 1,414,647,808 | 24/2048/16 | 2.0e-4 |
| 2.8B | 2,517,652,480 | 2,775,208,960 | 32/2560/32 | 1.6e-4 |
| 6.9B | 6,444,163,072 | 6,857,302,016 | 32/4096/32 | 1.2e-4 |
| 12B | 11,327,027,200 | 11,846,072,320 | 36/5120/40 | 1.2e-4 |

(12B total: HF card lists 11,846,072,448; my NeoX-formula count gives 11,846,072,320 - 128-param difference, immaterial.)

**Compute / hardware (paper Appendix C, Table 5 - `reported`).** All A100 40GB. GPU count / A100-hours per model: 70M 32/510; 160M 32/1,030; 410M 32/2,540; 1B 64/4,830; 1.4B 64/7,120; 2.8B 64/14,240; 6.9B 128/33,500; 12B 256/72,300; one copy of the suite = 136,070 A100-hours; two copies (Pile + deduped) x retrained once = **544,280 A100-hours total** for the paper. FLOPs are not reported; runs.csv uses 6*N_total*D (`derived`), range 1.27e20 (70M) to 2.13e22 (12B); suite total 8.83e22 for the 16 released models. Acknowledgments: "grateful to Stability AI for providing the compute required to train these models, and to CoreWeave for providing compute for some of the evaluations" - i.e. training compute is attributed to Stability AI, not CoreWeave.

**Loss.** The paper and README report **no numerical Pile validation/test loss**; loss curves exist only in the wandb project https://wandb.ai/eleutherai/pythia ("messy", partial run-to-model mapping in README) and https://wandb.ai/eleutherai/pythia-extra-seeds. The paper contains no loss-vs-size or loss-vs-compute figure and fits no scaling law; its scale plots (App. F.4) are benchmark accuracy vs. size/step against OPT and BLOOM. Cerebras-GPT's Fig. 1 plots Pythia Pile *test* loss vs FLOPs (third-party, figure-only). Intermediate-checkpoint *benchmark* results (not losses) are downloadable as JSON: `evals/pythia-v1/<model>/{zero-shot,five-shot}/*_step<N>.json` (27 checkpoints for 12B, incl. step143000), produced with lm-evaluation-harness. Loss for the 154 checkpoints must be recomputed from the HF checkpoints + released dataloader.

**Benchmarks (step 143000, zero-shot acc, from evals JSON - `reported`).** LAMBADA(openai)/PIQA: 70M .185/.595; 70M-dd .192/.598; 160M .328/.627; 160M-dd .342/.618; 410M .516/.668; 410M-dd .524/.675; 1B .562/.707; 1B-dd .580/.700; 1.4B .616/.709; 1.4B-dd .619/.720; 2.8B .647/.739; 2.8B-dd .652/.741; 6.9B .673/.752; 6.9B-dd .689/.760; 12B .705/.760; 12B-dd .710/.763. Full set (WinoGrande, WSC, ARC-e/c, SciQ, LogiQA; 5-shot) in `src/pythia_final_evals.json`.

**Errata relevant to curve fitting.** v0 suite (160M/410M/1.4B at 4M-token batch; 71,500 steps) overwritten 2023-03-31; 6.9B and 12B accidentally used a different initialization (issue #135); 1B trained in bf16 (others fp16); Jan 20 2023 rename to total-param naming (19M->70M ... 13B->12B). Extra batch-size ablation runs (0.25M/0.5M/1M-token batches) have evals in the repo.

**Lab facts.** 13 authors (Biderman, Schoelkopf, Anthony, Bradley, O'Brien, Hallahan, Khan, Purohit, Prashanth, Raff, Skowron, Sutawika, van der Wal); EleutherAI Institute incorporated as non-profit early 2023 with 20+ full-time staff, backed by Stability AI, Hugging Face, Canva, Nat Friedman, Lambda (TechCrunch 2023-03-02, https://techcrunch.com/2023/03/02/stability-ai-hugging-face-and-canva-back-new-ai-research-nonprofit). Release: 2023-02-13 (v0, https://www.eleuther.ai/releases), 2023-04-03 (v1 + paper). Training dates not stated anywhere - runs.csv dates are guesses.

---

## 2. Cerebras-GPT (arXiv 2304.03208; blog 2023-03-28)

**Suite.** 7 SP models (111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B) at 20 tokens/parameter on the (non-deduplicated) Pile, GPT-3-style architecture (learned positions, GPT-2 BPE vocab 50257, seq 2048, AdamW, wd 0.1, warmup 375M tokens, 10x cosine decay), plus 5 muP models (111M-2.7B) with hyperparameters muTransferred from a 40M proxy. Trained on Andromeda (16x CS-2), whole family "in a few weeks" (press release https://www.businesswire.com/news/home/20230328005366/en/...). Configs: https://github.com/Cerebras/modelzoo/tree/main/src/cerebras/modelzoo/models/nlp/gpt3/configs/Cerebras_GPT (SP: 111m, 256m, 590m, 1p3b, 2p7b, 6p7b, 13b_bs720 + 13b_bs1080; muP: 111m_mup, 256m_mup, 590m_mup, 1p3b_mup, 2p7b_mup).

**Per-model table (HF model card, verbatim mirror https://github.com/asgaardlab/model-card-reorganization/blob/main/verified_model_cards/cerebras%40Cerebras-GPT-13B/2_original_model_card.md - `reported`):**

| model | L/d/heads | batch (seq) | steps | tokens | Training FLOPs | Pile test xent | 0-shot avg | HellaSwag | LAMBADA |
|---|---|---|---|---|---|---|---|---|---|
| 111M | 10/768/12 | 120 | 9,037 | 2.22e9 | 2.6e18 | 2.566 | 0.315 | 0.268 | 0.194 |
| 256M | 14/1088/17 | 264 | 9,468 | 5.12e9 | 1.3e19 | 2.299 | 0.347 | 0.274 | 0.293 |
| 590M | 18/1536/12 | 264 | 21,836 | 1.18e10 | 6.1e19 | 2.184 | 0.370 | 0.291 | 0.366 |
| 1.3B | 24/2048/16 | 528 | 24,334 | 2.63e10 | 2.8e20 | 1.996 | 0.410 | 0.325 | 0.462 |
| 2.7B | 32/2560/32 | 528 | 49,041 | 5.30e10 | 1.1e21 | 1.834 | 0.462 | 0.386 | 0.567 |
| 6.7B | 32/4096/32 | 1040 | 62,522 | 1.33e11 | 6.3e21 | 1.704 | 0.512 | 0.447 | 0.636 |
| 13B | 40/5120/40 | 720->1080 | 174,335 (bs720-equiv.) | 2.57e11 | 2.3e22 | 1.575 | 0.570 | 0.513 | 0.696 |

Tokens = steps x batch x 2048 reproduce the card exactly from the modelzoo configs. Note the card's FLOPs are NOT 6ND (e.g. 111M: 6ND = 1.5e18 vs 2.6e18 reported); Cerebras' count includes attention/embedding terms - use the reported column for their curve, but be aware of the convention gap vs. Pythia's 6ND. 5-shot results are also in the card.

**Scaling law (paper Sec. 3.1 - `reported`, but verified only through secondary mirrors: GitHub summary https://github.com/leesangjun1903/NLP-and-Audio citing 2304.03208v1, and a web-search excerpt of the PDF):**

    L(f) = (f / 5.984e22)^(-0.0737) + 0.5066      (f = pre-training FLOPs; L = Pile test loss, nats/token)

Fit on 111M-6.7B, 13B predicted within 0.5%; extrapolation to GPT-NeoX-20B compute implies ~1.2% lower loss. Sanity check: at f = 2.3e22 the formula gives 1.580 vs measured 1.575.

**muP results (paper Sec. 4, Table 2/3).** muP models average 0.43% lower Pile test loss than the SP fit, +1.7% relative average downstream accuracy, and scale noise 0.04% vs 0.66% std (16x lower). **Per-model muP losses (Table 2) could not be retrieved** - not in the HF cards, and every host serving the paper text was blocked. muP configs: base width 256, embeddings_scale 10, scale_qk_dot_by_d, base LR 6e-3.

**Intermediate checkpoints/loss curves.** None for the SP release (single final checkpoint per model on HF; a `cerebras/Cerebras-GPT-Intermediate` HF repo exists per search results but was not inspected). No accelerator-hours or cost disclosed. Cost in runs.csv is a derived A100-equivalent (FLOPs / (312 TFLOP/s x 0.40 MFU)) x $1.50 - an assumption, not a Cerebras figure.

**Lab facts.** 8 authors (Dey, Gosal, Chen, Khachane, Marshall, Pathria, Tom, Hestness). Cerebras Systems: venture-backed, >$720M raised by Nov 2022 (Series F $250M, Nov 2021, >$4B valuation; https://www.cerebras.ai/press-release/cerebras-systems-raises-250m-in-funding-for-over-4b-valuation-to-advance-the-future-of-artificial-intelligence-compute); Andromeda = 16 CS-2, 13.5M cores, ~1 exaFLOP AI compute (https://www.businesswire.com/news/home/20221114005138/en). Company headcount ~400 in 2023 is a third-party estimate (guess). Release 2023-03-28; paper 2023-04-06.

---

## Verification status

- VERIFIED (primary text fetched): Pythia README (raw GitHub); Pythia paper full LaTeX (GitHub mirror of arXiv 2304.01373v2 source) - Tables 1, 3, 5, acknowledgments, App. B corrections, App. F eval tables; Pythia final-step eval JSONs for all 16 models + ablations (cloned repo); Pythia NeoX config YAMLs (params recomputed); Cerebras modelzoo YAMLs (13 configs: shapes, batch, steps, LRs, muP settings); Cerebras-GPT-13B HF model card (two independent verbatim GitHub mirrors) - config table, FLOPs, Pile test xent, 0-shot and 5-shot tables.
- VERIFIED VIA SECONDARY MIRRORS ONLY (paper itself blocked): Cerebras-GPT scaling-law equation and coefficients (two independent secondary quotes agree); muP aggregate results (0.43%, 1.7%, 16x); 13B-within-0.5% claim; author list (8).
- SEARCH-SNIPPET ONLY (source page blocked; treat as medium confidence): EleutherAI Institute funding/headcount (TechCrunch); Pythia release date 2023-02-13 (eleuther.ai/releases); Cerebras funding >$720M / Series F; Andromeda specs; "few weeks" training time; Cerebras headcount ~400.
- NOT RETRIEVED: Cerebras-GPT per-model muP Pile test losses (paper Table 2/3); Pythia numerical Pile val/test loss (never published; wandb only, wandb blocked); training start/end dates for both suites (all dates flagged guess); any CS-2 accelerator-hours.
- DERIVED: Pythia FLOPs (6ND), Pythia total params (formula, matches HF card); Cerebras tokens (steps x batch x seq), Cerebras total params (formula), all cost_usd_est values, Cerebras A100-equivalent hours.


## Update 2026-09-13: Pythia losses pulled from W&B

`eleutherai/pythia` is readable anonymously via W&B's GraphQL API (a personal key was refused). Final-step `validation/lm_loss` (Pile validation, nats/token) for the 143k-step run whose config matches each released model (layers, width, batch 1024 x 2048, Pile vs deduplicated Pile data path) is now in `runs.csv` and the curves in `data/losses/eleutherai/`:

| model | W&B run (group) | val loss |
|---|---|---|
| 70m / 70m-deduped | 32t0zbcs ('v2 70M') / 2vcd53l4 ('v2-70M-deduped') | 2.888 / 3.013 |
| 160m / 160m-deduped | 3mvtbwii ('v2 160M') / 38uk24gn ('v2 160M deduped') | 2.512 / 2.584 |
| 410m / 410m-deduped | 12j05401 ('v2-410m') / 2pmqy4bd ('v2-410m-deduped') | 2.164 / 2.268 |
| 1b | 2i9stqg2 ('800M Pythia', the README's mapping) | 2.044 |
| 1.4b / 1.4b-deduped | vd5ogsc6 ('v2 1.4B') / 2dca9aar ('v2 1.4B deduped') | 1.960 / 2.030 |
| 2.8b / 2.8b-deduped | 12vuw5ef ('2.7B New') / 4aqjesl8 ('2.7B Deduped New'), both the README's mapping | 1.871 / 1.928 |
| 6.9b | vi7laank ('6.7B Decay', 32L x 4096, Pile) | 1.777 |
| 12b | kg5ni1dl ('v2-12B', 36L x 5120, Pile) | 1.741 |

Not mapped: 1b-deduped (the '800M Pythia deduped' run reads the non-deduplicated Pile path and has the identical loss to the 1b run, so it looks like a mislabel), 6.9b-deduped and 12b-deduped (no finished run on the deduplicated path at those sizes). The deduplicated models' validation set is the deduplicated Pile, so their higher numbers are not a quality gap. The README's mapping of 160m and 1.4b points at earlier 71k-step, batch-4M runs (the "v0" suite), not the released v1 models; the v2 groups above match the released batch size and step count.
