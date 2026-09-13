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


## Verification pass 2026-09-13 (arXiv reachable): item-by-item status

Method: arXiv HTML renders (LaTeXML) of both papers were fetched and converted to text - https://arxiv.org/html/2304.03208 (Cerebras-GPT, v1 dated 06 Apr 2023) and https://arxiv.org/html/2304.01373 (Pythia, v2 dated 31 May 2023); PDFs were also downloaded (no pdftotext/pypdf available, so the HTML text is what was read). Hugging Face: `EleutherAI/pythia-12b` README fetched (200); every `cerebras/*` endpoint (model page, `/raw/main/README.md`, `/resolve/`, `/api/models/`) returned HTTP 401 "Invalid username or password" through the proxy, and `git clone https://huggingface.co/...` needs credentials, so the Cerebras HF card itself is still not fetched directly (values below come from the paper's own Table 7/8 instead). cerebras.ai / cerebras.net: CONNECT 403 (blocked). W&B GraphQL (`https://api.wandb.ai/graphql`) is anonymously readable and was used for training dates (full dump of the 10,352 runs in `eleutherai/pythia` saved to the scratchpad as `reference/wandb_pythia_runs.json`). Quotes are as printed in the HTML render.

### Items previously "VERIFIED VIA SECONDARY MIRRORS ONLY (paper itself blocked)"

- Cerebras-GPT scaling-law equation and coefficients -> **CONFIRMED** (primary). arXiv 2304.03208 Sec. 3.1, Eq. (1): "\mathcal{L}(f)=(f/5.984e22)^{-0.0737}+0.5066", introduced as "the compute-optimal frontier scaling law here ( f is compute FLOPs to loss, \mathcal{L} )". Footnote 3: "Pile test loss is crossentropy in nats/token."
- muP aggregate results (0.43%, 1.7%, 16x) -> **CONFIRMED**. Sec. 3.3: "our µP models exhibit an average of 0.43% improved Pile test loss and 1.7% higher average downstream task accuracy compared to our SP models" and "µP models show an average of 0.43% better Pile test loss compared to the Cerebras-GPT SP scaling law fit. Further, µP models show substantially lower variance with just 0.04% standard deviation relative to the SP scaling law, while SP models show deviation 0.66% ( \sim16\times more noisy). For perspective, the run-to-run standard deviation in loss when using different initialization and data random seeds is around 0.35%." Sec. 3 intro rounds it to "improves the compute-optimal frontier loss by 0.4%". Caveat now visible: the 0.43% is vs the SP *fit*; per-model, muP is worse than the SP model at 256M and 2.7B (see table below), and the paper says of 2.7B "we were just lucky when choosing the SP 2.7B model hyperparameters".
- 13B-within-0.5% claim -> **CONFIRMED**. Sec. 3.1: "We estimated the 13B model loss using a similar scaling law from models up to 6.7B parameters, and the 13B model trained to within 0.5% of projected loss. Extending the existing scaling law shows that if we budgeted to train a model with FLOPs equivalent to GPT-NeoX 20B, we would expect the Cerebras-GPT model loss to be \sim1.2\% better than GPT-NeoX 20B." Also: "The largest Pythia model at 12B parameters is trained with 25.3 tokens per parameter and is just 0.3% loss above the Cerebras-GPT scaling law."
- Author list (8) -> **CONFIRMED**. Title block: Nolan Dey, Gurpreet Gosal, Zhiming (Charles) Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, Joel Hestness; affiliation Cerebras Systems; contact {nolan,joel}@cerebras.net.

### Items previously "NOT RETRIEVED"

- **Cerebras-GPT per-model muP Pile test losses (paper Table 2/3)** -> **RETRIEVED / CONFIRMED**. They are in Sec. 3.3 Table 3 ("Pile pre-training test loss and zero-shot downstream task results for µP and SP models") and repeated with Training FLOPs in Appendix C.2 Table 8. (Note: the paper's "Table 2" is the comparison against OPT/Pythia/GPT-NeoX, not the muP table.) As printed:

    | model | Training FLOPs (Tab. 8) | SP Pile test xent (paper Tab. 3/8) | muP Pile test xent | muP HellaSwag | muP PIQA | muP WinoGrande | muP Lambada | muP ARC-e | muP ARC-c | muP OBQA | muP downstream avg |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 111M | 2.6e18 | 2.608 | 2.588 | 0.268 | 0.598 | 0.519 | 0.204 | 0.390 | 0.176 | 0.124 | 0.325 |
    | 256M | 1.3e19 | 2.349 | 2.359 | 0.274 | 0.617 | 0.505 | 0.287 | 0.427 | 0.194 | 0.156 | 0.351 |
    | 590M | 6.1e19 | 2.181 | 2.155 | 0.295 | 0.644 | 0.517 | 0.362 | 0.470 | 0.194 | 0.172 | 0.379 |
    | 1.3B | 2.8e20 | 1.997 | 1.984 | 0.334 | 0.682 | 0.512 | 0.471 | 0.515 | 0.223 | 0.196 | 0.419 |
    | 2.7B | 1.1e21 | 1.834 | 1.846 | 0.388 | 0.697 | 0.557 | 0.558 | 0.569 | 0.241 | 0.218 | 0.461 |

  SP downstream numbers in Table 3/8 (e.g. 111M 0.268 / 0.594 / 0.488 / 0.194 / ... / avg 0.315; 13B 0.513 / 0.766 / 0.646 / 0.696 / 0.714 / 0.367 / 0.286 / 0.570) are identical to the HF-card table above. Paper Table 1 confirms muP LR 6.0E-03 for all five muP models, linear decay, same batch/tokens as the SP twins.

  **DISCREPANCY (new): the paper's SP Pile test losses differ from the HF card for 5 of 7 models.** Paper Table 3/8 vs HF card (as mirrored above): 111M 2.608 vs 2.566; 256M 2.349 vs 2.299; 590M 2.181 vs 2.184; 1.3B 1.997 vs 1.996; 2.7B 1.834 vs 1.834; 6.7B 1.704 vs 1.704; 13B 1.572 vs 1.575. The paper's 13B value (1.572) appears identically in Table 2 (Sec. 3.2) and Table 8; neither 2.566 nor 1.575 occurs anywhere in the paper text. The Training FLOPs column (2.6e18, 1.3e19, 6.1e19, 2.8e20, 1.1e21, 6.3e21, 2.3e22) matches the card exactly. Not resolvable without the live card (401); `runs.csv` currently carries the card values. For any SP-vs-muP comparison use the paper's SP column, since the muP losses exist only in the paper.

- **Pythia numerical Pile val/test loss** -> partly superseded by the W&B pull above (validation loss); additionally, **REFUTED as "figure-only"**: Cerebras-GPT Appendix C.2 Table 8 prints numeric Pile *test* cross-entropy (nats/token, GPT-2 vocab, evaluated by Cerebras: "we run evaluation ourselves on all checkpoints rather than using published numbers") for all 16 Pythia models, with their FLOPs count: Pythia 70M 1.6e20 / 2.504; 160M 4.1e20 / 2.186; 410M 1.1e21 / 1.971; 1B 2.2e21 / 1.845; 1.4B 3.2e21 / 1.793; 2.8B 6.1e21 / 1.720; 6.9B 1.4e22 / 1.626; 12B 2.4e22 / 1.582; "Pythia Pile-dedup" 70M 2.549; 160M 2.204; 410M 1.989; 1B 1.858; 1.4B 1.889; 2.8B 1.724; 6.9B 1.644; 12B 1.601 (also Table 2: Pythia 12B 2.4e22, 1.582). These are third-party (Cerebras) evaluations on the Pile *test* split - a different split from the W&B `validation/lm_loss` numbers, and for the deduped models the test set is the standard Pile test set, unlike the W&B validation set. Appendix C.1 Figure 8 additionally plots intermediate Pile test loss vs FLOPs through training for both suites (figure-only). The Pythia paper itself still reports no numeric loss (grep of the full text: the only "perplexity"/"loss" mentions are the CrowS-Pairs / LAMBADA bias case study) - CONFIRMED absent.

- **Training start/end dates, Pythia** -> **PARTLY RETRIEVED from W&B (primary logs), still absent from the paper.** The paper states no training dates (only: v0 "overwritten ... on March 31, 2023", rename "on January 20, 2023", App. B). W&B groups are per-restart segments (e.g. the 12B finishing segment `v2-12B_2w9vgi4o` ran 36 h on 32 nodes, vs ~282 h implied by 72,300 A100-h / 256 GPUs), so start dates below are the earliest config-matched segment sharing the finishing group's name prefix (Pile vs deduped path, 143k iters, batch 1024 seq); end dates are the heartbeat of the segment that logged step 143000 with the loss recorded in `runs.csv`. UTC.

    | model | W&B name prefix | first segment created | final segment (step 143000) finished | nodes (x8 A100) |
    |---|---|---|---|---|
    | 70m | `v2 70M` | 2023-02-04 | 2023-02-05 (`_1ephjcv4`) | 4 |
    | 70m-deduped | `v2-70M-deduped` | 2023-02-04 | 2023-02-05 (`_3q84unyq`) | 4 |
    | 160m | `v2 160M` | 2023-01-24 | 2023-01-26 (`_3j6ymzmx`) | 4 |
    | 160m-deduped | `v2 160M deduped` | 2023-01-24 | 2023-01-26 (`_1prktcxu`) | 4 |
    | 410m | `v2-410m` | 2023-01-31 | 2023-02-05 (`_1dz9o23i`) | 4 |
    | 410m-deduped | `v2-410m-deduped` | 2023-01-31 | 2023-02-05 (`_qmo52ze3`) | 4 |
    | 1b (README mapping) | `800M Pythia` | 2022-10-11 | 2022-10-14 (`_1zw5etef`) | 8 |
    | 1.4b | `v2 1.4B` | 2023-01-24 | 2023-01-31 (`_1ey6n7g1`) | 8 |
    | 1.4b-deduped | `v2 1.4B deduped` | 2023-01-24 | 2023-01-31 (`_1dhzgs7f`) | 8 |
    | 2.8b (README mapping) | `2.7B New` | 2022-11-08 | 2022-11-19 (`_36751euw`) | 8 |
    | 2.8b-deduped (README mapping) | `2.7B Deduped New` | 2022-11-08 | 2022-11-22 (`_1ygfbs9n`) | 8 |
    | 6.9b | `6.7B Decay` | 2022-10-19 | 2022-11-01 (`_mxigvd4u`) | 16 |
    | 12b | `v2-12B` | 2023-02-01 | 2023-03-09 (`_2w9vgi4o`) | 32 |

  Node counts (W&B runs per group; 4/4/4/8/8/8/16/32 nodes = 32/32/32/64/64/64/128/256 GPUs) match paper Table 5 exactly. Caveats: (i) a second complete 12B group `v2-12B_78mtlikk` finished 2023-02-14 at step 143000 with val loss 1.7420 (vs 1.7412 for `_2w9vgi4o`); which one is the released `pythia-12b` is not determinable from W&B alone. (ii) A `v2-1b-bf16` group (16L x 2048, Pile, 143k steps) finished 2023-03-12 with val loss 2.0492; given the paper/README statement that the released 1B was trained in bf16 and the Jan-Mar 2023 v1 retrain, it is a plausible alternative to the README's `800M Pythia` mapping (Oct 2022, val 2.0445) - STILL UNVERIFIED which is the released checkpoint. (iii) `v2 2.8B_2dwg5j9x` (Feb 2023, val 1.8693) also exists alongside the README-mapped `2.7B New` (Nov 2022). (iv) No deduped 6.9B run and no deduped 12B run reached 143k steps in the project (`v2-12B-deduped` segments stop at step 74,982, Feb 2023), so 6.9b-deduped / 12b-deduped dates remain unknown. Net: the `runs.csv` guess "2023-01 -> 2023-03" is consistent for 70m/160m/410m/1.4b (+deduped) and 12b, but REFUTED for the README-mapped 1b (2022-10), 2.8b / 2.8b-deduped (2022-11) and 6.9b (2022-10 -> 2022-11).

- **Training start/end dates, Cerebras-GPT** -> **STILL UNVERIFIED (not in the paper).** The only date-like statements: Appendix Table 7 model card "Model date: March 2023" (release, not training) and the arXiv stamp "arXiv:2304.03208v1 [cs.LG] 06 Apr 2023". No wall-clock, "weeks" or calendar statement anywhere in the paper; the "few weeks" claim remains press-release-only (blocked). Keep `guess`.

- **Any CS-2 accelerator-hours (Cerebras)** -> **STILL UNVERIFIED / CONFIRMED NOT DISCLOSED in the paper.** No accelerator-hours, absolute throughput, or wall-clock figures appear; Sec. 5.3 Tables 4-6 give only *relative* scaling ("Performance relative to 1 CS-2", "Relative Utilization"). New reported fact that partially fixes the hardware cell: Table 6 ("Andromeda FLOP/s utilization relative to 1 CS-2 training the 111M parameter model") lists the number of CS-2 systems and per-CS-2 batch used per training run: 111M 1 (batch 120); 256M 1 (264); 590M 1 (264, rel. util. 0.92); 1.3B 4 (528 -> 132/CS-2, 0.96); 2.7B 4 (528 -> 132, 0.96); 6.7B 16 (1040 -> 65, 1.05); 13B 12 (1080 -> 45, 1.02). So only the 6.7B run used all 16 CS-2s of Andromeda; `runs.csv` hardware strings saying "16x CS-2" for every model are imprecise. Also Sec. 5.3: "Andromeda achieves linear scaling within 9% for all model sizes and CS-2 system counts."

### Re-checks of items previously marked VERIFIED via GitHub mirrors (now against the arXiv text)

- Pythia Appendix D Table 5 A100-hours -> **CONFIRMED verbatim**: "| 70 M | 32 | 510 | 160 M | 32 | 1,030 | 410 M | 32 | 2,540 | 1.0 B | 64 | 4,830 | 1.4 B | 64 | 7,120 | 2.8 B | 64 | 14,240 | 6.9 B | 128 | 33,500 | 12 B | 256 | 72,300 | Total | | 136,070", "calculated via (iteration time (s) \times number of iterations \times number of GPUs \div 3600 s/hour). All GPUs are A100s with 40GB of memory." and "Thus the total compute required for training the models for this paper was 544,280 A100-hours." (Appendix D is titled "Training Hardware and GPU hours"; the earlier note calling it "Appendix C" was off by one - the table number, 5, is right.)
- Pythia Table 1 non-embedding params and LRs -> **CONFIRMED**: 70M 18,915,328 / 6 / 512 / 8 / 10.0e-4; 160M 85,056,000 / 12 / 768 / 12 / 6.0e-4; 410M 302,311,424 / 24 / 1024 / 16 / 3.0e-4; 1.0B 805,736,448 / 16 / 2048 / 8 / 3.0e-4; 1.4B 1,208,602,624 / 24 / 2048 / 16 / 2.0e-4; 2.8B 2,517,652,480 / 32 / 2560 / 32 / 1.6e-4; 6.9B 6,444,163,072 / 32 / 4096 / 32 / 1.2e-4; 12B 11,327,027,200 / 36 / 5120 / 40 / 1.2e-4.
- Pythia acknowledgments -> **CONFIRMED verbatim**: "We are grateful to Stability AI for providing the compute required to train these models, and to CoreWeave for providing compute for some of the evaluations."
- Pythia Appendix B corrections -> **CONFIRMED**: v0 160M/410M/1.4B at 4M-token batch; uniform 2M-token batch in v1; Flash Attention added before the retrain; v0 6.9B/12B decayed LR to 0 vs 10% for the rest, fixed in v1; "We overwrote the previously public preliminary version of the suite ... on March 31, 2023"; renamed "on January 20, 2023". Author list (13) and affiliations CONFIRMED from the title block (Purohit listed with EleutherAI + Stability AI).
- Cerebras-GPT Table 1 -> **CONFIRMED with one correction to the suite summary above.** As printed: 111M 768/10/64/3072, 2.2B tokens, batch 246K tokens, LR 6.0E-04, **Linear**; 256M 1088/14/64/4352, 5.1B, 541K, 6.0E-04, Linear; 590M 1536/18/128/6144, 11.8B, 541K, 2.0E-04, Linear; 1.3B 2048/24/128/8192, 26.3B, 1.08M, 2.0E-04, **Cosine**; 2.7B 2560/32/80/10240, 53.0B, 1.08M, 2.0E-04, Cosine; 6.7B 4096/32/128/16384, 133.2B, 2.13M, 1.2E-04, Linear; 13B 5120/40/128/20480, 257.1B, "1.47M \to 2.21M", 1.2E-04, Cosine; all five muP models 6.0E-03, Linear. Sec. 2.3: "We find that linear learning rate decay tends to perform better than cosine decay, so we use it in most of our pre-training runs. With either decay type, we warm up learning rate linearly over 375M tokens and then decay to 10\% of the maximum learning rate." and "For the 13B parameter model, we train with a batch size of 720 sequences of length 2048 tokens for the first 84B tokens ... we increased the batch size to 1080 sequences for the rest of training." So the "10x cosine decay" wording in the Cerebras suite summary is **REFUTED in part**: decay to 10% of max LR is right, but the schedule is linear for 111M/256M/590M/6.7B and all muP runs, cosine only for 1.3B/2.7B/13B. Also from Sec. 2.3: AdamW betas (0.9, 0.95), eps 1e-8 (1e-9 for 6.7B/13B), wd 0.1, no dropout, grad-clip 1.0, bf16 for all released models.
- Cerebras-GPT FLOPs convention -> **CONFIRMED** (Appendix F): "we account for the dot product between softmax(QK^T) and V" and "embedding layers do not need to calculate a delta gradient"; called "Algorithmic FLOPs"; excludes recomputation. Appendix E Table 12 gives the parameter-count formula (embedding = vocab*d + d*seq; per layer 12 d^2 + 13 d; final LN 2d) - the same formula used for the `params_total` values in `runs.csv`.
- Cerebras "Intermediate checkpoints/loss curves: none" -> amend: Appendix C.1 Figure 8 shows "the intermediate Pile test losses achieved throughout training for Pythia and Cerebras-GPT models" (figure-only; no numeric intermediate values).
