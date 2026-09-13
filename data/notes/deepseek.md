# DeepSeek: experiments vs final training runs — factual notes

Collected 2026-09-13. Primary sources are the DeepSeek technical reports; because arxiv.org, huggingface.co, wikipedia and most news sites were blocked by the network egress proxy in this session, paper text was obtained from (a) the PDFs DeepSeek hosts in its own GitHub repos (V2, R1, V3.2-Exp — text extracted locally) and (b) GitHub-hosted plain-text transcriptions of the arXiv papers (DeepSeek LLM, V3, V3.2, R1 v2). Source URLs below point at the canonical arXiv/GitHub locations.

## 1. The lab

DeepSeek is the AI research arm spun out of the Hangzhou quant hedge fund High-Flyer (founder Liang Wenfeng), founded July 2023 and self-funded by High-Flyer through at least 2025 (secondary sources; not directly fetched). Its own SC24 paper describes the infrastructure inheritance: High-Flyer built Fire-Flyer 2 in 2021 with **10,000 PCIe A100 GPUs** (~1,250 nodes x 8), "achieved performance approximating the DGX-A100 while reducing costs by half and energy consumption by 40%" (https://arxiv.org/abs/2408.14158). All V2/V3-era training reports describe an **NVIDIA H800 cluster**; the V3 report gives its size as **2048 H800 GPUs** (https://arxiv.org/abs/2412.19437). External estimates of the total fleet are much larger: SemiAnalysis (Jan 31 2025) estimated ~50,000 Hopper-class GPUs (about 10k H800 + 10k H100 + H20s), ~$1.6B server capex and ~$944M operating cost (https://semianalysis.com/2025/01/31/deepseek-debates/ — page blocked; figures from search snippets).

Headcount: press reports say ~150–160 employees (FT via Tom's Hardware, Dec 2024). Author counts on the reports grow steadily and are a reasonable proxy: DeepSeek LLM **86** (counted), V2 **158** (counted from PDF appendix), V3 **139** (Interconnects count of v1) to ~200 (arXiv listing), R1 **195** (counted, v1, includes departed members), V3.2 **342** (count by a mirrored transcription; unverified).

## 2. Final runs

| Model | Params (total/active) | Tokens | Compute | Release |
|---|---|---|---|---|
| DeepSeek LLM 7B / 67B | 6.9B / 67B dense | 2T each | 67B: 601K H800-h **derived** from V2's "300.6K GPU hours" per trillion tokens; FLOPs 8.5e22 / 9.9e23 via the paper's M·D | Nov 2023 (paper Jan 5 2024, https://arxiv.org/abs/2401.02954) |
| DeepSeek-V2 | 236B / 21B | 8.1T | 1.40M H800-h **derived** (172.8K/T x 8.1T); "saves 42.5% of training costs" vs 67B | May 6 2024 (https://arxiv.org/abs/2405.04434) |
| DeepSeek-V2-Lite | 15.7B / 2.4B | 5.7T | — | May 16 2024 |
| DeepSeek-V3 | 671B / 37B | 14.8T | **2.664M** pretrain + **119K** context ext. + **5K** post-train = **2.788M H800-h = $5.576M at $2/h** (reported); Pile-test 0.548 BPB; MMLU 87.1 | Dec 26 2024 (https://arxiv.org/abs/2412.19437) |
| DeepSeek-R1 | 671B / 37B (RL on V3-Base) | — | **147K H800-h = $294K** (R1-Zero 101K, SFT data 5K, R1 41K) on 512 H800s; 198 h + 80 h (arXiv v2 Table 7, https://arxiv.org/abs/2501.12948v2) | Jan 20 2025 |
| DeepSeek-V3.1 | 671B / 37B | +840B continued pretraining (630B @32K, 209B @128K) | not stated | Aug 21 2025 (https://x.com/deepseek_ai/status/1958417072536608952) |
| DeepSeek-V3.2-Exp / V3.2 / Speciale | 671B / 37B | +2.1B dense warm-up +943.7B sparse (DSA) | post-training budget "exceeding 10% of the pre-training cost" (https://arxiv.org/abs/2512.02556) | Sep 29 / Dec 1 2025 |
| DeepSeek-V4-Pro / Flash | 1.6T/49B ; 284B/13B | 33T ; 32T | GPU-hours **not disclosed** per two third-party readings of the report (https://arxiv.org/abs/2606.19348) | Apr 24 2026 |

V3 training took "less than two months" on the 2048-GPU cluster, at "180K H800 GPU hours" per trillion tokens ("3.7 days"), with "no irrecoverable loss spikes or ... rollbacks."

## 3. Experimental runs (what is disclosed)

**DeepSeek LLM (2024).** The only DeepSeek report with a full scaling-law programme. (i) Hyperparameter law: a batch-size/learning-rate grid at C=1e17 (177M FLOPs/token model), then sweeps over budgets "ranging from 1e17 to 2e19", validated at 1e20 (2.94B FLOPs/token); fits η_opt = 0.3118·C^−0.1250, B_opt = 0.2920·C^0.3271. (ii) IsoFLOP allocation: "8 different compute budgets ranging from 1e17 to 3e20, and ... around 10 different model/data scale allocations for each budget" (~80 runs, derived); fits M_opt = 0.1715·C^0.5243, D_opt = 5.8316·C^0.4757, where M = 72·n_layer·d_model² + 12·n_layer·d_model·l_seq is non-embedding FLOPs/token and C = M·D (they argue 6N mis-estimates compute by up to 50% at small scale). At the top budget the optimum is ~1.55B non-embedding params on ~32B tokens (derived), i.e. the suite tops out ~500x below the 7B run and ~3000x below the 67B run. (iii) Loss law: a power law of optimal BPB vs C fitted on the small runs, with 7B and 67B overlaid as blue stars (Fig. 5): "Their performance is well-predicted by the scaling curve" and "small-scale experiments can accurately predict the performance of models with 1000x compute budget." **No numerical prediction error or BPB values are printed**; the fit constants for the loss law are not given in the text. (iv) Data-quality dependence (Table 4): exponents a/b = 0.450/0.550 (early data), 0.524/0.476 (current data), 0.578/0.422 (OpenWebText2).

**DeepSeek-V2 (2024).** Appendix D: three 7B dense models (MHA/GQA/MQA) each on 1.33T tokens; MLA vs MHA at two MoE scales — ~16B total on 1.33T tokens and ~250B total on 420B tokens (Tables 8–9). No scaling-law fits. The 1.33T-token/16B and ~250B/420B pattern recurs in V3.

**DeepSeek-V3 (2024).** Sec 4.5 + App. B: MTP ablation on a 15.7B MoE (1.33T tokens) and a 228.7B MoE (540B tokens); auxiliary-loss-free balancing on the same sizes (578B tokens for the large); FP8 vs BF16 on "two model scales similar to DeepSeek-V2-Lite and DeepSeek-V2" (16B and 230B) for "approximately 1 trillion tokens", relative loss error "consistently below 0.25%"; batch-wise vs sequence-wise balance on 1B MoEs (validation losses 2.258/2.253/2.253) and expert-load analysis on 16B models. Summing the disclosed ablations with 6·N_active·D gives roughly 5e23 FLOPs (derived), about 15% of the final run's ~3.3e24 — a lower bound, since the report says the $5.576M figure "include[s] only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data." Nathan Lambert estimated total pretraining experimentation at 2–4x the final-run figure (https://www.interconnects.ai/p/deepseek-v3-and-the-actual-cost-of).

**DeepSeek-R1 (2025).** The arXiv v2 (Jan 2026, aligned with the Nature version) adds a cost table and states: "we utilized the A100 GPUs to prepare for the experiments with a smaller model (30B parameters). The results from this smaller model have been promising, which has allowed us to confidently scale up to 660B R1-Zero and R1." R1-Zero ran 10,400 GRPO steps (batch 512, 16 samples, LR 3e-6, KL 0.001); the second RL stage 1,700 steps. Sec 4.1 reports an RL-vs-distillation ablation on Qwen-32B-Base (>10K RL steps). Sec 4.2 lists unsuccessful attempts (PRM, MCTS) without numbers. The $294K explicitly excludes the V3-Base pretraining.

**V3.2 / V4.** V3.2 quantifies RL only relatively ("exceeding 10% of the pre-training cost") and attributes its knowledge gap to "fewer total training FLOPs." V4 notes (third-party) say the report discloses tokens (33T/32T) but not GPU-hours or hardware; a Chinese reading guide states the report puts post-training above 10% of pre-training cost.

## 4. Pattern

DeepSeek's disclosed experiment-to-final ratio falls over time: a dense-model scaling-law suite (~80 runs ≤3e20 FLOPs) predicting 7B/67B in 2024; a handful of 1.3T-token ablations at 1/40 and 1/3 of final size for V2/V3; and for R1/V3.2/V4 only qualitative mentions of small-scale preparation. Cost disclosure is consistently for the official run only.

## Verification status (unconfirmed / needs primary check)

- DeepSeek-LLM release date (2023-11-29), V3.2-Exp date (2025-09-29), V3.2 date (2025-12-01), V4 date (2026-04-24), company founding date (2023-07-17): from memory or secondary pages; primary pages blocked.
- V3 author count: 139 (Interconnects, v1) vs ~200 (arXiv v2 listing); appendix not counted directly.
- V3.2 author count 342: produced by a transcription-reading model, not counted by me.
- SemiAnalysis fleet/capex figures and the "~160 employees" FT figure: taken from search snippets; the articles themselves were blocked.
- Active parameter counts for V3's 228.7B ablation model and for the FP8 16B/230B models (2.4B/21B assumed from V2 similarity) — flagged as guesses in runs.csv.
- DeepSeek-V3.1's 840B/630B/209B token figures: from DeepSeek's X post as quoted in search results; not fetched directly.
- V4 "post-training >10% of pre-training cost" and "GPU-hours not disclosed": from third-party notes, not from the report text.
- The 6.9B parameter count for DeepSeek LLM 7B comes from the V2 report's comparison table, not the LLM paper.
- Derived GPU-hours for the 67B (601K) and V2 (1.40M) assume the per-trillion-token rates in the V2 report apply uniformly over the whole run.
- The V3 loss-law figure (Fig. 5 of DeepSeek LLM) and IsoFLOP points are figure-only; no raw data released.
