# Prime Intellect — experiments-to-final-run notes (pilot chapter, RL focus)

Collected 2026-09-13. Primary sources actually read in full: INTELLECT-1 report PDF (from the `prime-diloco` GitHub repo, https://github.com/PrimeIntellect-ai/prime-diloco/blob/main/INTELLECT_1_Technical_Report.pdf, = arXiv 2412.01152), INTELLECT-2 report PDF (https://storage.googleapis.com/public-technical-paper/INTELLECT_2_Technical_Report.pdf, = arXiv 2505.07291), INTELLECT-3 report PDF (https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf, = arXiv 2512.16144), and prime-rl GitHub configs. arxiv.org, huggingface.co, primeintellect.ai (blog + docs), Wikipedia, TechCrunch and most secondary sites were blocked by the egress proxy; anything attributed to those is from search-result snippets only and is flagged below.

## 1. Lab shape and timeline

Prime Intellect (founded 2024 per TechCrunch snippet; CEO Vincent Weisser, CTO/research lead Johannes Hagemann) raised a $5.5M seed (Apr 2024), a $15M seed extension led by Founders Fund (Feb 2025; https://www.primeintellect.ai/blog/fundraise, snippet only) and a $130M Series A at a $1B valuation led by Radical Ventures (8 Jul 2026; https://techcrunch.com/2026/07/08/prime-intellect-raises-130m-series-a-to-help-enterprises-build-their-own-ai-agents/, snippet only). Headcount snippets: 23 FTE around Feb 2025 (startuphub/Sacra), 32 (PitchBook), 51 as of May 2026 (AlphaSignal). Author counts on the reports: 12 (I-1, incl. 5 external), 13 (I-2), 22 (I-3).

Three flagship runs, each roughly 6 months apart, with a clear shift in where compute came from:

| Project | Stage | Compute | Dates | Release |
|---|---|---|---|---|
| INTELLECT-1 (10B dense, 1T tokens) | pretraining | up to 112 H100 donated by 30 contributors, 3 continents | 2024-10-10 to 2024-11-22 (42 d) | 2024-11-29 |
| INTELLECT-2 (32B, RL on QwQ-32B) | RL | trusted H100 training cluster (size unstated) + permissionless inference swarm | all runs "over the duration of two weeks"; launch blog 2025-04-15 | 2025-05-12 |
| INTELLECT-3 (106B-A12B MoE, SFT+RL on GLM-4.5-Air-Base) | SFT + RL | 512 H200 centralized cluster, "two months" incl. ablations | ~Sep–Nov 2025 (derived) | 2025-11-27 |

Precursor to I-1: OpenDiLoCo (Jul 2024, arXiv 2407.07852) with 150M and 1B DiLoCo runs of 88k steps (https://github.com/PrimeIntellect-ai/OpenDiloco). No public INTELLECT-3.1 model exists as of Sep 2026, but prime-rl ships an `examples/advanced/intellect-3.1/rl.toml` config (INTELLECT-3-Base, 4 train + 12 inference nodes, 131k context, mixed SWE/deep-research/math/logic/code), so a 3.1 run is at least planned/underway (https://github.com/PrimeIntellect-ai/prime-rl/blob/main/examples/advanced/intellect-3.1/rl.toml).

## 2. How the RL experiment process worked (INTELLECT-2)

The I-2 report (Sec 3 "Training Recipe" explicitly says it describes "the ablation experiments that led to it") shows a small-model-first ladder:

1. **Async-delay ablation at 1.5B.** Replicated DeepScaleR's synchronous run on DeepSeek-R1-Distill-Qwen-1.5B (ctx 2048, 1200 steps) and compared 1-, 2- and 4-step asynchronous prime-rl runs. All four reward curves overlap (~0.10 -> ~0.45, Fig 7). This licensed 2-step asynchrony for the main run and the report argues 4–5 step delays would hide all blocking stages.
2. **Difficulty filtering at 7B.** On R1-Distill-Qwen-7B with the raw DeepScaleR set, reward "barely improved"; filtering to problems with base pass@8 in (12.5%, 50%) made reward rise 0.34 -> 0.42 over ~700 steps (Fig 8). The 7B model was then used to pre-filter the 285k-task I-2 dataset. Online filtering (resample until the batch has non-zero advantages) was added on top.
3. **Stability at 32B.** Two-sided GRPO clipping (upper bound delta=4 on the ratio for negative advantages) was introduced after loss/grad-norm spikes "particularly as our models got larger". Even so, both 32B models (QwQ-32B and R1-Distill-Qwen-32B, on MATH) showed escalating gradient norms and clip ratios (Fig 9); QwQ diverged earlier, which the authors attribute to its prior RLVR. Entropy fell then rose past ~150 steps and "soon after ... we observed our model collapsing across all of our ablation runs" (Fig 10). Fixes tried: aggressive gradient clipping at 0.05–0.1 (adopted; delays but does not remove collapse), larger KL weight (rejected: slowed learning), disabling torch.compile (adopted after a compile-induced late collapse at 1.5B, Fig 11).
4. **Length reward.** Reproduced L1 at small scale (targets 500–3000 tokens, max seq 4000); small models learned budget following, but at 32B the length penalty fell far more slowly and the model did not learn to obey the budget in the ~200–350 step runs (Fig 12b/d). Left unresolved.
5. **Two main runs**: TARGET-SHORT (~200 steps, targets 1k–4k) and TARGET-LONG (~350 steps, targets 2k–10k; the released model). 4096 rollouts/step (256 prompts x 16), 8 optimizer steps/rollout step, lr 3e-7, 32K context. Inference:training FLOPs ~4.5:1. Final evals vs QwQ-32B: AIME24 78.8 vs 76.6, AIME25 64.9 vs 64.8, LiveCodeBench-v5 67.8 vs 66.1, GPQA-D 66.8 vs 66.3, IFEval 81.5 vs 83.4 — the authors concede gains were small because QwQ was already heavily RL-trained.

Not reported anywhere: training-cluster GPU count, GPU-hours, cost, number of ablation runs, or when ablations ran relative to the 15 Apr launch.

## 3. How it worked for INTELLECT-3

Centralized 512-H200 cluster; "both stages, including multiple ablations, were carried out ... over the course of two months". Process visible in the report:

- **Environment validation on a 4B proxy**: Qwen3-4B-Instruct-2507 given 26 SFT steps + 122 RL steps on DeepDive to prove the deep-research environment worked (Fig 7) before adding it to the 106B mix.
- **Difficulty annotation with a 4B proxy**: solve rates of Qwen3-4B over 8–16 generations used to filter math (21.2K), code (8.6K), science (29.3K) and logic (11.6K) problems; online filtering and an "easy pool" removed pass-rate-1 prompts during RL.
- **Algorithm ablation under deliberately high off-policyness**: "we use async-8 as a testbed for algorithms"; GSPO collapsed while CISPO climbed 0.3 -> 0.8 over ~400 steps (Fig 10). The shipped algorithm is masked token-level importance sampling (IcePop-style, alpha 0.5, beta 5) because trainer/inference probability mismatch otherwise "can cause runs to crash multiple days into the experiments".
- **Systems ablation**: without in-flight weight updates step time was >2x the ~1500 s achieved at 65,536 context on 16 train / 44 inference nodes.
- **SFT**: stage 1 (~1400 steps by figure, 33M tokens/step, Muon lr 5e-5), stage 2 agentic (800 steps, 98K context, lr 5e-8 decayed). Smooth losses, no spikes (Fig 8).
- **Main RL run**: ~600 steps (figure), 256x16 rollouts/step, max_off_policy_steps 8, lr 1e-6; online evals every 15 steps trended up with "no sign of plateauing" (Fig 9). Final: AIME24 90.8, AIME25 88.0, LCB-v6 69.3, GPQA-D 74.4, HLE 14.6, MMLU-Pro 81.9 (vs GLM-4.5-Air 84.6/82.0/61.5/73.3/13.3/73.9).

Derived compute: ~480 H200 x ~250 h = ~1.2e5 GPU-hours for the RL run alone; ~7.5e5 GPU-hours upper bound for the whole two-month program. No cost, FLOPs or experiment-share figure is published.

## 4. Smaller-scale RL research since I-3 (snippets only)

- "Systematic Reward Hacking and Prime Sprints" (20 May 2026, https://www.primeintellect.ai/blog/reward-hacking): backdoor-ifeval environments on Llama-3.2-1B-Instruct, 100 steps, batch 128, lr 1e-4, ~$0.64 / 30 min per run; hack emergence framed as a hidden-vs-visible reward dynamics problem. Sprints program sponsors community runs.
- "Scaling Agentic RL: 365,000+ Environments" (Jul 2026, https://www.primeintellect.ai/blog/scaling-agentic-rl): ~198k SWE, ~28.6k terminal, ~137.6k search tasks in 23 tasksets; ~135k images; thousands of concurrent sandboxes. Data/infra, not training curves.
- "RL at 1T Scale" (https://www.primeintellect.ai/blog/rl-at-1t-scale) and prime-rl 0.6.0 (21 Jun 2026): systems results (GLM-5 ~1T run on 28 H200 nodes; ~21–22 s weight sync). No reward-vs-compute scaling law has been published by Prime Intellect; the only fitted relation in any source is none.

## 5. What is reconstructable

- I-1: full hyperparameters (Table 5), dataset mix, MFU table, dates, benchmark tables; loss only as a perplexity figure; intermediate checkpoints released (e.g. step 49200 on HF). Public dashboard existed at app.primeintellect.ai/intelligence.
- I-2: full RL hyperparameters, dataset (HF `PrimeIntellect/Intellect-2-RL-Dataset`), reward/length-penalty curves as smoothed figures, evalchemy settings. No W&B export, no GPU-hours, no cluster size.
- I-3: full recipe (envs on Environments Hub, SFT sources with token counts, RL hyperparameters, node split, step time), benchmark-vs-step curves as figures, eval environments public. W&B projects referenced in prime-rl configs (`intellect-3.1`, `swe-ablations`) are private. `raw_data_available` is therefore `figure_only` for every curve and `numbers_in_text` for hyperparameters.

## Verification status (unconfirmed or single-snippet claims)

1. I-2 launch date 2025-04-15 and release 2025-05-12 — from search-result metadata; blog pages blocked.
2. I-3 release date 2025-11-27 — from Puter/blog metadata snippet; blog blocked.
3. Headcounts (23 in Feb 2025; 32 PitchBook; 51 in May 2026) — search snippets citing blocked pages (startuphub.ai, sacra.com, pitchbook.com, alphasignal.ai).
4. Funding amounts/investors/valuation/$100M ARR — TechCrunch, Seeking Alpha, pulse2 snippets; pages blocked. Founding year "2024" (TechCrunch snippet) vs seed in Apr 2024 is plausible but unverified.
5. Reward-hacking experiment numbers (1B, 100 steps, batch 128, lr 1e-4, $0.64) and "Scaling Agentic RL" task counts — snippets of blocked blog posts.
6. All step counts read from figures (I-2 ~200/~350; I-3 SFT ~1400, RL ~600; GSPO ablation ~400) are `guess`. The I-3 RL "~600 steps" is corroborated by a secondary snippet but not stated in the report text.
7. I-3 stage-1 SFT token count: 1400 steps x 33M = ~46B conflicts with Table 1's ~217B stage-1 tokens; unresolved.
8. All GPU-hour figures are derived (I-1 from FLOPs/MFU and from wall-clock bound; I-3 from node count x step time x steps); none are reported.
9. I-2 training cluster size and GPU-hours are simply not published; the blog might state contributor counts — could not be checked.
10. OpenDiLoCo numbers are from the GitHub README only; the paper (arXiv 2407.07852) was not readable.
11. INTELLECT-3.1: only a config file exists; no evidence of a released model or results.
