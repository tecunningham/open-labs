# AI R&D benchmarks in closed-lab model cards: collection notes

Compiled 2026-09-13 by three research agents working from web-search snippets only (anthropic.com, openai.com, deepmind.google and most secondary hosts were blocked by the egress policy). Curated into `data/ai_rd_benchmarks.csv`; the `confidence` column is `snippet` or `memory` for every row, since no card was read directly. Open questions below are the verification list for a session with access to the cards.

---

# GDM AI R&D benchmark compilation — notes (compiled 2026-09-13)

Environment: WebFetch/curl blocked; WebSearch only. The session's WebSearch budget was
exhausted (200/200) after ~40 queries in this task, so ~10 planned confirmation queries
were never run (listed under Open questions). Rows marked `confidence=memory` were
not confirmed by any snippet.

## Cards / reports covered (in release order)

| Model | Card date | Document | URL |
|---|---|---|---|
| Gemini 1.5 Pro | 2024 | Gemini 1.5 tech report (dangerous-capability evals; no ML R&D) | https://arxiv.org/pdf/2403.05530 |
| Gemini 2.0 Flash | card published 2025-04-15 | Gemini 2.0 Flash Model Card | https://modelcards.withgoogle.com/assets/documents/gemini-2-flash.pdf |
| Gemini 2.5 Pro Preview | 2025-04-16 (upd. 2025-05-09) | Gemini 2.5 Pro Preview Model Card | https://storage.googleapis.com/model-cards/documents/gemini-2.5-pro-preview.pdf |
| Gemini 2.5 Pro GA | 2025-06-27 | Gemini 2.5 Pro Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Pro-Model-Card.pdf |
| Gemini 2.5 (family) | 2025-06/07 | Gemini 2.5 tech report, "Frontier Safety" section | https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf (arXiv 2507.06261) |
| Gemini 2.5 Deep Think | 2025-08-01 | Gemini 2.5 Deep Think Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf |
| Gemini 2.5 Computer Use | 2025-10-07 | Gemini 2.5 Computer Use Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Computer-Use-Model-Card.pdf |
| Gemini 3 Pro | 2025-11-18 | Gemini 3 Pro Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf |
| Gemini 3 Pro | 2025-11 | Gemini 3 Pro Frontier Safety Framework Report | https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf |
| Gemini 3 Deep Think | 2025-12 (V2 Feb 2026) | No standalone card found; Gemini 3 Pro card Deep Think addendum | https://deepmind.google/models/model-cards/gemini-3-pro/ |
| Gemini 3 Flash | 2025-12-17 | Gemini 3 Flash Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf |
| Gemini 3.1 Pro | 2026-02 (Feb 19) | Gemini 3.1 Pro Model Card (full FSF eval, Deep Think focus) | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf |
| Gemini 3.1 Flash-Lite | early 2026 | Gemini 3.1 Flash-Lite Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Flash-Lite-Model-Card.pdf |
| Gemini 3.5 Flash | 2026-05 (GA May 19, I/O) | Gemini 3.5 Flash Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Model-Card.pdf |
| Gemini 3.5 Flash-Lite / 3.5 Flash Cyber | mid 2026 | Model cards (no ML R&D content) | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Lite-Model-Card.pdf |
| Gemini 3.5 Pro | NOT RELEASED (announced I/O 2026-05-19; GA slipped June/July/Jul 17) | none | — |
| Gemini 3.6 Flash | 2026-07-21 | Gemini 3.6 Flash Model Card | https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-6-Flash-Model-Card.pdf |
| Gemini 3.7 Flash | 2026-08-13 | Gemini 3.7 Flash Model Card + Gemini 3.7 Flash FSF Report (Aug 2026) | https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_fsf_report.pdf |
| Gemini 3.8 Flash (+3.8 Flash Cyber) | 2026-09-02 | Gemini 3.8 Flash Model Card | https://deepmind.google/models/model-cards/gemini-3-8-flash/ |

Framework context surfaced by search: FSF v2 (Feb 2025) used "Machine Learning R&D
Autonomy Level 1" (and Uplift Level 1); FSF v3 (Sep 2025) renamed to "ML R&D acceleration
level 1" / "automation level 1" and added harmful manipulation; FSF April-2026 update
(v3.1) added Tracked Capability Levels (TCLs) — later cards say "no T/CCLs reached".

## Headline ML R&D numbers (RE-Bench, human-normalized average)

- Gemini 2.5 Pro (preview and GA): best runs 50%–125% of best human-written solution per
  task; average below human baseline; two tasks' best runs exceeded best human reference;
  early-warning threshold not reached. Alert threshold "set above human performance".
- Gemini 3 Pro: 1.04 (as quoted in the 3.1 Pro card). FSF report methodology: 7 envs,
  16 attempts/task aggregated to a 32-hour total time budget, 95% CI from 24 runs,
  humans = 71 eight-hour attempts by 61 experts. Better than 2.5 mainly on Scaling Law
  Experiment and Optimize LLM Foundry; "substantially below the alert threshold for both
  acceleration and automation".
- Gemini 3.1 Pro (Deep Think mode): 1.27; Optimise LLM Foundry 300s -> 47s vs human 94s;
  "average across all RE-Bench challenges remains below the alert threshold"; Zvi:
  GDM says model "too inconsistent to qualify".
- Gemini 3.5/3.6/3.7/3.8 Flash: no new RE-Bench number surfaced; cards defer to
  3.1 Pro (3.5, 3.6) or to 3.7 Flash (3.8). 3.7 Flash has its own FSF report whose
  ML R&D content was not visible in snippets.

## Queries that worked (produced usable snippets)

- "Gemini 3.1 Pro model card RE-Bench 1.27 Deep Think ML R&D"
- "Gemini 3 Pro Frontier Safety Framework report RE-Bench score machine learning R&D"
- "\"Gemini 3 Pro\" FSF report RE-Bench \"human-normalized\" alert threshold hours time budget"
  (gave the 16 attempts / 32-hour / 24-run methodology)
- "Gemini 2.5 Pro model card Frontier Safety machine learning R&D CCL RE-Bench"
- "Gemini 2.5 Pro Preview model card \"Figure 4\" RE-Bench \"Key results\" ... METR methodology"
- "Gemini 3 Pro model card SWE-bench Verified 76.2 Terminal-Bench 2.0 54.2 LiveCodeBench Pro Elo"
- "Gemini 3.1 Pro SWE-bench Verified 80.6 Terminal-Bench 2.0 68.5 LiveCodeBench Pro 2887"
- "Gemini 2.5 Pro model card SWE-bench Verified 63.8 67.2 Aider Polyglot 82.2 LiveCodeBench"
- "Gemini 3.5 Flash model card May 2026 frontier safety machine learning R&D RE-Bench"
  (revealed the 3.5/3.6/3.7/3.8 Flash card series)
- "Gemini 3.6 Flash model card frontier safety ..." / "Gemini 3.7 Flash model card frontier safety ..."
- "Gemini 3.8 Flash model card September 2026 frontier safety CCL ... Terminal-Bench 2.1 90.8"
- "\"Gemini 3.5 Pro\" release model card September 2026" (confirmed not shipped)
- "Zvi \"Gemini 3\" model card safety framework RE-Bench ... acceleration automation"
- "\"Gemini 3 Deep Think\" \"model card\" storage.googleapis.com deepmind-media"
  (gave the "Deep Think mode ... consistent with the original Gemini 3 Pro safety assessment" line)
- "Gemini 1.5 Pro technical report frontier safety dangerous capability evaluations ..."

Queries that did not help: anything asking for the Gemini 2.0 Pro Experimental card's
FSF text; "Gemini 3 Deep Think model card December 2025 ..." (only launch coverage);
Gemini 3.7 Flash FSF report ML R&D content (report indexed but ML R&D lines not in snippet).

## Open questions / unconfirmed numbers

1. Gemini 3 Pro FSF report: per-task RE-Bench scores (only in a figure); whether 1.04 was
   computed under the 32-hour/16-attempt protocol; whether any internal ML R&D task suite
   results were reported (none surfaced).
2. Gemini 3.1 Pro card: time budget / attempts behind the 1.27 Deep Think figure; whether
   a non-Deep-Think RE-Bench figure was given; exact "inconsistent" wording.
3. Gemini 3.7 Flash FSF report (Aug 2026): does it contain a new RE-Bench / ML R&D score
   or TCL determination for ML R&D? Not visible in snippets.
4. Gemini 2.5 Deep Think card: ML R&D section wording (recorded from memory).
5. Gemini 2.5 Computer Use card: FSF deferral wording (memory).
6. Gemini 2.0 Flash card: CCL determination wording (memory); no Gemini 2.0 Pro card found.
7. Gemini 2.5 Pro single-attempt SWE-bench Verified 59.6% and 03-25 63.8% (memory).
8. Gemini 3 Flash Terminal-Bench 2.0 47.6% (memory); its FSF deferral wording (memory).
9. Gemini 3.7 Flash Terminal-Bench 2.1: 81.6% (two sources) vs 85.8% (one) — unresolved.
10. Gemini 3.5 Flash SWE-bench Verified 78.8 comes from an aggregator, not confirmed as card.
11. Gemini 3.5 Flash card date: snippet says PDF dated May 1 2026 vs May 19 launch.
12. "Gemini Pro received 73% on the RE-Bench" (Trustible) — provenance unknown; ignored.
13. SWE-bench Verified for Gemini 3.6/3.7/3.8 Flash cards: cards appear to have switched
    to SWE-Bench Pro / DeepSWE; Verified figures not surfaced.

---

# OpenAI system cards — AI R&D / self-improvement benchmark scrape notes

Compiled 2026-09-13. Companion file: `openai.csv` (124 rows; 103 tagged `snippet`, 21 tagged `memory`; 51 rows carry a numeric score).

Environment constraint: WebFetch/curl blocked for all relevant hosts; WebSearch only. The session-wide WebSearch budget (200) was exhausted after ~70 queries from this task, so a final planned batch (Astra numbers, GPT-5.6 revised-suite numbers, GPT-5.2/5.2-Codex/5.3-Codex/5.4 per-benchmark values, o3/o4-mini, GPT-4.5, deep research, o1-preview, gpt-oss values) could not run. Those gaps are listed under Open questions.

## Cards covered (chronological)

| # | Card | Date | URL |
|---|------|------|-----|
| 1 | GPT-4o System Card | 2024-08-08 | https://cdn.openai.com/gpt-4o-system-card.pdf |
| 2 | OpenAI o1 System Card (o1-preview / o1-mini) | 2024-09-12 | https://cdn.openai.com/o1-system-card.pdf |
| 3 | OpenAI o1 System Card | 2024-12-05 | https://cdn.openai.com/o1-system-card-20241205.pdf (arXiv 2412.16720) |
| 4 | OpenAI o3-mini System Card | 2025-01-31 | https://cdn.openai.com/o3-mini-system-card-feb10.pdf |
| 5 | Deep Research System Card | 2025-02-25 | https://cdn.openai.com/deep-research-system-card.pdf |
| 6 | GPT-4.5 System Card | 2025-02-27 | https://openai.com/index/gpt-4-5-system-card/ |
| 7 | OpenAI o3 and o4-mini System Card | 2025-04-16 | https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf |
| 8 | Addendum to o3/o4-mini card: Codex (codex-1) | 2025-05-16 | https://cdn.openai.com/pdf/8df7697b-c1b2-4222-be00-1fd3298f351d/codex_system_card.pdf |
| 9 | ChatGPT Agent System Card | 2025-07-17 | https://cdn.openai.com/pdf/6bcccca6-3b64-43cb-a66e-4647073142d7/chatgpt_agent_system_card_launch.pdf |
| 10 | gpt-oss-120b & gpt-oss-20b Model Card | 2025-08-05 | https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf (arXiv 2508.10925) |
| 11 | GPT-5 System Card | 2025-08-07 (rev. 08-13) | https://cdn.openai.com/gpt-5-system-card.pdf (arXiv 2601.03267) |
| 12 | Addendum to GPT-5 card: GPT-5-Codex | 2025-09-15 | https://cdn.openai.com/pdf/97cc5669-7a25-4e63-b15f-5fd5bdc4d149/gpt-5-codex-system-card.pdf |
| 13 | GPT-5.1 Instant & Thinking System Card Addendum | 2025-11-12 | https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf |
| 14 | GPT-5.1-Codex-Max System Card | 2025-11-18 | https://cdn.openai.com/pdf/2a7d98b1-57e5-4147-8d0e-683894d782ae/5p1_codex_max_card_03.pdf |
| 15 | Update to GPT-5 System Card: GPT-5.2 | 2025-12-11 | https://cdn.openai.com/pdf/3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf |
| 16 | Addendum to GPT-5.2 card: GPT-5.2-Codex | 2025-12-18 | https://cdn.openai.com/pdf/ac7c37ae-7f4c-4442-b741-2eabdeaf77e0/oai_5_2_Codex.pdf |
| 17 | GPT-5.3-Codex System Card | 2026-02-05 | https://cdn.openai.com/pdf/23eca107-a9b1-4d2c-b156-7deb4fbc697c/GPT-5-3-Codex-System-Card-02.pdf |
| 18 | GPT-5.4 Thinking System Card | 2026-03-05 | https://deploymentsafety.openai.com/gpt-5-4-thinking/gpt-5-4-thinking.pdf |
| 19 | GPT-5.5 System Card | 2026-04-23 | https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf |
| 20 | GPT-5.5 Instant System Card | 2026-05-04 | https://deploymentsafety.openai.com/gpt-5-5-instant/gpt-5-5-instant.pdf |
| 21 | GPT-5.6 Preview System Card (Sol/Terra/Luna) | 2026-06-25 | https://deploymentsafety.openai.com/gpt-5-6-preview/gpt-5-6-preview.pdf |
| 22 | GPT-5.6 System Card | 2026-07-09 | https://deploymentsafety.openai.com/gpt-5-6/gpt-5-6.pdf |
| 22b | GPT-5.6 August Updates | 2026-08-06 | https://cdn.openai.com/pdf/GPT_5_6_August_Updates.pdf (not mined) |
| 23 | GPT-6 Astra System Card | 2026-09-03 | https://deploymentsafety.openai.com/gpt-6-astra |

Not covered / not applicable: Operator (Jan 2025) and o3 Operator addendum (computer-use; no self-improvement evals); GPT-4.1 (no system card); "GPT-5.4" non-Thinking variants (single card); "o5" does not exist as a card — the post-o4 line was folded into GPT-5.x.

Naming shift: cards up to GPT-4.5 use "Model autonomy" (Low/Medium/High/Critical risk). From o3/o4-mini (Preparedness Framework v2, Apr 2025) the category is "AI Self-improvement" with a High capability threshold defined (per 5.4/5.5 cards) as "equivalent to a performant mid-career research engineer" and Critical as fully automating self-improvement. Every card through GPT-5.6 states the model is below High for AI self-improvement; Astra's designation was not confirmed.

## Headline numeric trajectory recovered from snippets

| Model | SWE-bench Verified | SWE-Lancer IC Diamond | OpenAI PRs | MLE-bench-30 (bronze+) | PaperBench(-10) | OpenAI-Proof Q&A | METR 50% horizon |
|---|---|---|---|---|---|---|---|
| o1-preview (Dec card) | 41.3% (Agentless, 5 tries) | – | – | 37% pass@10 (full MLE-bench) | – | – | – |
| o1 | 40.9% | – | – | 27% pre / 24% post pass@10 | – | 36% agentic tasks | ~human 2h (adapted scaffold) |
| o3-mini | 39% Agentless / 61% internal tools | – | ~0% ("failed entirely") | "doesn't impress" | – | 27% agentic tasks; RE interview 93% coding / 80% MCQ | – |
| o3 / o4-mini | 69.1% / 68.1% | – | – | – | ~24% (ambiguous) | – | 1h30m (o3) |
| ChatGPT agent | ~o3 | – | – | 9% | highest (w/ browsing) | – | – |
| gpt-5-thinking | 74.9% | no improvement | (45%?) | 8% | 24% (from 22%) | 2% | 2h17m |
| GPT-5.1 | – | 67%* | 45%* | 12% | 34% | 2% | – |
| GPT-5.1-Codex-Max | 77.9% | 80% | 53% | 17% | 40% | 8% | 2h40m |
| GPT-5.2 | 80.0% (memory) | (40/237 omitted) | – | 12.2%† | – | – | ~6.6h |
| GPT-5.3-Codex | – | 81.4% | – | not reported | – | – | – |
| GPT-5.4 Thinking | – | – | – | 23% | – | 5.8% | – |
| GPT-5.5 | – | – | – | 37% | – | 1.7% | – |
| GPT-5.6 Sol | – | – | – | 48.0% (revised suite) | – | – | not robust (cheating) |
| GPT-6 Astra | – | – | – | – | – | – | – |

\* "from 67%" / "from 45%" baselines in the Codex-Max card may refer to GPT-5.1-Codex rather than GPT-5.1 Thinking.
† 12.2% attributed to GPT-5.2 by inference (Zvi: 5.4 "moved from 12.2% to 23%, though this test wasn't reported by GPT-5.3-Codex").

New evals appearing in 2026 cards: Internal Research Debugging Eval (GPT-5.5 median 50.5%, not significantly above 5.4), Monorepo-Bench (5.4/5.5, "little improvement"), and a "revised suite" including "MLE-bench (revised)" from GPT-5.6 onward.

## Queries that worked (snippet yielded numbers)

- `o1 system card SWE-bench Verified MLE-bench "OpenAI Research Engineer Interview" agentic tasks scores` → 41.3 / 40.9 / 37% pass@10 / 18 coding + 97 MCQ.
- `o3-mini system card "SWE-Lancer" "OpenAI PRs" "MLE-bench" "PaperBench" percent` → 39% Agentless / 61% internal tools.
- `o3-mini system card January 2025 "OpenAI PRs" "MLE-bench" "OpenAI Research Engineer Interview" ...` (surfaced Zvi "o3-mini Early Days") → 93% coding, 80% MCQ, agentic 27% vs 36%, PRs failed, o1 48% w/ tools.
- `GPT-5 system card "OpenAI PRs" "PaperBench" "MLE-bench" gpt-5-thinking percent AI self-improvement high` → 8% MLE-30, 22→24% PaperBench, no SWE-Lancer improvement, "did not meet High".
- `GPT-5.4 Thinking system card OpenAI PRs 60% MLE-bench PaperBench OpenAI-Proof Q&A results March 2026` → OPQA 20 problems, gpt-5-thinking 2%.
- `Zvi GPT-4.5 system card SWE-Lancer OpenAI PRs MLE-bench PaperBench deep research` (domain-restricted to thezvi/lesswrong) → Codex-Max deltas: SWE-Lancer 67→80, PaperBench-10 24→34→40, MLE-30 8→12→17, PRs 45→53, OPQA 2→8.
- `Zvi "GPT 5.5: The System Card" ...` → OPQA 5.8→1.7, MLE-30 23→37.
- `Zvi GPT-5.4 system card self-improvement ...` → MLE-30 12.2→23; 5.3-Codex didn't report MLE.
- `Zvi "GPT-5.6: The System Card" MLE-bench ...` → Sol 48.0% MLE-bench; revised suite.
- `GPT-5.5 system card "AI Self-improvement" "High" threshold ...` → "performant mid-career research engineer"; Internal Research Debugging Eval 50.5%.
- `OpenAI system card METR time horizon 50% GPT-5 GPT-5.1-Codex-Max GPT-5.2 hours cited` → 2h40m; `METR evaluation OpenAI GPT-5.2 ... metr.org` → 2h17m, o3 1h30m, Sol cheating; `Zvi GPT-5.2 system card ...` → 6.6h.
- `"GPT-6" OpenAI system card 2026 Astra release date self-improvement` → Astra exists, Sep 3 2026, Critical cyber.
- `GPT-5.5 Instant system card GPT-5.6 Preview system card AI self-improvement evaluations` → both below High; revised suite statement.
- `codex-1 Codex system card May 2025 ...` → 72.1% / 83.8% (secondary), SAG not-High statement.
- `GPT-5.3-Codex system card February 2026 ...` → SWE-bench Pro 56.8/56.4/55.6, SWE-Lancer 81.4.
- Restricting `allowed_domains` to `deploymentsafety.openai.com` was useful for discovering which cards exist (5.3-Codex, 5.4 Thinking, 5.5, 5.5 Instant, 5.6 Preview, 5.6, Astra) but the hub pages' snippets rarely include numbers.

Queries that did not work: anything asking for GPT-4.5 / deep research / o3-o4-mini / gpt-oss / GPT-5.2 / 5.2-Codex per-benchmark values — snippets return only benchmark descriptions from the PDFs' boilerplate. Zvi's "On GPT-4.5", "We're in Deep Research", "o3 Will Use Its Tools For You" and "The Codex of Ultimate Vibing" posts were found but snippets did not expose the numbers.

## Open questions (numbers not confirmed)

1. **GPT-6 Astra**: AI Self-Improvement designation (High or not?) and every per-benchmark value; METR Astra report / time horizon. Highest-priority gap.
2. **GPT-5.6 (Sol/Terra/Luna)**: composition of the "revised suite" and all values other than Sol MLE-bench 48.0%; meaning of "above the threshold" in that snippet; August Updates doc.
3. **GPT-5.5 / 5.4 Thinking**: SWE-bench Verified, SWE-Lancer Diamond, OpenAI PRs, PaperBench values; Monorepo-Bench and 5.4 Internal Research Debugging Eval values; whether 5.5 card actually re-ran these or copied 5.4 (Zvi suggests copy).
4. **GPT-5.3-Codex, GPT-5.2-Codex, GPT-5.2**: OpenAI PRs, PaperBench-10, SWE-Lancer (197-task variant), OPQA values. GPT-5.2 SWE-bench Verified 80.0% is memory from launch blog, not the card.
5. **GPT-5.1 Thinking vs GPT-5.1-Codex**: which model the Codex-Max card's 67% (SWE-Lancer) and 45% (PRs) baselines refer to; gpt-5-thinking's own OpenAI PRs value.
6. **GPT-5-Codex addendum**: self-improvement values.
7. **gpt-oss**: OpenAI PRs, PaperBench values (card includes them); SWE-bench Verified 62.4% from launch.
8. **ChatGPT agent**: PaperBench-with-browsing value; SWE-Lancer / PRs values.
9. **o3 / o4-mini**: SWE-Lancer $ earned, OpenAI PRs, MLE-bench, PaperBench (24% snippet is a secondary source with ambiguous attribution).
10. **GPT-4.5**: all values except SWE-bench Verified 38% (memory). Card includes SWE-Lancer, OpenAI PRs, MLE-bench, RE interviews, agentic tasks.
11. **deep research**: all values (SWE-bench Verified, SWE-Lancer, PRs, MLE-bench, RE interviews, agentic tasks).
12. **o1-preview Sep 2024 card**: SWE-bench Verified / MLE-bench / RE interview / agentic-task values as printed in that version (Dec card superseded them).
13. **GPT-4o**: RE interview and agentic-task values; METR summary statistic.
14. **o1 Dec card**: per-task agentic pass rates (OpenAI API proxy, Mistral 7B in Docker, buy GPU…), absolute RE-interview values, o1-mini SWE-bench Verified.
15. **SWE-Lancer dollar figures** were never surfaced for any card (cards report $ earned and % on IC SWE Diamond; only percentages surfaced).
16. Whether the GPT-5.2 METR 6.6h figure is cited in the GPT-5.2 card itself or only in METR's own report.

Suggested follow-up when fetch access is available: pull the PDFs listed above (cdn.openai.com / deploymentsafety.openai.com) and read the "AI Self-improvement" (or "Model autonomy") sections directly; each has a summary table with all values.

---

# Anthropic system cards: AI R&D benchmark compilation notes

Compiled 2026-09-13 via WebSearch snippets only (WebFetch/curl blocked; session search budget of 200 calls was exhausted after ~72 queries in this task, so the last planned verification batch of 16 queries did not run). CSV: `anthropic.csv` (114 rows; 103 tagged `snippet`, 11 tagged `memory`).

## Cards covered (in order)

| # | Model(s) | Card date | Card | URL status |
|---|---|---|---|---|
| 1 | Claude 3 Opus | Mar 2024 | Model Card and Evaluations for Claude Models | confirmed CDN URL; no AI R&D evals |
| 2 | Claude 3.5 Sonnet | Jun 2024 | Claude 3.5 Sonnet Model Card Addendum | confirmed |
| 3 | Claude 3.5 Sonnet (upgraded) + 3.5 Haiku | Oct 2024 | Model Card Addendum: Claude 3.5 Haiku and Upgraded Claude 3.5 Sonnet | confirmed |
| 4 | Claude 3.7 Sonnet | Feb 2025 | Claude 3.7 Sonnet System Card | confirmed |
| 5 | Claude Opus 4 / Sonnet 4 | May 2025 | System Card: Claude Opus 4 & Claude Sonnet 4 | confirmed |
| 6 | Claude Opus 4.1 | Aug 2025 | System Card Addendum: Claude Opus 4.1 | confirmed |
| 7 | Claude Sonnet 4.5 | Sep 2025 | System Card: Claude Sonnet 4.5 | anthropic.com slug confirmed |
| 8 | Claude Haiku 4.5 | Oct 2025 | System Card: Claude Haiku 4.5 | anthropic.com slug confirmed |
| 9 | Claude Opus 4.5 | Nov 2025 | System Card: Claude Opus 4.5 | confirmed |
| 10 | Claude Opus 4.6 | Feb 2026 | System Card: Claude Opus 4.6 | confirmed (several CDN hashes exist) |
| 11 | Claude Sonnet 4.6 | Feb 17 2026 | System Card: Claude Sonnet 4.6 | confirmed |
| 12 | Claude Mythos Preview | Apr 2026 | Claude Mythos Preview System Card (244 pp; model not released, Project Glasswing) | **URL is a best guess** |
| 13 | Claude Opus 4.7 | Apr 16 2026 | System Card: Claude Opus 4.7 (232 pp) | confirmed |
| 14 | Claude Opus 4.8 | May 28 2026 | System Card: Claude Opus 4.8 | confirmed |
| 15 | Claude Fable 5 & Claude Mythos 5 | Jun 9 2026 | System Card: Claude Fable 5 & Claude Mythos 5 | **URL is a best guess** |
| 16 | Claude Sonnet 5 | Jun 30 2026 | System Card: Claude Sonnet 5 (144 pp) | confirmed |
| 17 | Claude Opus 5 | Jul 24 2026 | System Card: Claude Opus 5 | confirmed |
| 18 | Claude Fable 5.1 & Claude Mythos 5.1 | Sep 1 2026 | System Card: Claude Fable 5.1 & Claude Mythos 5.1 (212 pp) | confirmed |

No Claude Haiku 4.6 exists (search confirmed). Fable 5.1 and Mythos 5.1 share identical weights and differ only in safeguards; Mythos 5 / Mythos 5.1 are the lighter-safeguard partner-only configurations.

Threshold naming: cards through Opus 4.6 use "AI R&D-4" (RSP v2.2). From Opus 4.7 / Mythos Preview onward the card uses a revised "automated AI R&D capability threshold" (operationalized as: can fully substitute for research scientists/engineers, or produces a sustained AI-attributable 2x acceleration in the pace of progress). Sonnet 5, Opus 5 and Opus 4.8 cards did not run new internal surveys and point back to Mythos 5 sec. 2.3 / Mythos Preview sec. 2.3 / Opus 4.7 sec. 2.3.5.

## Queries that worked best

- `Claude Opus 4 system card "Internal AI Research Evaluation Suite 1" LLM training quadruped RL novel compiler score` -> surfaced per-task Opus 4 / Opus 4.1 numbers and the Opus 5 card statement that only unsaturated/unbounded tasks are still reported.
- `Claude Opus 4.5 system card "Internal AI Research Evaluation Suite 2" score threshold 0.6 AI R&D-4 rule out` -> 0.604 vs 0.6; 0/18 survey.
- `Zvi "Opus 4.5" model card "RE-Bench" "Suite 1" "18 participants" uplift survey productivity percent` -> median 100% / mean 220%.
- `Claude 4 system card AI R&D "internal survey" researchers productivity uplift Opus 4 "AI R&D-4"` -> Opus 4.6 survey (30-700%, mean 152%, median 100%; 11/3/2 of 16 on L4 replacement).
- `Claude Sonnet 4.5 system card AI R&D "Internal AI research evaluation suite 2" score` -> Opus 4.6 Suite 2 = 0.6124.
- `Zvi "Sonnet 4.5" system card "AI R&D" internal survey "Suite" kernel "LLM training" speedup "AI R&D-4"` -> Sonnet 4.5 LLM training 5.5x (first over 4x expert line); list of Suite 1 task sections.
- `Mythos Preview system card AI R&D "Suite 1" OR "Suite 2" ... kernel 399x` -> 399x kernel, 52x LLM training, Table 2.3.3.A (4h/8h all tasks, 40h on 2/3), 1/18 L4 survey, geometric-mean ~4x uplift.
- `Opus 4.6 system card "Suite 1" ... "RE-Bench"` -> 427x kernel (novel scaffold; 300x ~ 40 expert-hours), 34x LLM training vs 4x human line, 20.96 / 21.99 vs threshold 12.
- `kingy "Claude 3.7 Sonnet" system card summary AI R&D "kernel" "RE-Bench" "32 hours" human baseline` -> 32h budget ~ median human 8h on 5-task RE-Bench subset.
- Vellum "benchmarks explained" posts were reliable for SWE-bench Verified / Terminal-Bench numbers for Opus 4.5-4.8, Sonnet 5, Fable 5/Mythos 5.
- Zvi (thezvi.substack / wordpress / lesswrong) writeups exist for every card from 3.7 onward and were the best snippet source for AI R&D section details.

Queries that did not help: anything asking for RE-Bench per-task names ("Optimize LLM foundry", "Fix embedding", "Scaling law experiment") in Anthropic cards; "PaperBench" in Anthropic cards (no hits; PaperBench does not appear to be reported by Anthropic); 3.7 Sonnet Suite 1 per-task numbers.

## Memory-tagged rows (11)

- Opus 4 SWE-bench Verified high-compute 79.4; Opus 4 Terminal-bench high-compute 50.0; Sonnet 4 SWE-bench high-compute 80.2; Sonnet 4 Terminal-bench 35.5 (41.3 high-compute); Opus 4.1 Terminal-bench 43.3; Opus 4.5 Terminal-Bench 2.0 59.3.
- Opus 4 AI R&D-4 wording; Sonnet 4 Suite 1 (no numbers); Sonnet 4.5 AI R&D-4 wording; Sonnet 4.6 determination; Claude 3 Opus "no evals" row.

## Open questions / numbers not confirmed

1. **Claude 3.7 Sonnet Internal AI research evaluation suite** per-task results (kernel, time-series, text-based RL, LLM training) - card has them; not surfaced. Also exact subset size for the 70.3% custom-scaffold SWE-bench figure.
2. **Claude 4 (Opus 4 / Sonnet 4) Internal AI Research Evaluation Suite 2** score and any internal-use survey; Sonnet 4 per-task Suite 1 numbers.
3. **Opus 4.1** kernel-optimization speedup (card says slightly below Opus 4's 72.65x); whether the novel-compiler 74.4% / 6.81% figures are truly identical to Opus 4 or carried over in the snippet.
4. **Sonnet 4.5** Suite 2 score, other Suite 1 per-task numbers, and "Internal model evaluation and use survey" results.
5. **Opus 4.5** numeric Suite 1 values (kernel speedup, LLM-training speedup, quadruped, text-based RL); Terminal-Bench 2.0 59.3% is memory only.
6. **Opus 4.6** which Suite 1 task the 20.96 / 21.99 (threshold 12 = 4 human-effort hours) scores belong to (tentatively text-based RL); time-series / quadruped values; whether the card reports RE-Bench at all. Reconcile the "0 of 16" (Opus 4.7 card's summary of the 4.6 survey) vs "11 / 3 / 2 of 16" (Opus 4.6 card) framings.
7. **Sonnet 4.6** AI R&D section contents (likely "does not advance frontier beyond Opus 4.6"; no numbers found).
8. **Mythos Preview** exact card URL; Suite 2 numeric score if any (table is threshold-based); full survey counts; RE-Bench mention.
9. **Opus 4.7** survey counts (Zvi: "most not expecting a full drop-in L4; ~50/50 on weeklong ambiguous tasks"); Suite 1 numbers.
10. **Fable 5 / Mythos 5** card URL; survey counts; internal acceleration measure value; Terminal-Bench 2.1 for Mythos 5.
11. **Sonnet 5 SWE-bench Verified**: conflicting 72.7% (likely garbled - equals Sonnet 4's number) vs 85.2%. Left blank.
12. **Opus 5 SWE-bench Verified**: conflicting 96.0% (several secondary sites) vs "Anthropic did not report SWE-bench for Opus 5" (llm-stats, aireleasetracker). Left blank. Terminal-Bench 2.1 89.1% may be third-party rather than card-reported.
13. **Fable 5.1 / Mythos 5.1**: Anthropic reportedly published no SWE-bench Verified; survey / uplift numbers and Suite results not surfaced; METR tasks (Sunlight, Budget NanoGPT Speedrun, LMCA) only qualitative.
14. Whether any Anthropic card reports **PaperBench** (no evidence found) or a formal **AI R&D uplift RCT** (only surveys and the separate Jun 2026 "When AI builds itself" report: >80% of merged production code by Claude in May 2026; 130-staff poll median ~4x output with Mythos Preview; 8x code merged per engineer/day vs 2024, flagged by Anthropic as an overstatement).
15. Terminal-bench harness drift: Sonnet 4.5 announcement lists Opus 4.1 at 46.5% (Terminus 2) vs 43.3% at Opus 4.1 launch; benchmark versions move 1.0 -> 2.0 (Opus 4.5) -> 2.1 (Opus 4.8) -> 4.0 / -Science (Fable 5.1), so Terminal-bench rows are not directly comparable across cards.
