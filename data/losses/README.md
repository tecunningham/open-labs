# Loss curves

Per-run loss histories pulled from public W&B projects by `python -m src.fetch_wandb` (see the
docstring for the API-key requirement). One CSV per run, `_step` plus the loss keys that were
logged, downsampled to about 2,000 points, with a `.meta.json` beside it carrying the run URL.

To turn a curve into a point on the chapter's scaling curve, take the loss at the final step (or
at the step matching the tokens in `data/runs.csv`) and put it in that run's `loss` column with
`loss_eval_set` naming the metric and `confidence=reported`.

Known public sources:

| lab | where | notes |
|---|---|---|
| Ai2 OLMo 3 | `ai2-llm/Olmo-3-1025-7B`, `ai2-llm/Olmo-3-1125-32B` | linked from the OLMo-core official README |
| Ai2 OLMo 2 7B/13B | `ai2-llm` W&B | 32B is on comet.ml, not W&B |
| EleutherAI Pythia | `eleutherai/pythia` | run-to-model mapping is "rough and partial" per the repo README; 154 checkpoints per model are on Hugging Face if recomputing loss is preferred |
| Marin | W&B links inside each experiment issue on GitHub | entity/project vary by experiment; the 8B and 32B retrospectives link the flagship runs |
| Cerebras-GPT | not on W&B | final Pile losses are already in `runs.csv` from the model card |
