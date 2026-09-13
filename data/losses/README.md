# Loss curves

Per-run loss histories pulled from public W&B projects by `python -m src.fetch_wandb` (one lab
name as the argument, or none for all). One CSV per `runs.csv` row: `_step`, the loss keys that
were logged, and `_segment` (the W&B run id the point came from, since Ai2 logs each restart as a
new run). Restart segments are concatenated and sorted by step, then thinned to about 3,000
points; eval-loss points are always kept. A `.meta.json` beside each CSV carries the W&B URLs,
tokens per step where the batch size was fixed, and the final-step value of every key.

`python -m src.fetch_wandb --apply` writes the final losses into `data/runs.csv` (`loss`,
`loss_eval_set`, and a "Loss from W&B" tag in `notes`). `--summary` prints them.

`src/losses.py` reads the CSVs back and converts W&B steps to cumulative training tokens with a
per-run batch-size schedule (`MARIN_8B`, `MARIN_32B`), so a multi-phase run can be drawn as one
trajectory (`plots.loss_trajectory`, used in the Marin chapter). Add a schedule and a phase spec
there for any other lab whose phases are separate `runs.csv` rows.

## Access

W&B's GraphQL endpoint serves public projects **anonymously**. In the 2026-09-13 session a
personal API key was refused for every public project ("the provided API key cannot access this
resource") while the same queries worked with no credentials, so the script sends none. PyPI was
blocked, so it uses `requests` rather than the `wandb` package. Two hosts stay out of reach:
`stanford-mercury/marin` (the 8B retrospective's report links; private) and comet.ml (OLMo 2 32B;
the REST API needs a key even for shared reports).

## Which metric is the `loss` column

| lab | key | why |
|---|---|---|
| Ai2 OLMo | `train/CE loss` (Olmo 3) or `train/CrossEntropyLoss` (OLMo 2), median of the last 20 sampled points | the only loss both generations log; Olmo 3 stage 1 also logs in-loop `eval/lm/*` CE losses (c4_en, pile, dolma, wikitext), kept in the CSVs and noted in `runs.csv`; OLMo 2 logs no eval loss to W&B. Train loss is on the training mix, so Olmo 2 and Olmo 3 values are not one curve, and midtraining losses (anneal mix) are lower still. |
| EleutherAI Pythia | `validation/lm_loss` at the last step | Pile validation loss from the GPT-NeoX trainer; for the deduplicated models the validation set is the deduplicated Pile |
| Marin | `eval/paloma/c4_en/loss` at the last step | the curve the retrospectives plot; Llama 3 tokenizer, nats per token. `train/loss`, `eval/loss`, `eval/macro_loss` are in the CSVs |

## Run mapping

Pinned in `TARGETS` in `src/fetch_wandb.py` with a note per row. Highlights:

- **Olmo 3**: stage 1 is 39 (7B) and 69 (32B) restart segments in one W&B group each; the
  script pulls the group. 7B: 4,194,304 tokens per step, final step 1,413,814 = 5.93T. 32B:
  8,388,608 per step, final step 678,999 = 5.70T (the released midtraining branched at 656,000).
- **OLMo 2**: 11 (7B) and 75 (13B) segments; anneal ingredients stored as parallel segments.
- **Pythia**: the 143k-step runs whose config (layers, width, batch 1024 x 2048, Pile vs
  deduplicated Pile path) matches the released model. 13 of 16 mapped.
- **Marin**: run *names* are stable, display names are not (`llama-32b-tootsie-2` is shown as
  "Marin 32B run1"). 8B batch schedule 1024 -> 3072 -> 4096 sequences of 4096; 32B 8192 x 4096.

| lab | where | notes |
|---|---|---|
| Ai2 OLMo 3 | `ai2-llm/Olmo-3-1025-7B`, `ai2-llm/Olmo-3-1125-32B` | linked from the OLMo-core official README |
| Ai2 OLMo 2 7B/13B | `ai2-llm/OLMo-2-1124-7B`, `ai2-llm/OLMo-2-1124-13B` | 32B is on comet.ml, not pulled |
| Ai2 ladder | `ai2-llm/olmo-ladder` | not public; ladder rows still have no loss |
| EleutherAI Pythia | `eleutherai/pythia` | 10,352 runs; `eleutherai/pythia-extra-seeds` has seed replicates of the small models |
| Marin | `marin-community/marin` | 113k runs; filter on `name` |
| Cerebras-GPT | not on W&B | final Pile losses are already in `runs.csv` from the model card |
