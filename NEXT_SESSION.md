# Next session: pull loss curves and verify claims

Run this in a cloud session whose environment allows `wandb.ai`, `api.wandb.ai`, `arxiv.org`,
`huggingface.co`, `www.comet.com` and has `WANDB_API_KEY` set (see README, "Environment caveat").

1. `pip install wandb` then `python -m src.fetch_wandb --list ai2-llm/Olmo-3-1025-7B` to confirm
   the key works and to read the real loss metric names; pin them in `src/fetch_wandb.py` TARGETS.
2. `python -m src.fetch_wandb ai2-olmo`, then `eleutherai`. For Marin, open the 8B and 32B
   retrospectives in `docs/reports/` of marin-community/marin, copy the W&B run URLs into TARGETS,
   then fetch.
3. For each fetched run, write the final-step loss into `data/runs.csv` (`loss`, `loss_eval_set`,
   `confidence=reported`) for the matching `run_id`, and add ladder-run losses where the runs
   exist (OLMo ladder: `allenai/OLMo-ladder` checkpoints and W&B).
4. Re-render (`quarto render`) and check that the OLMo, Marin and Pythia chapters now show a
   fitted scaling curve instead of the "no loss values" note.
5. Work through the "Verification status" lists in `data/notes/*.md` against arXiv directly, and
   flip `confidence` from `guess` to `reported` where confirmed.
6. Commit the CSVs. Never commit the key; it lives only in the environment's variables.
