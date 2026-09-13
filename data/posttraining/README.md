# Post-training experiment data

`marin_sft_scaling.csv`: Marin issue 3956 (April to May 2026), supervised fine-tuning of the Marin
8B and 32B bases on OpenThoughts-4 math subsets (256 to 50k examples) with reasoning traces from
eight teacher models, three training repeats per cell, evaluated with evalchemy on AIME24, AIME25,
AMC23, HMMT and MATH500 (means over 3 to 10 sampling seeds). One row per (base, teacher,
examples, repeat, checkpoint step, benchmark); `wandb_run` is the evalchemy run id in
`marin-community/marin`. Rebuild with `python -m src.fetch_evalchemy`.

SFT compute is not logged anywhere; the chapter converts examples to FLOPs at an assumed 10,000
tokens per example (6ND). The `_pt2` / `_pt3` suffixes in the W&B names are the training repeats
(`repeat` 2 and 3 here).

`../experiments_data/marin_delphi_ladder.csv`: the released plot data behind Marin's Delphi
scaling suite (`marin-community/delphi-blog-data`, config `delphi-ladder`, 102 rows): 81 IsoFLOP
runs at seven budgets with params, tokens, loss (Paloma/Nemotron eval mix, not c4_en) and total
GFLOPs; the seven fitted optima; and the 1e21/1e22/1e23 held-out runs with seeds.
