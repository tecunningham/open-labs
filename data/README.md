# Data

Hand-curated. Every row carries a `source_url`. Numeric cells are plain numbers (`8e9`, not `8B`).

| file | one row per | key columns |
|---|---|---|
| `runs.csv` | training run (experiment or final) | `run_type` in ladder / ablation / final / midtrain / aborted / unknown; `stage` in pretraining / midtraining / post-training/RL; `confidence` in reported / derived / guess |
| `experiments.csv` | published scaling experiment or ablation series | `raw_data_available` in logs / figure_only / numbers_in_text |
| `labs.csv` | lab × project | headcount, dates, total compute, hardware, funding |
| `claims.csv` | verbatim quantitative statement by a lab | quote and citation, to compare against reconstructed numbers |

`flops_method`: `reported`, `6ND` (6 × active params × tokens), or `gpu_hours×peak×MFU` with the assumption in `notes`.
