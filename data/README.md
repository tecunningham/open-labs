# Data

Hand-curated. Every row carries a `source_url`. Numeric cells are plain numbers (`8e9`, not `8B`).

| file | one row per | key columns |
|---|---|---|
| `runs.csv` | training run (experiment or final) | `run_type` in ladder / ablation / final / midtrain / aborted / unknown; `stage` in pretraining / midtraining / post-training/RL; `confidence` in reported / derived / guess |
| `experiments.csv` | published scaling experiment or ablation series | `raw_data_available` in logs / figure_only / numbers_in_text |
| `labs.csv` | lab × project | headcount, dates, total compute, hardware, funding |
| `claims.csv` | verbatim quantitative statement by a lab | quote and citation, to compare against reconstructed numbers |
| `ai_rd_benchmark_series.csv` | benchmark series in the appendix figures (lab x benchmark, with version and scale splits) | `ceiling` (blank = unbounded), `direction` higher/lower, `human_ref` and `threshold` with labels (drawn as red dotted and grey dashed lines), `category`, `overview` yes/no for the shared 0 to 100 percent panel, `deprecated_date` and `deprecated_note` for the card at which the lab stopped reporting the series (drawn as a black cross) |
| `ai_rd_benchmarks.csv` | AI R&D benchmark score in a closed-lab model or system card (GDM, OpenAI, Anthropic) | `frontier` yes/no picks the columns shown in the appendix; `confidence` in reported (card or the METR report it cites) / announcement (launch post) / snippet / memory; `family` groups benchmarks; `conditions` holds scaffold, subset and budget |

`flops_method`: `reported`, `6ND` (6 × active params × tokens), or `gpu_hours×peak×MFU` with the assumption in `notes`.
