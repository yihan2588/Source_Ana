# Source_Ana Repository Overview

The repository contains four Python modules:

| Module | Role |
| --- | --- |
| `main.py` | command-line interface, runs the full involvement pipeline |
| `involvement.py` | loads SourceRecon CSV files, applies subject-specific thresholds, computes wave metrics |
| `stat.py` | aggregates wave data, performs Wilcoxon and Mann–Whitney tests, reports percentage changes |
| `visual.py` | produces optional region plots and summary charts |

Supporting files:

- `Subject_Condition.json` – example subject mapping (Active vs SHAM)
- `data/` – placeholder for any supplemental resources
- `pixi.lock` / `pixi.toml` – environment definitions

The pipeline expects `EEG_data/Subject_xxx/Nighty/Output/SourceRecon/*.csv` under the chosen data directory. Running `python main.py <DATA_DIR>` creates `Involvement_Analysis/` with wave metrics, subject summaries, statistical tables, and plots.
