# Source_Ana

Tools for analyzing slow-wave EEG source reconstruction data. The pipeline builds subject-specific thresholds, summarizes involvement, runs non-parametric statistics, and generates basic plots.

## Requirements

- Python 3.8+
- Packages: numpy, pandas, scipy, matplotlib, seaborn
- Optional: pixi (`pixi install`) for environment management

Install with pip if pixi is not used:

```bash
pip install numpy pandas scipy matplotlib seaborn
```

## Data Layout

```
<DATA_DIR>/
├── EEG_data/
│   ├── Subject_001/
│   │   ├── Night1/Output/SourceRecon/*.csv
│   │   └── Night2/...
│   └── Subject_002/
└── Subject_Condition.json
```

CSV files must include a `Time` column followed by numeric columns named with time values (seconds). Filenames should contain `proto{1-8}` and a stage label (`pre`, `stim`, `post`, `early`, `late`).

`Subject_Condition.json` maps subject IDs to `Active` or `SHAM`.

## Running the Pipeline

```bash
python main.py <DATA_DIR> [--subjects Subject_001 Subject_002] [--nights Night1 Night2] [--visualize-regions] [--process-origins]
```

Outputs are written to `<DATA_DIR>/Involvement_Analysis` and include:

- `involvement_analysis.log` – processing log
- `subject_thresholds.csv` – 95th percentile thresholds by subject
- `wave_involvement_data.csv` – wave-level metrics
- `subject_averaged_involvement.csv` – subject means per stage
- `stage_statistics.csv` – descriptive involvement statistics
- `statistical_results.csv` – Wilcoxon and Mann–Whitney summaries
- `percentage_changes.csv` – stim/pre and post/pre deltas where available
- `comprehensive_statistical_results.csv` – combined table used by the standalone statistics routine
- `plots/` – box plot and Active pre/stim paired plot

Region-level plots for individual waves are created only when `--visualize-regions` is provided.

## Module Overview

- `main.py` – command-line interface, orchestration, file I/O
- `involvement.py` – data loading, voxel peak detection, subject thresholding
- `stat.py` – aggregation, statistical tests, percentage-change utilities
- `visual.py` – plotting helpers used by the pipeline and optional wave-level QC

## Notes

- Only protocols 1–8 are processed.
- Origin detection (earliest 10% of involved voxels) is optional because it is slower; enable it with `--process-origins` when needed.
- Logging is verbose by design so that processing decisions are traceable.
