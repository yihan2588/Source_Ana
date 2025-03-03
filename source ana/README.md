# Source Reconstruction Analysis Tool

This tool processes and analyzes slow wave data from source reconstruction experiments. It calculates involvement and origin statistics, performs statistical tests, and generates visualizations.

## Features

- Process CSV files containing time series data
- Analyze wave origins and involvement percentages
- Perform statistical tests to compare stages (pre, early, late, post)
- Generate visualizations (boxplots, bar charts)
- Save results to CSV files
- Generate comprehensive PDF report

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

You will be prompted to enter the path to your source reconstruction directory containing CSV files.

## Output

The tool generates the following outputs in the `results` directory:

### CSV Files 
(what is meta: I treat the protocols collapsed group (waves from all protocols=312 only
separated into Pre=130, Early=31, Late=42, post=109) as a new protocol (called
meta) and perform the same analysis)

- `involvement_statistics.csv`: Statistics about wave involvement for each protocol and stage
- `[protocol]_origin_statistics.csv`: Origin statistics for each protocol
- `statistical_test_results.csv`: Results of statistical tests comparing stages
- `meta_involvement_statistics.csv`: Meta involvement statistics (protocols collapsed)
- `meta_origin_statistics.csv`: Meta origin statistics (protocols collapsed)
- `meta_statistical_test_results.csv`: Meta statistical test results

### Visualizations (PNG Files)

- `involvement_by_protocol_and_stage.png`: Boxplot showing involvement percentages
- `origins_[protocol].png`: Horizontal bar charts showing origin regions for each protocol
- `involvement_summary.png`: Summary bar chart of mean involvement
- `meta_involvement_boxplot.png`: Boxplot of involvement for protocols collapsed data
- `origins_meta.png`: Horizontal bar chart of origins for protocols collapsed data

## Understanding the Results

- **Involvement**: Percentage of voxels (3D pixels, likely brain regions) participating in a wave. Higher involvement = more widespread brain activity.
- **Origins**: Brain regions where waves originate from. Identifies which regions initiate the waves.
- **Statistical Tests**: Tests for significant differences in involvement and origin distribution between stages.
- **Meta Protocol Analysis**: Analysis of protocol combined data across all protocols to identify overall patterns.
