# Region Time Series Visualization

## Overview

This feature enhances the Source_Ana package with the ability to visualize time series data for each brain region. The visualization shows:

1. Time series data for each region (averaged from all voxels in that region)
2. Local maxima (peaks) highlighted with blue dots
3. Origins (earliest active regions) highlighted with red stars

## Implementation Details

The feature has been implemented through the following changes:

1. A new function `visualize_region_time_series()` in `visualize.py`
2. Integration into the main analysis pipeline in `analysis.py`
3. A demonstration script `demo_region_visualization.py` for easy testing

### How It Works

1. For each wave analysis, voxel-level data is grouped by brain region
2. The time series data for each region is calculated by averaging all voxels in that region
3. Local maxima (peaks) are identified within each region's time series
4. The "origin" points (earliest activity) are marked with red stars
5. Plots are saved in a structured directory hierarchy under `results/region_plots/[protocol]/[stage]/`

## Usage

### Automatic Integration

The visualization is automatically generated during normal analysis. It's enabled by default when running:
```python
process_directory() 
# or
process_eeg_data_directory()
```

You can disable it by setting the `visualize_regions` parameter to `False`:
```python
process_directory(directory_path, visualize_regions=False)
```

### Manual Usage

To manually generate a visualization for a specific wave:

```python
from analysis import analyze_slow_wave
from visualize import visualize_region_time_series

# Analyze the wave
wave_result = analyze_slow_wave(df, wave_name)

# Generate visualization
visualize_region_time_series(wave_result, csv_file_path, output_dir="results")
```

### Demo Script

A demonstration script is provided for easy testing:

```
python demo_region_visualization.py [csv_file_path]
```

If no CSV file is provided, the script will try to find one automatically.

## Output

The visualizations are saved in:
```
results/region_plots/[protocol]/[stage]/[wave_name]_region_time_series.png
```

Each plot includes:
- Time series data for each region (different colored lines)
- Blue dots marking local maxima within each region
- Red stars marking the "origin" points (earliest activity)
- A vertical line at t=0
- A legend identifying each region

## Example

![Example Region Visualization](results/region_plots/proto1/early/example_region_time_series.png)
