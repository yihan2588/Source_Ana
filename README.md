# Source_Ana: EEG Slow Wave Source Analysis Pipeline

## Overview

Source_Ana is a comprehensive Python pipeline for analyzing EEG slow wave propagation and source localization data. The pipeline is designed to compare treatment effects (Active vs SHAM) across multiple protocols and temporal stages, providing statistical analysis and visualization of slow wave origins and cortical involvement patterns.

## Purpose

This pipeline analyzes EEG source reconstruction data to:
- **Identify slow wave origins**: Determine the earliest brain regions showing slow wave activity
- **Quantify cortical involvement**: Measure the percentage of brain regions participating in slow waves
- **Compare treatment effects**: Statistical comparison between Active and SHAM stimulation conditions
- **Analyze temporal dynamics**: Compare activity across different temporal stages (pre-stimulation, stimulation/early, late, post-stimulation)
- **Protocol-specific analysis**: Analyze effects across multiple experimental protocols (proto1-8)

## Key Features

- **Adaptive thresholding**: Uses 95th percentile of pre-stimulation data to identify significant activity
  - **Global thresholds**: Single threshold calculated across all subjects (original approach)
  - **Subject-specific thresholds**: Individual thresholds calculated per subject for enhanced personalization
- **Subject-weighted statistics**: Accounts for varying numbers of waves per subject
- **Comprehensive statistical testing**: Mann-Whitney U, Wilcoxon signed-rank, Chi-square, Fisher's exact tests
- **Multiple correction**: FDR correction for multiple comparisons
- **Threshold comparison**: Built-in tools to compare global vs subject-specific approaches
- **Interactive interface**: User-friendly command-line interface for data selection
- **Automated visualization**: Generates time-series plots and statistical visualizations
- **Detailed logging**: Comprehensive logging of all analysis steps

## Threshold Approaches: Global vs Subject-Specific

### Global Threshold Approach (Original)
- Pools proto1-8 pre-stimulation data from **all subjects**
- Calculates a single 95th percentile threshold applied universally
- **Advantages**: Consistent detection criteria across all subjects
- **Considerations**: May not account for individual physiological differences

### Subject-Specific Threshold Approach (Enhanced)
- Calculates 95th percentile threshold **separately for each subject**
- Uses only that subject's own proto1-8 pre-stimulation data
- Applies personalized threshold to that subject's data only

#### Benefits of Subject-Specific Thresholds:

1. **Individual Physiological Normalization**
   - Accounts for natural variation in EEG signal amplitude between subjects
   - Normalizes for individual brain anatomy and skull thickness differences
   - Controls for subject-specific baseline activity patterns

2. **Technical Variability Control**
   - Eliminates recording session differences (electrode impedance, amplifier settings)
   - Accounts for preprocessing variations that might affect individual subjects
   - Reduces impact of technical artifacts specific to individual recordings

3. **Enhanced Statistical Robustness**
   - Prevents outlier subjects from skewing the global threshold
   - Each subject contributes equally regardless of their absolute signal strength
   - More resistant to systematic baseline differences between treatment groups

4. **Improved Sensitivity to Treatment Effects**
   - Detects subtle involvement patterns relative to individual baselines
   - Particularly beneficial when Active vs SHAM groups have different baseline characteristics
   - Focuses on relative changes rather than absolute amplitude differences

5. **Better Methodological Control**
   - Ensures each subject's analysis is relative to their own physiological baseline
   - Similar to z-score normalization in statistical analysis
   - More aligned with within-subject experimental design principles

#### When to Use Each Approach:
- **Subject-Specific**: Recommended for heterogeneous populations, when baseline differences are expected, or when maximizing sensitivity to individual treatment responses
- **Global**: Appropriate when population homogeneity is assumed or when establishing universal detection criteria across studies
- **Comparison**: Use built-in comparison tools to determine which approach best suits your data

## Installation and Setup

### Prerequisites
- Python 3.8+
- [Pixi](https://prefix.dev/) package manager (recommended)

### Installation with Pixi (Recommended)
```bash
# Clone the repository
git clone https://github.com/yihan2588/Source_Ana.git
cd Source_Ana

# Install dependencies
pixi install

# Activate the environment
pixi shell
```

### Manual Installation
```bash
pip install numpy pandas matplotlib scipy statsmodels jupyter pytest black isort
```

## Input Data Structure

### Required Directory Structure
```
your_data_directory/
├── EEG_data/
│   ├── Subject_001/
│   │   ├── Night1/
│   │   │   └── Output/
│   │   │       └── SourceRecon/
│   │   │           ├── proto1_pre_wave1.csv
│   │   │           ├── proto1_stim_wave1.csv
│   │   │           ├── proto1_post_wave1.csv
│   │   │           └── ...
│   │   └── Night2/
│   └── Subject_002/
└── Subject_Condition.json
```

### Input Files

#### 1. EEG CSV Files
- **Location**: `EEG_data/Subject_XXX/NightX/Output/SourceRecon/`
- **Format**: CSV files with time series data
- **Naming**: `proto{N}_{stage}_wave{N}.csv` where:
  - `{N}`: Protocol number (1-8)
  - `{stage}`: pre, early/stim, late, post
- **Structure**: 
  - First column: Voxel names (brain regions)
  - Subsequent columns: Time points (in seconds, converted to ms internally)
  - Values: EEG amplitude data

#### 2. Subject_Condition.json
- **Purpose**: Maps subject IDs to treatment conditions
- **Format**:
```json
{
  "Subject_001": "Active",
  "Subject_002": "SHAM",
  "Subject_003": "SHAM",
  "Subject_004": "Active"
}
```

## Core Functions

### Main Analysis Pipeline (`main.py` / `main_subject_specific.py`)

#### `main()`
**Purpose**: Interactive entry point for the analysis pipeline
**Process**:
1. Prompts user for data directory path
2. Validates directory structure and files
3. Allows user to select subjects and nights to process
4. Configures analysis options (visualization, origin analysis, threshold approach)
5. Executes full analysis pipeline
6. Generates comprehensive results and visualizations

#### Subject-Specific Features (`main_subject_specific.py`)
- **Threshold Approach Selection**: Choose between global, subject-specific, or comparison analysis
- **Individual Processing**: Each subject processed with their personalized threshold
- **Comparison Analysis**: Direct comparison between threshold approaches
- **Enhanced Logging**: Subject-specific threshold tracking and validation

### Slow Wave Analysis (`analysis.py`)

#### `analyze_slow_wave(df, wave_name, threshold_percent=50, process_origins=True, fixed_threshold=None)`
**Purpose**: Analyzes individual slow wave files to extract origin and involvement metrics

**Parameters**:
- `df`: DataFrame containing wave data
- `wave_name`: Identifier for the wave
- `threshold_percent`: Percentage of max amplitude for threshold (default: 50%)
- `process_origins`: Whether to calculate origin analysis (default: True)
- `fixed_threshold`: Optional fixed threshold value

**Returns**: Dictionary containing:
- `wave_name`: Wave identifier
- `origins`: DataFrame of earliest 10% regions with peak times
- `involvement_count`: Number of voxels above threshold
- `involvement_percentage`: Percentage of total voxels involved
- `involved_voxels`: List of involved voxel names
- `window`: Analysis time window (-50ms to +50ms)
- `threshold`: Applied threshold value

#### `process_eeg_data_directory(directory_path, subject_condition_mapping, selected_subjects, selected_nights, visualize_regions=True, process_origins=True, source_dir=None)`
**Purpose**: Processes entire EEG data directory with global threshold approach

**Pass 1**: Collects proto1-8 pre-stimulation data from ALL subjects to calculate single adaptive threshold (95th percentile)
**Pass 2**: Processes all data using the global threshold

#### `process_eeg_data_directory_subject_specific(directory_path, subject_condition_mapping, selected_subjects, selected_nights, visualize_regions=True, process_origins=True, source_dir=None)`
**Purpose**: Processes EEG data directory with subject-specific thresholds

**Per-Subject Processing**: 
- Calculates individual threshold from each subject's proto1-8 pre-stim data
- Processes that subject's data using their personalized threshold
- Maintains subject-specific threshold tracking

**Returns**: Nested dictionary structure + subject thresholds:
```python
results_by_treatment_group = {
  "Active": {
    "Subject_001": {
      "proto1": {
        "pre": [wave_result1, wave_result2, ...],
        "stim": [wave_result1, wave_result2, ...],
        "post": [wave_result1, wave_result2, ...]
      }
    }
  },
  "SHAM": { ... }
}

subject_thresholds = {
  "Subject_001": 1.234e-06,
  "Subject_002": 2.456e-06,
  ...
}
```

#### `compare_threshold_approaches(directory_path, subject_condition_mapping, selected_subjects, selected_nights)`
**Purpose**: Compares global vs subject-specific threshold approaches
**Features**:
- Runs both approaches on the same dataset
- Analyzes differences in involvement percentages
- Statistical comparison between approaches
- Detailed per-subject and per-group comparisons

### Statistical Analysis Functions

#### `analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list)`
**Purpose**: Compares treatment groups across all protocols and subjects
**Analysis Types**:
- Between-group comparisons (Active vs SHAM) using Mann-Whitney U tests
- Within-group stage comparisons using Wilcoxon signed-rank tests
- Origin distribution comparisons using Chi-square or Fisher's exact tests

#### `analyze_proto_specific_comparison(results_by_treatment_group, master_region_list)`
**Purpose**: Protocol-specific treatment comparisons for proto1-8
**Features**:
- Individual protocol analysis
- Subject-specific data collection for proto1
- Same statistical tests as overall comparison

#### `analyze_within_group_stage_comparison(results_by_treatment_group, master_region_list)`
**Purpose**: Within-treatment group comparisons across stages
**Analysis**: Paired statistical tests comparing different temporal stages within each treatment group

### Utility Functions (`utils.py`)

#### `extract_region_name(full_name)`
**Purpose**: Extracts anatomical region names from full voxel identifiers
**Usage**: Standardizes region naming across different voxel naming conventions

#### `calculate_involvement_statistics(involvement_data)`
**Purpose**: Calculates descriptive statistics (mean, median, std, count) for involvement data
**Returns**: Dictionary with statistical measures

#### `compute_master_region_list(results_by_protocol)`
**Purpose**: Creates ordered list of brain regions based on frequency of origin occurrence
**Usage**: Ensures consistent region ordering across visualizations and analyses

## Output Structure

### Generated Directories

#### Global Threshold Analysis
```
your_data_directory/
└── Source_Ana/
    ├── source_ana_run.log                    # Comprehensive log file
    ├── Overall_Treatment_Comparison/         # Overall analysis results
    │   ├── overall_involvement_stats.csv
    │   ├── overall_origin_stats.csv
    │   └── plots/
    ├── Proto_Specific_Comparison/            # Protocol-specific results
    │   ├── proto1_involvement_stats.csv
    │   ├── proto1_origin_stats.csv
    │   └── plots/
    ├── Within_Group_Stage_Comparison/        # Within-group comparisons
    │   ├── Active_involvement_stats.csv
    │   ├── SHAM_involvement_stats.csv
    │   └── plots/
    ├── Percentage_Changes/                   # Percentage change analysis
    │   ├── overall_percentage_changes.csv
    │   ├── protocol_percentage_changes.csv
    │   └── plots/
    ├── Time_Series_Plots/                    # Individual wave visualizations
    └── consolidated_statistical_results.csv  # All statistical test results
```

#### Subject-Specific Threshold Analysis
```
your_data_directory/
└── Source_Ana_SubjectSpecific/
    ├── source_ana_subject_specific_run.log   # Subject-specific analysis log
    ├── threshold_approach_comparison.csv     # Threshold comparison results (if comparing)
    └── Source_Ana_SubjectSpecific/           # Analysis results with subject-specific thresholds
        ├── Overall_Treatment_Comparison/
        ├── Proto_Specific_Comparison/
        ├── Within_Group_Stage_Comparison/
        ├── Percentage_Changes/
        ├── Time_Series_Plots/
        └── consolidated_statistical_results.csv
```

### Key Output Files

#### Statistical Results
- **consolidated_statistical_results.csv**: All statistical test results in unified format
- **involvement_stats.csv**: Descriptive statistics for cortical involvement
- **origin_stats.csv**: Origin region frequency and statistics
- **percentage_changes.csv**: Percentage changes between stages

#### Visualizations
- **Time-series plots**: Individual wave propagation visualizations
- **Statistical plots**: Group comparisons and trend analyses
- **Percentage change plots**: Changes between temporal stages

## Usage Examples

### Basic Usage
```bash
# Run the main analysis pipeline (global thresholds)
python main.py

# Run subject-specific threshold analysis
python main_subject_specific.py

# Or using pixi
pixi run run-analysis
```

### Testing Single Wave
```bash
# Test analysis on a single wave file
python test_single_wave.py

# Or using pixi
pixi run test-single-wave
```

### Development Tasks
```bash
# Format code
pixi run format-code

# Run Jupyter lab
pixi run jupyter-lab
```

### Subject-Specific Threshold Analysis
```bash
# Run with interactive options
python main_subject_specific.py

# Choose from:
# 1. Subject-specific thresholds only
# 2. Global threshold only  
# 3. Compare both approaches
```

## Analysis Workflow

### 1. Data Preprocessing
- Validates input directory structure
- Reads subject-condition mapping
- Scans available subjects and nights
- User selects data subset for processing

### 2. Threshold Calculation
- **Global Approach**: Collects all proto1-8 pre-stimulation data from all subjects, calculates single 95th percentile threshold
- **Subject-Specific Approach**: Calculates individual 95th percentile thresholds for each subject from their own proto1-8 pre-stim data
- Ensures consistent detection methodology while accounting for individual differences

### 3. Wave Analysis
- **Pass 2**: Processes all waves with adaptive threshold
- Identifies peak times within -50ms to +50ms window
- Calculates involvement percentages
- Determines origin regions (earliest 10% of involved regions)

### 4. Statistical Analysis
- **Overall Comparison**: Active vs SHAM across all protocols
- **Protocol-Specific**: Individual protocol comparisons
- **Within-Group**: Stage comparisons within treatment groups
- **Multiple Correction**: FDR correction for multiple comparisons

### 5. Visualization and Export
- Generates comprehensive visualizations
- Exports statistical results to CSV files
- Creates detailed log of all analysis steps

## Statistical Methods

### Between-Group Comparisons
- **Mann-Whitney U Test**: Non-parametric comparison of Active vs SHAM groups
- **Effect Size**: Reported for significant differences

### Within-Group Comparisons
- **Wilcoxon Signed-Rank Test**: Paired comparisons across stages
- **Subject-Level Pairing**: Uses subject means for paired analysis

### Origin Distribution Analysis
- **Chi-Square Test**: For larger contingency tables
- **Fisher's Exact Test**: For 2x2 comparisons
- **Contingency Tables**: Based on top regions by frequency

### Multiple Comparisons
- **FDR Correction**: Benjamini-Hochberg procedure
- **Family-Wise Error Rate Control**: Applied to related test families

## Key Parameters

### Analysis Parameters
- **Time Window**: -50ms to +50ms around voltage peak
- **Threshold Options**:
  - **Global**: Single 95th percentile threshold across all subjects' proto1-8 pre-stimulation data
  - **Subject-Specific**: Individual 95th percentile thresholds per subject from their own proto1-8 pre-stim data
- **Origin Definition**: Earliest 10% of involved regions
- **Protocol Range**: proto1-8 (protocols 9+ are filtered out)

### Statistical Parameters
- **Significance Level**: α = 0.05
- **Multiple Correction**: FDR (Benjamini-Hochberg)
- **Minimum Sample Size**: 3 subjects for paired tests

## Reproducibility Features

### Logging
- Comprehensive logging of all analysis steps
- Individual wave validation logs
- Statistical test results and parameters
- File processing summaries

### Version Control
- Git integration for tracking analysis versions
- Standardized output structure
- Detailed parameter documentation

### Data Validation
- Input file format validation
- Statistical assumption checking
- Missing data handling
- Error reporting and recovery

## Troubleshooting

### Common Issues

#### "No CSV files found"
- **Cause**: Incorrect directory structure or file naming
- **Solution**: Ensure CSV files follow naming convention: `proto{N}_{stage}_wave{N}.csv`

#### "Subject not found in condition mapping"
- **Cause**: Subject_Condition.json missing entries
- **Solution**: Add all subjects to the JSON mapping file

#### "Not enough data for statistical tests"
- **Cause**: Insufficient sample sizes
- **Solution**: Ensure at least 3 subjects per group for meaningful statistics

#### "Threshold calculation failed"
- **Cause**: No proto1-8 pre-stimulation data found
- **Solution**: Verify pre-stimulation data exists for protocols 1-8

### Performance Considerations
- Large datasets may require significant memory and processing time
- Consider processing subset of subjects/nights for initial analysis
- Enable logging to monitor progress and identify bottlenecks

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use provided formatting tools: `pixi run format-code`
- Add comprehensive docstrings for new functions

### Testing
- Add tests for new functionality
- Use `pytest` for unit testing
- Validate with known datasets

### Documentation
- Update README for new features
- Add inline documentation for complex algorithms
- Provide usage examples

## Citation

If you use this pipeline in your research, please cite the associated publication and this repository.

## License

[Add appropriate license information]

## Contact

For questions or issues, please contact:
- GitHub Issues: [Repository Issues](https://github.com/yihan2588/Source_Ana/issues)
- Email: [Add contact email if appropriate]
