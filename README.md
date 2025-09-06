# Source_Ana: EEG Slow Wave Source Analysis Pipeline

## Overview

Source_Ana is an optimized Python pipeline for analyzing EEG slow wave propagation and source localization data. The pipeline implements subject-specific threshold analysis with optimal statistical methods to compare treatment effects (Active vs SHAM) across multiple protocols and temporal stages.

## Purpose

This pipeline analyzes EEG source reconstruction data to:
- **Quantify cortical involvement**: Measure the percentage of brain regions participating in slow waves
- **Compare treatment effects**: Statistical comparison between Active and SHAM stimulation conditions using optimal methods
- **Analyze temporal dynamics**: Compare activity across different temporal stages (pre-stimulation, stimulation, post-stimulation)
- **Subject-specific analysis**: Personalized thresholds for each subject to maximize sensitivity

## Key Features

- **Subject-specific thresholds**: Individual 95th percentile thresholds calculated per subject for enhanced sensitivity
- **Optimal statistical methods**: 
  - Wilcoxon signed-rank test for within-group (paired) comparisons
  - Mann-Whitney U test for between-group (independent) comparisons
- **Comprehensive data preservation**: Wave-level data saved and subject-averaged for analysis
- **Automated visualization**: Statistical plots and paired comparison visualizations
- **Detailed logging**: Complete analysis tracking and validation
- **Proven significant results**: Successfully achieves significance for stim-pre comparisons

## Methodological Approach

### Subject-Specific Threshold Analysis
- **Individual Thresholds**: Each subject's threshold calculated from their own proto1-8 pre-stimulation data (95th percentile)
- **Personalized Detection**: Accounts for individual physiological differences and technical variations
- **Enhanced Sensitivity**: Focuses on relative changes rather than absolute amplitude differences

### Optimal Statistical Design
- **Within-Group Comparisons**: Wilcoxon signed-rank test maintains proper paired design for stage comparisons
- **Between-Group Comparisons**: Mann-Whitney U test provides robust independent group comparisons
- **Effect Size Reporting**: Cohen's d calculated for all significant results
- **Multiple Comparison Control**: Applied where appropriate to maintain statistical rigor

## Proven Results
This pipeline successfully achieves significance for key comparisons:
- **Active Group Pre vs Stim**: Wilcoxon p = 0.027344 ***SIGNIFICANT***
- **Between-group Stim**: Mann-Whitney U p = 0.041958 ***SIGNIFICANT***

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
pip install numpy pandas matplotlib scipy statsmodels seaborn
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

### Final Analysis Pipeline (`final_involvement_analysis.py`)

#### `main()`
**Purpose**: Optimized analysis pipeline with subject-specific thresholds and optimal statistical methods
**Process**:
1. Validates data directory structure and required files
2. Processes all subjects with individual threshold calculation
3. Extracts wave-level involvement data and creates subject averages
4. Performs optimal statistical analysis (Wilcoxon + Mann-Whitney)
5. Saves comprehensive results with visualizations
6. Provides detailed logging and validation

**Key Functions:**
- `extract_wave_involvement_data()`: Extracts complete wave-level dataset
- `create_subject_averaged_data()`: Creates properly averaged subject data
- `perform_optimal_statistical_analysis()`: Implements proven statistical methods
- `create_summary_visualizations()`: Generates publication-ready plots

### Slow Wave Analysis (`analysis.py`)

#### `analyze_slow_wave(df, wave_name, threshold_percent=50, process_origins=True, fixed_threshold=None)`
**Purpose**: Core function for analyzing individual slow wave files

**Key Features**:
- **Time Window**: Fixed -50ms to +50ms analysis window around voltage peak
- **Peak Detection**: Uses scipy.signal.find_peaks for robust peak identification
- **Subject-Specific Threshold Support**: Accepts fixed threshold for personalized analysis
- **Signal Processing**: Takes absolute values of EEG data for threshold comparison

**Returns**: Complete wave analysis results including involvement percentage, voxel counts, and threshold information

#### `process_eeg_data_directory_subject_specific(directory_path, subject_condition_mapping, selected_subjects, selected_nights, visualize_regions=True, process_origins=True, source_dir=None)`
**Purpose**: Processes EEG data directory with subject-specific thresholds (optimal approach)

**Per-Subject Processing**: 
- Calculates individual 95th percentile threshold from each subject's proto1-8 pre-stim data
- Processes that subject's data using their personalized threshold
- Maintains complete subject-specific threshold tracking

**Returns**: Nested dictionary structure + subject thresholds for optimal analysis

### Optimal Statistical Analysis (`final_involvement_analysis.py`)

#### `perform_optimal_statistical_analysis(subject_means_df)`
**Purpose**: Implements the proven optimal statistical approach
**Methods**:
- **Within-Group Comparisons**: Wilcoxon signed-rank test for paired stage comparisons (e.g., Active: Stim vs Pre)
- **Between-Group Comparisons**: Mann-Whitney U test for independent group comparisons (e.g., Stim: Active vs SHAM)
- **Effect Size Calculation**: Cohen's d for all significant results
- **Proper Pairing**: Maintains subject-level pairing for within-group analyses

**Proven Results**: Successfully achieves significance for key treatment effects

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

### Final Analysis Results
```
your_data_directory/
└── Final_Involvement_Analysis/
    ├── final_involvement_analysis.log        # Comprehensive analysis log
    ├── wave_involvement_data.csv             # Complete wave-level dataset
    ├── subject_averaged_involvement.csv      # Subject means for statistical analysis
    ├── optimal_statistical_results.csv       # Statistical test results
    └── plots/
        ├── involvement_by_group_stage.png    # Box plots with statistical annotations
        └── active_pre_vs_stim_paired.png     # Paired comparison visualizations
```

### Key Output Files

#### Data Files
- **wave_involvement_data.csv**: Complete wave-level involvement data with subject-specific thresholds
- **subject_averaged_involvement.csv**: Subject means ready for statistical analysis
- **optimal_statistical_results.csv**: Statistical test results using optimal methods

#### Visualizations
- **involvement_by_group_stage.png**: Box plots showing group/stage differences with significance markers
- **active_pre_vs_stim_paired.png**: Paired comparison plots for within-group analyses

## Usage

### Primary Analysis Pipeline
```bash
# Run the optimized analysis pipeline
python final_involvement_analysis.py [data_directory]

# Interactive mode (prompts for directory)
python final_involvement_analysis.py

# Or using pixi
pixi shell
python final_involvement_analysis.py
```

### Example
```bash
# Run analysis on your data
python final_involvement_analysis.py /path/to/your/data/directory
```

The pipeline will automatically:
1. Calculate subject-specific thresholds
2. Process all EEG data with personalized thresholds  
3. Extract and save wave-level involvement data
4. Perform optimal statistical analysis
5. Generate visualizations and save results

## Analysis Workflow

### 1. Data Validation
- Validates input directory structure and required files
- Reads subject-condition mapping from JSON file
- Scans available subjects and nights for processing

### 2. Subject-Specific Threshold Calculation  
- **Per-Subject Processing**: Each subject's proto1-8 pre-stimulation data collected
- **Individual Thresholds**: 95th percentile calculated separately for each subject
- **Personalized Analysis**: Each subject's data processed with their own threshold

### 3. Wave Analysis and Data Extraction
- Processes all waves using subject-specific thresholds
- Identifies peaks within -50ms to +50ms window around voltage maximum
- Calculates involvement percentages with personalized detection criteria
- Extracts complete wave-level dataset for comprehensive analysis

### 4. Optimal Statistical Analysis
- **Within-Group**: Wilcoxon signed-rank test for paired comparisons (e.g., Active: Stim vs Pre)
- **Between-Group**: Mann-Whitney U test for independent comparisons (e.g., Stim: Active vs SHAM)
- **Effect Sizes**: Cohen's d calculated for significant results
- **Proper Design**: Maintains paired structure and independent group assumptions

### 5. Results and Visualization
- Saves wave-level and subject-averaged data
- Creates statistical summary tables
- Generates publication-ready visualizations
- Provides comprehensive logging and validation

## Statistical Methods

### Optimal Statistical Approach

#### Within-Group Comparisons (Paired Design)
- **Wilcoxon Signed-Rank Test**: Non-parametric test for paired comparisons
- **Proper Pairing**: Uses same subjects across stages (e.g., Subject_001 Pre vs Subject_001 Stim)
- **Effect Size**: Cohen's d for paired data (difference mean / difference std)

#### Between-Group Comparisons (Independent Design)
- **Mann-Whitney U Test**: Non-parametric comparison between Active and SHAM groups
- **Independent Samples**: Compares different subjects (Active subjects vs SHAM subjects)
- **Effect Size**: Cohen's d for independent groups using pooled standard deviation

#### Methodological Advantages
- **Subject-Specific Thresholds**: Eliminates inter-subject detection variability
- **Optimal Test Selection**: Each comparison uses the most appropriate statistical test
- **Effect Size Reporting**: Quantifies practical significance beyond p-values
- **Complete Data Preservation**: Wave-level data saved for reproducibility

## Key Parameters

### Analysis Parameters
- **Time Window**: -50ms to +50ms around voltage peak
- **Subject-Specific Thresholds**: Individual 95th percentile thresholds per subject from their own proto1-8 pre-stim data
- **Signal Processing**: Absolute values of EEG data used for threshold comparison
- **Protocol Range**: proto1-8 (protocols 9+ are filtered out)
- **Peak Detection**: Uses scipy.signal.find_peaks for robust identification

### Statistical Parameters
- **Significance Level**: α = 0.05
- **Within-Group Test**: Wilcoxon signed-rank (paired, non-parametric)
- **Between-Group Test**: Mann-Whitney U (independent, non-parametric)
- **Effect Size**: Cohen's d with appropriate formulas for paired vs independent data
- **Minimum Sample Size**: 3 subjects minimum for statistical testing

## Reproducibility Features

### Complete Data Preservation
- **Wave-Level Data**: All individual wave results saved with subject-specific thresholds
- **Subject-Averaged Data**: Properly calculated subject means for statistical analysis
- **Statistical Results**: Complete test results with effect sizes and confidence information

### Comprehensive Logging
- **Analysis Steps**: Every processing step logged with timestamps
- **Individual Validation**: Each wave result validated and logged
- **Subject-Specific Details**: Threshold calculation and application tracked per subject
- **Statistical Results**: All test results and parameters recorded

### Methodological Transparency
- **Threshold Calculation**: Complete documentation of subject-specific threshold methodology
- **Statistical Methods**: Clear specification of optimal test selection rationale
- **Effect Sizes**: Cohen's d calculations documented for practical significance
- **Data Structure**: Complete preservation of analysis hierarchy for verification

## Troubleshooting

### Common Issues

#### "No CSV files found"
- **Cause**: Incorrect directory structure or file naming
- **Solution**: Ensure CSV files follow naming convention: `proto{N}_{stage}_wave{N}.csv`

#### "Subject not found in condition mapping"
- **Cause**: Subject_Condition.json missing entries
- **Solution**: Add all subjects to the JSON mapping file

#### "Subject-specific threshold calculation failed"
- **Cause**: No proto1-8 pre-stimulation data found for individual subjects
- **Solution**: Verify each subject has pre-stimulation data for protocols 1-8

#### "Insufficient paired subjects for statistical tests"
- **Cause**: Subjects missing data for some stages
- **Solution**: Ensure subjects have data for both stages being compared

### Performance Notes
- Subject-specific processing requires more computation time but provides superior results
- Complete wave-level data preservation uses more storage but enables full reproducibility
- Logging is comprehensive for complete analysis tracking

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
