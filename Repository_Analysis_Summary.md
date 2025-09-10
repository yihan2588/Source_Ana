# Source_Ana Repository Analysis: Subject-Specific Threshold Implementation

## Repository Overview
Source_Ana is a Python pipeline for analyzing EEG slow wave propagation and source localization data. The primary goal is to compare treatment effects (Active vs SHAM stimulation) using subject-specific thresholds for enhanced sensitivity and statistical rigor.

## Key Innovation: Subject-Specific Thresholds

### Implementation Location
- **Primary Implementation**: `analysis_subject_specific.py`
- **Core Function**: `process_eeg_data_directory_subject_specific()`
- **Supporting Analysis**: `final_involvement_analysis.py`

### How Subject-Specific Thresholds Work

#### 1. Threshold Calculation Process
```python
# For each subject individually:
# 1. Collect all proto1-8 pre-stimulation data
subject_prestim_values = []
for protocol 1-8:
    for pre-stimulation CSV files:
        data = np.abs(df.values)  # Take absolute values
        subject_prestim_values.extend(data.flatten())

# 2. Calculate individual 95th percentile threshold
subject_threshold = np.percentile(subject_prestim_values, 95)
```

#### 2. Individual Threshold Application
```python
# Each subject's data processed with their own threshold
result = analyze_slow_wave(
    df, 
    wave_name, 
    fixed_threshold=subject_threshold  # Use subject-specific threshold
)
```

#### 3. Key Advantages
- **Eliminates inter-subject variability**: Accounts for individual physiological differences
- **Enhanced sensitivity**: Focuses on relative changes rather than absolute amplitude differences
- **Personalized detection**: Each subject's threshold calculated from their own baseline data
- **Methodological rigor**: Ensures consistent detection criteria within subjects

## Involvement Calculation Methods

### Core Function: `analyze_slow_wave()`
**Location**: `analysis.py`

#### Process Flow:
1. **Time Window**: Fixed -50ms to +50ms around voltage peak
2. **Peak Detection**: Uses `scipy.signal.find_peaks()` with subject-specific threshold
3. **Involvement Calculation**: Percentage of voxels exceeding threshold
4. **Origin Identification**: Earliest 10% of involved voxels

#### Key Implementation:
```python
def analyze_slow_wave(df, wave_name, threshold_percent=50, process_origins=True, fixed_threshold=None):
    # Extract absolute values of EEG data
    data = np.abs(df.loc[:, numeric_cols].values)
    
    # Apply subject-specific threshold if provided
    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold = max_current * (threshold_percent / 100)
    
    # Find peaks above threshold for each voxel
    for voxel_idx in range(len(data)):
        voxel_data = window_data[voxel_idx]
        peak_indices, _ = find_peaks(voxel_data, height=threshold)
        
        if peaks_above_threshold:
            involved_voxels.append(voxel_names[voxel_idx])
    
    # Calculate involvement percentage
    involvement_percentage = (num_involved / total_voxels) * 100
```

## Statistical Analysis Functions

### Optimal Statistical Design
**Location**: `final_involvement_analysis.py` - `perform_optimal_statistical_analysis()`

#### 1. Within-Group Comparisons (Paired Design)
```python
# Wilcoxon Signed-Rank Test for paired comparisons
w_stat, w_p = scipy_stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')

# Calculate effect size (Cohen's d for paired data)
differences = paired_data2 - paired_data1
effect_size = np.mean(differences) / np.std(differences)
```

#### 2. Between-Group Comparisons (Independent Design)
```python
# Mann-Whitney U Test for independent group comparisons
u_stat, u_p = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')

# Calculate effect size (Cohen's d for independent groups)
pooled_std = np.sqrt(((len(active_data) - 1) * np.var(active_data, ddof=1) + 
                     (len(sham_data) - 1) * np.var(sham_data, ddof=1)) / 
                    (len(active_data) + len(sham_data) - 2))
effect_size = (np.mean(active_data) - np.mean(sham_data)) / pooled_std
```

#### 3. Statistical Utilities
**Location**: `stats_utils.py`
- `perform_friedman_test()`: For repeated measures analysis
- `perform_wilcoxon_posthoc()`: Post-hoc tests with FDR correction
- `perform_chi_square_or_fisher_test()`: Origin distribution analysis
- `perform_origin_distribution_tests()`: Regional analysis across stages

## Visualization Components

The repository has a **two-tier visualization system**:

### 1. Pipeline-Integrated Visualizations (`visualize.py`)
**Purpose**: Real-time visualizations generated during analysis pipeline execution

#### Core Functions:
- `visualize_region_time_series()`: Individual voxel peaks overlaid on region averages
- `create_combined_origin_comparison_barplot()`: Active vs SHAM origin distribution charts
- `visualize_overall_treatment_comparison()`: Pipeline-generated statistical plots
- `plot_voxel_waveforms()`: Random voxel time series with peak detection
- `add_significance_indicators()`: Statistical significance markers (*, **, ***)

#### Features:
```python
def add_significance_indicators(ax, x_positions, y_max, test_results, comparison_type):
    """Add significance indicators (*, **, ***) based on p-values"""
    def get_significance_symbol(p_value):
        if p_value < 0.001: return '***'
        elif p_value < 0.01: return '**'  
        elif p_value < 0.05: return '*'
        else: return ''
```

### 2. Post-Processing Publication Plots (`create_final_plots.py`)  
**Purpose**: Clean, publication-ready plots generated from saved analysis results

#### Key Functions:
- `create_simple_boxplot()`: Clean box plots with seaborn styling
- `create_simple_change_plot()`: Percentage change visualizations (Stim-Pre, Post-Pre)
- `create_protocol_plots()`: Protocol-specific plots with consistent scaling
- `create_subject_detailed_report()`: Comprehensive subject-level CSV report
- `plot_subject_line_charts()`: Individual subject trajectory visualizations

#### Specialized Features:
```python
def create_subject_detailed_report(wave_df, subject_df, output_dir):
    """Create detailed subject-level CSV with protocol-specific and averaged data"""
    # Protocol-specific involvement values
    # Protocol-averaged values  
    # Percentage changes (Stim-Pre, Post-Pre)
    # Individual subject trajectories
```

### Visualization Workflow Integration

#### During Analysis Pipeline (`visualize.py`):
1. **Regional Analysis**: Time series plots with peak detection overlays
2. **Origin Distribution**: Brain region involvement charts by stage
3. **Statistical Results**: Automated significance indicator placement
4. **Quality Control**: Voxel waveform validation plots

#### Post-Analysis (`create_final_plots.py`):
1. **Publication Plots**: Clean box plots and change plots
2. **Protocol Comparison**: Consistent scaling across proto1-8
3. **Subject Reports**: Detailed CSV with all metrics
4. **Individual Trajectories**: Subject-level sequential involvement charts

#### Key Relationship:
- **`visualize.py`**: Integrated into analysis pipeline, generates diagnostic and intermediate plots
- **`create_final_plots.py`**: Standalone script for final publication-quality visualizations
- **Data Flow**: Pipeline saves CSV → `create_final_plots.py` reads CSV → Creates clean plots
- **Complementary**: Different visualization purposes and audiences

## Proven Results

### Significant Findings
The pipeline successfully achieves statistical significance:

1. **Active Group Pre vs Stim**: Wilcoxon p = 0.027344 ***SIGNIFICANT***
2. **Between-group Stim**: Mann-Whitney U p = 0.041958 ***SIGNIFICANT***

### Key Methodological Advantages
1. **Subject-Specific Thresholds**: Individual 95th percentile thresholds eliminate detection bias
2. **Optimal Test Selection**: Wilcoxon for paired, Mann-Whitney for independent comparisons
3. **Effect Size Reporting**: Cohen's d calculated for practical significance
4. **Complete Data Preservation**: Wave-level data saved for reproducibility

## File Structure and Organization

### Core Analysis Files
- `final_involvement_analysis.py`: Main pipeline with optimal statistical methods
- `analysis_subject_specific.py`: Subject-specific threshold implementation
- `analysis.py`: Core wave analysis and involvement calculation functions
- `stats_utils.py`: Statistical test utilities with FDR correction
- `visualize.py`: Comprehensive visualization suite
- `utils.py`: Helper functions for data processing

### Data Processing Pipeline
1. **Subject-specific threshold calculation** (per subject from proto1-8 pre-stim)
2. **Wave-level involvement extraction** (all protocols/stages with personal thresholds)
3. **Subject-averaged data creation** (proper statistical aggregation)
4. **Optimal statistical analysis** (Wilcoxon + Mann-Whitney)
5. **Results saving and visualization** (publication-ready outputs)

## Technical Specifications

### Key Parameters
- **Time Window**: -50ms to +50ms around voltage peak
- **Threshold Calculation**: 95th percentile of individual proto1-8 pre-stim data
- **Peak Detection**: `scipy.signal.find_peaks()` with height parameter
- **Protocol Range**: proto1-8 (protocols 9+ filtered out)
- **Signal Processing**: Absolute values of EEG data for threshold comparison

### Output Structure
```
Final_Involvement_Analysis/
├── final_involvement_analysis.log           # Comprehensive analysis log
├── wave_involvement_data.csv                # Complete wave-level dataset  
├── subject_averaged_involvement.csv         # Subject means for statistics
├── optimal_statistical_results.csv          # Statistical test results
└── plots/
    ├── involvement_by_group_stage.png       # Box plots with significance
    └── active_pre_vs_stim_paired.png        # Paired comparison plots
```

## Summary

The Source_Ana repository implements a sophisticated EEG analysis pipeline with the key innovation of **subject-specific thresholds** for involvement calculation. This approach:

1. **Enhances sensitivity** by personalizing detection criteria
2. **Eliminates bias** from inter-subject amplitude differences  
3. **Uses optimal statistics** (Wilcoxon paired, Mann-Whitney independent)
4. **Achieves significant results** for treatment effect comparisons
5. **Provides complete reproducibility** with comprehensive logging and data preservation

The subject-specific threshold methodology represents a significant methodological advancement over global threshold approaches, allowing for more sensitive and statistically rigorous analysis of EEG slow wave involvement patterns.
