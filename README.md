# EEG Slow Wave Analysis Tool

This tool analyzes EEG slow wave data to extract metrics related to wave origins and involvement across different treatment groups, protocols, and stages. It provides comprehensive statistical analysis and visualization capabilities.

## Overview

The EEG Slow Wave Analysis Tool processes CSV files containing EEG data, extracts key metrics, performs statistical analyses, and generates visualizations to help understand the differences between treatment groups (Active vs. SHAM) across different protocols and stages.

## Data Processing Pipeline

1. **Data Loading and Organization**
   - Processes EEG data from a directory structure with treatment groups, subjects, and protocols
   - Supports both new EEG data structure (with Active/SHAM groups) and old SourceRecon structure
   - Organizes data by protocol and stage (pre, early, late, post)

2. **Slow Wave Analysis**
   - Analyzes individual slow wave CSV files to extract:
     - **Involvement**: Percentage of voxels involved in the wave
     - **Origins**: Brain regions where the wave originates (earliest 10% of involved voxels)
   - Applies thresholding to identify significant activity
   - Extracts region names from voxel identifiers

3. **Statistical Analysis**
   - **Involvement Statistics**: Mean, median, standard deviation across stages and groups
   - **Origin Statistics**: Counts and percentages of waves originating from each brain region
   - **Statistical Tests (not used now, subjected to change)**:
     - Kruskal-Wallis and post-hoc Mann-Whitney U tests for involvement comparisons
     - Chi-Square or Fisher's Exact tests for origin distribution comparisons
     - Multiple comparison correction using FDR (False Discovery Rate)

4. **Comparison Analyses**
   - **Treatment Group Comparison**: Compares Active vs. SHAM groups
   - **Protocol-Specific Comparison**: Compares treatment groups for each protocol
   - **Within-Group Stage Comparison**: Analyzes stage progression within each treatment group

5. **Visualization**
   - **Involvement Visualizations**:
     - Bar charts comparing involvement across stages and treatment groups
     - Individual and combined visualizations for each protocol and treatment group
   - **Origin Visualizations**:
     - Horizontal bar charts showing origin distributions across stages
     - Combined origin comparison charts between treatment groups
   - Visualizations are organized by:
     1. Treatment group comparison (Active vs. SHAM)
     2. Protocol-specific treatment comparison
     3. Within-treatment-group stage comparison

6. **Results Saving**
   - Saves all statistical results to CSV files
   - Organizes output files by analysis type and protocol
   - Creates a structured directory hierarchy for results

## Key Metrics

### Involvement
- Percentage of voxels that show significant activity during a slow wave
- Calculated as: (number of involved voxels / total voxels) * 100
- Thresholding based on maximum current amplitude

### Origins
- Brain regions where the slow wave originates
- Identified as the earliest 10% of involved voxels based on peak time
- Aggregated across waves to identify common origin regions

## Data Structure

The code expects EEG data in one of two formats:
1. **New EEG data structure**:
   ```
   EEG_data/
   ├── Active/
   │   ├── Subject_1/
   │   │   └── Night1/
   │   │       └── Output/
   │   │           └── SourceRecon/
   │   │               └── [CSV files]
   │   └── Subject_2/
   │       └── ...
   └── SHAM/
       ├── Subject_1/
       │   └── ...
       └── Subject_2/
           └── ...
   ```

CSV files should follow the naming convention: `*proto#*_(pre|early|late|post)-stim*.csv`

## CSV File Format

The expected CSV format has:
- First column: Voxel names/identifiers
- Column headers: Time points (in seconds)
- Values: Current/activity measurements at each voxel and time point

## Output

The tool generates:
1. **Statistical Results**: CSV files with detailed statistics
2. **Visualizations**: PNG files showing involvement and origin comparisons
3. **Console Output**: Summary of findings and statistical test results

## Usage

Run the main script and provide the path to your EEG data directory:

```
python main.py
```

## Visualization Types

1. **Involvement Bar Charts**: Mean involvement across stages and groups
2. **Origin Stage Comparison Barplots**: Horizontal bar charts showing origin distributions
3. **Combined Origin Comparison Barplots**: Side-by-side comparison of origin distributions

## Selection Criteria for Origin Barplots

Origin barplots don't display all brain regions but apply filtering:
1. **Minimum Count Threshold**: Regions must have at least `min_count` occurrences
2. **Master Region List Filtering**: Only regions in the master list are considered
3. **Top N Limitation**: Limited to the top 15 regions by total count
4. **Consistent Region Display**: Same filtered regions used across all stages

## Statistical Tests

1. **For Involvement**:
   - Kruskal-Wallis test to detect overall differences
   - Post-hoc Mann-Whitney U tests with FDR correction

2. **For Origin Distribution**:
   - Chi-Square test when assumptions are met
   - Fisher's Exact test for 2x2 tables or when Chi-Square assumptions are violated

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- statsmodels
