# EEG Slow Wave Analysis Tool

This tool analyzes EEG slow wave data to extract metrics related to wave origins and involvement across different treatment groups, protocols, and stages. It provides comprehensive statistical analysis and visualization capabilities.

## Overview

The EEG Slow Wave Analysis Tool processes CSV files containing EEG data, extracts key metrics, performs statistical analyses, and generates visualizations to help understand the differences between treatment groups (Active vs. SHAM) across different protocols and stages.

## Data Processing Pipeline

1. **Data Loading and Organization**
   - Processes EEG data from a directory structure with subjects, nights, and protocols
   - Uses a Subject_Condition JSON file to map subjects to treatment groups (Active/SHAM)
   - Allows selection of specific subjects and nights to process
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

The code expects EEG data in the following format:
```
EEG_data/
├── Subject_001/             # Individual subject
│   ├── Night1/
│   │   └── Output/
│   │       └── SourceRecon/ # Directory containing all CSV files
│   │           └── [CSV files]
│   └── Night2/
│       └── ...
├── Subject_002/
└── Subject_003/
```

Additionally, a Subject_Condition JSON file is required to map subjects to treatment groups:
```json
{
  "Subject_001": "Active",
  "Subject_002": "Active",
  "Subject_003": "SHAM",
  "Subject_004": "SHAM"
}
```

### Creating the Subject_Condition JSON File

The Subject_Condition JSON file is a simple mapping of subject IDs to their treatment group. Here's how to create it:

1. Create a new text file with a `.json` extension (e.g., `subject_condition.json`)
2. Add the mapping in the following format:

```json
{
  "Subject_001": "Active",
  "Subject_002": "Active",
  "Subject_003": "SHAM",
  "Subject_004": "SHAM"
}
```

3. Make sure each subject ID exactly matches the directory name in your data structure
4. Treatment group names must be either "Active" or "SHAM" (case-sensitive)
5. Save the file and provide its path when running the analysis script

Example of creating this file using a text editor:
```
1. Open your favorite text editor (Notepad, VS Code, etc.)
2. Copy and paste the template above
3. Replace the subject IDs with your actual subject IDs
4. Set the appropriate treatment group for each subject
5. Save the file as "subject_condition.json"
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

Run the main script:

```
python main.py
```

The script will prompt you for:
1. The path to your EEG data directory
2. The path to the Subject_Condition JSON file
3. Selection of subjects to process (you can select specific subjects or 'all')
4. Selection of nights to process (you can select specific nights or 'all')

After providing this information, the script will process the selected data and generate results.

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
