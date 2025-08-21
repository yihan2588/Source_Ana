import os
import re
import logging # Added
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests # Added for FDR correction

from utils import (
    extract_region_name,
    calculate_origin_statistics,
    calculate_involvement_statistics,
    collect_data_by_treatment_group,
    read_subject_condition_mapping,
    scan_available_subjects_and_nights
)


def validate_wave_result(result, csv_file_path=None):
    """
    Print and save validation information for a single wave file output.

    Args:
        result: Dictionary containing wave analysis results from analyze_slow_wave()
        csv_file_path: Path to the original CSV file (to save log file next to it)
    """
    wave_name = result['wave_name']
    involvement_percentage = result['involvement_percentage']
    involvement_count = result['involvement_count']

    # Prepare validation output
    validation_lines = []
    validation_lines.append(f"[VALIDATION] Wave: {wave_name}")
    validation_lines.append(f"[VALIDATION] Involvement: {involvement_percentage:.2f}% ({involvement_count} voxels)")

    # Add origin information
    origins = result.get('origins', None)
    if origins is not None and not origins.empty:
        validation_lines.append(f"[VALIDATION] Origin Regions ({len(origins)} regions):")
        for _, row in origins.iterrows():
            validation_lines.append(f"[VALIDATION]   - {row['region']} at {row['peak_time']:.2f}ms")
    else:
        validation_lines.append("[VALIDATION] No origin regions detected")

    # Add window and threshold information
    window = result.get('window', (0, 0))
    threshold = result.get('threshold', 0)
    validation_lines.append(f"[VALIDATION] Window: {window[0]}ms to {window[1]}ms, Threshold: {threshold:.10e}")
    validation_lines.append("[VALIDATION] ------------------------------")

    # Log to main pipeline log
    for line in validation_lines:
        logging.info(line)

    # Save to individual log file if CSV path is provided
    if csv_file_path:
        try:
            # Create log file next to the original CSV
            log_file_path = str(csv_file_path).replace('.csv', '.log')
            with open(log_file_path, 'w') as log_file:
                for line in validation_lines:
                    log_file.write(f"{line}\n")
            logging.info(f"[VALIDATION] Saved validation log to: {log_file_path}")
        except Exception as e:
            logging.error(f"[VALIDATION] Error saving validation log: {str(e)}")

# Updated imports for stats_utils
import itertools # Added for combinations
from statsmodels.stats.multitest import multipletests # Added for FDR correction
from stats_utils import (
    perform_friedman_test,
    perform_wilcoxon_posthoc,
    perform_origin_distribution_tests,
    perform_chi_square_or_fisher_test
)
# Keep scipy stats import for Mann-Whitney U (group comparisons)
from scipy import stats as scipy_stats
from scipy.signal import find_peaks # Added for peak detection
from visualize import visualize_region_time_series, plot_voxel_waveforms


def analyze_slow_wave(df, wave_name, threshold_percent=50, process_origins=True, fixed_threshold=None):
    """
    Analyze a single slow wave CSV file to extract origin and involvement metrics.
    Uses a fixed window of -50ms to +50ms.
    
    Args:
        df: DataFrame containing the wave data
        wave_name: Name identifier for the wave
        threshold_percent: Percentage of max amplitude to use as threshold (default 50%)
        process_origins: If True, calculate origin analysis; if False, skip origin processing
        fixed_threshold: If provided, use this threshold instead of calculating from data
    """
    # Debug logging can be controlled by setting the logger level if needed
    # logging.debug(f"\nAnalyzing {wave_name}...")
    try:
        if 'Time' in df.columns:
            # CSV format that starts with a 'Time' column
            numeric_cols = []
            for col in df.columns[1:]:
                if not str(col).startswith('Unnamed'):
                    try:
                        float(col)  # test if col is numeric
                        numeric_cols.append(col)
                    except ValueError:
                        continue

            time_points = np.array([float(t) for t in numeric_cols]) * 1000  # Convert to ms
            data = np.abs(df.loc[:, numeric_cols].values)
            voxel_names = df.iloc[:, 0].values
        else:
            raise ValueError("CSV format doesn't match expected Format 2")
    except Exception as e:
        raise ValueError(f"Could not parse CSV format: {str(e)}")

    window_start = -50  # ms
    window_end = 50     # ms
    window_mask = (time_points >= window_start) & (time_points <= window_end)
    if sum(window_mask) == 0:
        logging.warning(f"No time points found in window [{window_start}, {window_end}] ms for {wave_name}")
        window_mask = np.ones_like(time_points, dtype=bool) # Use all points if window is empty

    window_data = data[:, window_mask]
    window_times = time_points[window_mask]

    global_max_value = 0
    global_max_time = np.nan # Use NaN if no data
    if window_data.size > 0:
        max_current = np.max(window_data)
        # Use fixed threshold if provided, otherwise calculate from data
        if fixed_threshold is not None:
            threshold = fixed_threshold
        else:
            threshold = max_current * (threshold_percent / 100)
        # Find the time of the global maximum within the window
        flat_idx = np.argmax(window_data)
        voxel_idx_max, time_idx_max = np.unravel_index(flat_idx, window_data.shape)
        global_max_time = window_times[time_idx_max]
        global_max_value = max_current # This is the value
    else:
        threshold = fixed_threshold if fixed_threshold is not None else 0

    voxel_peak_times = []
    involved_voxels = []

    for voxel_idx in range(len(data)): # Corrected indentation
        if voxel_idx < window_data.shape[0]:
            voxel_data = window_data[voxel_idx]
            
            # Use scipy.signal.find_peaks
            peak_indices, _ = find_peaks(voxel_data, height=threshold)
            peaks_above_threshold = []
            if peak_indices.size > 0:
                # Get the times and values for the found peaks
                peak_times = window_times[peak_indices]
                peak_values = voxel_data[peak_indices]
                peaks_above_threshold = list(zip(peak_times, peak_values))

            if peaks_above_threshold:
                # Find the peak that is closest to 0 ms (voltage peak)
                closest_peak = min(peaks_above_threshold, key=lambda x: abs(x[0] - 0))
                closest_peak_time = closest_peak[0]
                
                region = extract_region_name(voxel_names[voxel_idx])
                voxel_peak_times.append({
                'full_name': voxel_names[voxel_idx],
                'region': region,
                'peak_time': closest_peak_time
                })
                involved_voxels.append(voxel_names[voxel_idx]) # Indented this line

    total_voxels = len(data)
    num_involved = len(involved_voxels)
    involvement_percentage = (num_involved / total_voxels) * 100 if total_voxels > 0 else 0

    # Identify "origins" as the earliest 10% of involved voxels
    origins = []
    if process_origins and voxel_peak_times:
        peak_times_df = pd.DataFrame(voxel_peak_times).sort_values('peak_time')
        n_origins = max(1, int(len(peak_times_df) * 0.1))
        origins = peak_times_df.head(n_origins)
    elif not process_origins:
        # Create empty DataFrame with expected columns when origins are skipped
        origins = pd.DataFrame(columns=['full_name', 'region', 'peak_time'])

    # Debug logging can be controlled by setting the logger level if needed
    # logging.debug(f"Involvement: {involvement_percentage:.1f}% ({num_involved}/{total_voxels} voxels)")
    # if voxel_peak_times:
    #     logging.debug("Top origin regions:")
    #     for _, row in origins.iterrows():
    #         logging.debug(f"- {row['region']} at {row['peak_time']:.1f}ms")

    return {
        'wave_name': wave_name,
        'origins': origins,  # a small DataFrame of earliest 10% voxel-peak times
        'involvement_count': num_involved,
        'involvement_percentage': involvement_percentage,
        'involved_voxels': involved_voxels,
        'window': (window_start, window_end),
        'threshold': threshold,
        'global_max_value': global_max_value,
        'global_max_time': global_max_time
    }


def process_eeg_data_directory(directory_path, subject_condition_mapping, selected_subjects=None, selected_nights=None, visualize_regions=True, process_origins=True, source_dir=None):
    """
    Process the EEG data directory using subject-condition mapping and optional subject/night filtering.
    Uses adaptive threshold based on 100th percentile of combined pre-stim distribution.
    
    Args:
        directory_path: Path to the EEG data directory
        subject_condition_mapping: Dictionary mapping subject IDs to conditions (Active/SHAM)
        selected_subjects: List of subject IDs to process (if None, process all)
        selected_nights: List of night IDs to process (if None, process all)
        visualize_regions: If True, generate region visualizations
        process_origins: If True, process origin analysis; if False, skip origin processing
        source_dir: Source directory where data is read from, used to construct output path
    """
    logging.info(f"Processing EEG data directory: {directory_path}")

    # Define the expected treatment groups
    treatment_groups = ["Active", "SHAM"]
    # New structure: {group: {subject: {protocol: {stage: [wave_results]}}}}
    results_by_treatment_group = {group: {} for group in treatment_groups}
    total_files = 0
    processed_files = 0
    error_files = 0
    
    # Get all subject directories
    subject_dirs = [d for d in Path(directory_path).iterdir() if d.is_dir() and d.name.startswith("Subject_")]
    
    # Filter subjects if specified
    if selected_subjects:
        subject_dirs = [d for d in subject_dirs if d.name in selected_subjects]

    if not subject_dirs:
        logging.warning(f"No subject directories found matching selection in {directory_path}.")
        return None

    logging.info(f"\n=== TWO-PASS ANALYSIS ===")
    logging.info(f"Pass 1: Collecting proto1-8 pre-stim data to calculate adaptive threshold")
    logging.info(f"Pass 2: Processing all data with calculated threshold")
    logging.info(f"\nProcessing {len(subject_dirs)} subjects...")

    # PASS 1: Collect proto1-8 pre-stim data to calculate threshold
    all_prestim_values = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        # Skip if subject not in mapping
        if subject_id not in subject_condition_mapping:
            continue
            
        # Get night directories
        night_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("Night")]
        
        # Filter nights if specified
        if selected_nights:
            night_dirs = [d for d in night_dirs if d.name in selected_nights]
            
        for night_dir in night_dirs:
            # Navigate to the SourceRecon directory
            source_recon_dir = night_dir / "Output" / "SourceRecon"
            if not source_recon_dir.exists():
                continue
                
            # Collect proto1-8 pre-stim CSV files
            csv_files = list(source_recon_dir.glob('*proto*pre*.csv'))
            for csv_file in csv_files:
                # Filter to only include proto1-8
                filename = csv_file.name
                protocol_match = re.search(r'proto(\d+)', filename, re.IGNORECASE)
                if protocol_match:
                    proto_num = int(protocol_match.group(1))
                    if proto_num < 1 or proto_num > 8:
                        continue  # Skip protocols outside the 1-8 range
                else:
                    continue  # Skip files without recognizable protocol pattern
                
                try:
                    df = pd.read_csv(csv_file)
                    # Extract activity values from pre-stim data
                    if 'Time' in df.columns:
                        numeric_cols = []
                        for col in df.columns[1:]:
                            if not str(col).startswith('Unnamed'):
                                try:
                                    float(col)
                                    numeric_cols.append(col)
                                except ValueError:
                                    continue
                        
                        if numeric_cols:
                            # Get absolute values and flatten for threshold calculation
                            data = np.abs(df.loc[:, numeric_cols].values)
                            all_prestim_values.extend(data.flatten())
                except Exception as e:
                    logging.warning(f"Error reading pre-stim file {csv_file}: {str(e)}")
    
    # Calculate adaptive threshold (95th percentile of combined proto1-8 pre-stim distribution)
    if all_prestim_values:
        adaptive_threshold = np.percentile(all_prestim_values, 95)
        logging.info(f"\nAdaptive threshold calculated: {adaptive_threshold:.6e} (95th percentile of proto1-8 pre-stim data)")
        logging.info(f"Based on {len(all_prestim_values)} proto1-8 pre-stim data points from {len(subject_dirs)} subjects")
    else:
        logging.error("No proto1-8 pre-stim data found for threshold calculation!")
        return None

    # PASS 2: Process all data with the calculated threshold
    logging.info(f"\n=== PASS 2: Processing all data with adaptive threshold ===")
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Skip if subject not in mapping
        if subject_id not in subject_condition_mapping:
            logging.warning(f"Subject {subject_id} not found in condition mapping. Skipping.")
            continue
        # Get treatment group for this subject
        group = subject_condition_mapping[subject_id]
        if group not in treatment_groups:
            logging.warning(f"Unknown treatment group '{group}' for {subject_id}. Skipping.")
            continue

        # Initialize subject entry if not present
        if subject_id not in results_by_treatment_group[group]:
            results_by_treatment_group[group][subject_id] = {}

        logging.info(f"Processing {subject_id} (Group: {group})...")

        # Get night directories
        night_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("Night")]

        # Filter nights if specified
        if selected_nights:
            night_dirs = [d for d in night_dirs if d.name in selected_nights]

        if not night_dirs:
            logging.warning(f"No selected night directories found for {subject_id}.")
            continue

        for night_dir in night_dirs:
            night_id = night_dir.name
            logging.info(f"  Processing {night_id}...")

            # Navigate to the SourceRecon directory
            source_recon_dir = night_dir / "Output" / "SourceRecon"
            if not source_recon_dir.exists():
                logging.warning(f"SourceRecon directory not found for {subject_id}/{night_id}. Skipping.")
                continue
            # Process the CSV files in the SourceRecon directory for this specific night
            # subject_night_results has format {protocol: {stage: [wave_results]}}
            subject_night_results = process_directory(source_recon_dir, quiet=True, visualize_regions=visualize_regions, process_origins=process_origins, source_dir=source_dir, fixed_threshold=adaptive_threshold)

            if subject_night_results:
                # Merge these night results into the subject's overall results
                for protocol in subject_night_results:
                    if protocol not in results_by_treatment_group[group][subject_id]:
                        results_by_treatment_group[group][subject_id][protocol] = {}

                    for stage in subject_night_results[protocol]:
                        if stage not in results_by_treatment_group[group][subject_id][protocol]:
                            results_by_treatment_group[group][subject_id][protocol][stage] = []
                        # Extend the list of wave results for this subject, protocol, stage
                        results_by_treatment_group[group][subject_id][protocol][stage].extend(subject_night_results[protocol][stage])

                # Update file counts (can be simplified later if needed)
                for protocol in subject_night_results:
                    for stage in subject_night_results[protocol]:
                        # This count might be slightly off if files are processed multiple times across nights,
                        # but gives a general idea. A more accurate count would track unique files.
                        # For now, just count processed waves.
                        num_waves_processed = len(subject_night_results[protocol][stage])
                        processed_files += num_waves_processed
                        # total_files could be calculated once at the start if needed accurately
                        # error_files count is handled within process_directory

    # Recalculate total_files based on collected data for accuracy
    total_files = processed_files # Approximation for now
    # Get error count from process_directory if possible, or keep track here
    # For now, error_files is not accurately tracked at this level, rely on process_directory logs

    logging.info(f"\nProcessing summary:")
    # logging.info(f"Total files found (approx): {total_files}") # This count is inaccurate
    logging.info(f"Total waves successfully processed: {processed_files}")
    # logging.info(f"Errors encountered: {error_files}") # Error count not reliable here

    if not any(results_by_treatment_group.values()):
        logging.error("No data was processed successfully.")
        return None

    return results_by_treatment_group


def process_directory(directory_path, quiet=False, visualize_regions=True, process_origins=True, source_dir=None, fixed_threshold=None):
    """
    Process all CSV files in the directory and organize by protocol and stage.
    
    Args:
        directory_path: Path to the directory containing CSV files
        quiet: If True, suppress most output
        visualize_regions: If True, generate region visualizations
        process_origins: If True, process origin analysis; if False, skip origin processing
        source_dir: Source directory where data is read from, used to construct output path
        fixed_threshold: If provided, use this threshold instead of calculating from data
    """
    # Use logging instead of quiet flag
    logging.info(f"Processing directory: {directory_path}")

    csv_files = list(Path(directory_path).glob('*.csv'))
    if not csv_files:
        logging.warning(f"No CSV files found in: {directory_path}")
        return None

    protocol_pattern = r'(proto\d+)'
    stage_pattern = r'(pre|early|late|post|stim)(?:-stim)?'

    results_by_protocol = {}
    total_files = len(csv_files)
    processed_files = 0
    error_files = 0

    for csv_file in csv_files:
        filename = csv_file.name
        protocol_match = re.search(protocol_pattern, filename, re.IGNORECASE)
        stage_match = re.search(stage_pattern, filename, re.IGNORECASE)
        if protocol_match and stage_match:
            protocol = protocol_match.group(1).lower()  # ensure consistent case (proto1, proto2, etc.)
            stage = stage_match.group(1).lower()
            
            # Filter to only process proto1-8
            protocol_number = protocol.replace('proto', '')
            try:
                proto_num = int(protocol_number)
                if proto_num < 1 or proto_num > 8:
                    continue  # Skip protocols outside the 1-8 range
            except ValueError:
                continue  # Skip if protocol number cannot be parsed
            if protocol not in results_by_protocol:
                results_by_protocol[protocol] = {}
            
            # Make sure the stage exists in the protocol's results
            if stage not in results_by_protocol[protocol]:
                results_by_protocol[protocol][stage] = []

            try:
                df = pd.read_csv(csv_file)
                wave_name = csv_file.stem
                # Call analyze_slow_wave with process_origins and fixed_threshold parameters
                result = analyze_slow_wave(df, wave_name, process_origins=process_origins, fixed_threshold=fixed_threshold)
                # Validate, log, and save wave result information
                validate_wave_result(result, csv_file) # Handles logging internally

                results_by_protocol[protocol][stage].append(result)
                processed_files += 1

                # Generate region time series visualization
                if visualize_regions:
                    # Use source_dir if provided, otherwise use the standard 'results' directory
                    visualize_region_time_series(result, csv_file, source_dir=source_dir) # output_dir handled internally

                # logging.info(f"Processed {filename} - Protocol: {protocol}, Stage: {stage}") # Logged by validate_wave_result
            except Exception as e:
                error_files += 1
                logging.error(f"Error processing {filename}: {str(e)}")

    logging.info(f"\nDirectory processing summary for: {directory_path}")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Successfully processed: {processed_files}")
    logging.info(f"Errors: {error_files}")

    return results_by_protocol


# Removed analyze_protocol_results function as requested


def analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list, min_occurrence_threshold=3):
    """
    Perform overall treatment group comparison by collapsing subjects, nights, and protos.
    Includes both between-group (Active vs SHAM) and within-group (stage) comparisons.
    """
    logging.info("\n=== Overall Treatment Group Comparison (Collapsing Subjects, Nights, and Protos) ===")

    # Collect all possible stages across all groups, subjects, and protocols
    all_stages = set()
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            for protocol in results_by_treatment_group[group][subject]:
                all_stages.update(results_by_treatment_group[group][subject][protocol].keys())

    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order
    
    # Collect all waves for each treatment group across all subjects, protocols, and stages
    all_waves_by_group = {'Active': {}, 'SHAM': {}}
    groups = list(results_by_treatment_group.keys())
    logging.debug(f"Starting aggregation for all_waves_by_group. Groups: {groups}")
    for group in groups:
        logging.debug(f"Processing group: {group}")
        all_waves_by_group[group] = {stage: [] for stage in stages} # Initialize stages for the group
        subjects_in_group = list(results_by_treatment_group[group].keys())
        logging.debug(f"Subjects in group '{group}': {subjects_in_group}")
        for subject in subjects_in_group:
            logging.debug(f"  Processing subject: {subject}")
            protocols_for_subject = list(results_by_treatment_group[group][subject].keys())
            logging.debug(f"   Protocols for subject '{subject}': {protocols_for_subject}")
            for protocol in protocols_for_subject:
                logging.debug(f"    Processing protocol: {protocol}")
                stages_for_protocol = list(results_by_treatment_group[group][subject][protocol].keys())
                logging.debug(f"     Stages for protocol '{protocol}': {stages_for_protocol}")
                for stage in stages: # Iterate through ALL possible stages
                    # Check if the stage exists for this subject/protocol
                    if stage in stages_for_protocol: # Check against actual stages for this proto/subj
                        wave_list = results_by_treatment_group[group][subject][protocol][stage]
                        logging.debug(f"      Extending stage '{stage}' for group '{group}' with {len(wave_list)} items. Source: {group}/{subject}/{protocol}/{stage}")
                        # Check type before extending
                        if not isinstance(wave_list, list):
                             logging.error(f"Expected list but got {type(wave_list)} for {group}/{subject}/{protocol}/{stage}")
                             continue
                        if wave_list and not all(isinstance(item, dict) for item in wave_list):
                             logging.error(f"Expected list of dicts but found other types in {group}/{subject}/{protocol}/{stage}")
                             # Log the non-dict items
                             for idx, item in enumerate(wave_list):
                                 if not isinstance(item, dict):
                                     logging.error(f"ERROR item index {idx}: type={type(item)}, value='{item}'")
                             continue
                        # If checks pass, extend
                        all_waves_by_group[group][stage].extend(wave_list)
                    # else:
                    #    logging.debug(f"      Stage '{stage}' not found for {group}/{subject}/{protocol}")


    # Calculate involvement (using subject-weighted averaging)
    involvement_stats = {}
    involvement_data = {}
    groups = list(all_waves_by_group.keys())
    # stages already defined above with detected stages

    for group in groups:
        # Calculate subject means first, then group means
        subject_means_by_stage = {}
        logging.debug(f"\nCalculating subject-weighted involvement for group '{group}'...")
        for stage in stages:
            subject_means = []
            # Get subjects for this group across all protocols
            for subject in results_by_treatment_group[group]:
                # Collect all waves for this subject across all protocols for this stage
                subject_involvement_values = []
                for protocol in results_by_treatment_group[group][subject]:
                    if stage in results_by_treatment_group[group][subject][protocol]:
                        subject_waves = results_by_treatment_group[group][subject][protocol][stage]
                        for res in subject_waves:
                            if isinstance(res, dict) and 'involvement_percentage' in res:
                                subject_involvement_values.append(res['involvement_percentage'])
                
                # Calculate this subject's mean for this stage
                if subject_involvement_values:
                    subject_mean = np.mean(subject_involvement_values)
                    subject_means.append(subject_mean)
                    logging.debug(f"  Subject {subject}, Stage {stage}: {len(subject_involvement_values)} waves, mean = {subject_mean:.2f}%")
            
            subject_means_by_stage[stage] = subject_means
            logging.debug(f"  Stage {stage}: {len(subject_means)} subjects with data")

        # Calculate group statistics from subject means (subject-weighted)
        stats_by_stage = {
            stage: calculate_involvement_statistics(subject_means_by_stage[stage])
            for stage in stages
        }
        involvement_stats[group] = stats_by_stage
        involvement_data[group] = subject_means_by_stage

    logging.info("\nOverall Involvement Statistics by Treatment Group (All Protos Combined):")
    for group in groups:
        logging.info(f"\n{group} Group:")
        for stage in stages:
            sdata = involvement_stats[group][stage]
            logging.info(f"  {stage.capitalize()}: {sdata['mean']:.1f}% ± {sdata['std']:.1f}% (n={sdata['count']})")

    # Perform between-group statistical tests (Active vs SHAM)
    between_group_tests = []
    if 'Active' in involvement_data and 'SHAM' in involvement_data:
        for stage in stages:
            active_data = involvement_data['Active'][stage]
            sham_data = involvement_data['SHAM'][stage]
            if active_data and sham_data:
                try:
                    u_stat, p_val = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                    between_group_tests.append({
                        'Analysis_Type': 'Overall',
                        'Comparison_Type': 'Between_Group',
                        'Test': 'Mann-Whitney U',
                        'Metric': f'Involvement: Active vs SHAM ({stage})',
                        'Stage': stage,
                        'Protocol': 'Overall',
                        'Statistic': u_stat,
                        'P_Value': p_val,
                        'Significant': p_val < 0.05
                    })
                    logging.info(f"\nMann-Whitney U Test for overall {stage} involvement (Active vs SHAM): "
                          f"U={u_stat:.2f}, p={p_val:.4f}")
                    if p_val < 0.05:
                        logging.info(f"Significant difference detected in overall {stage} involvement between Active and SHAM groups.")
                except Exception as e:
                    logging.error(f"Error performing statistical test for overall {stage} involvement: {str(e)}")

    # Perform within-group statistical tests (paired comparisons across stages)
    within_group_tests = []
    groups = list(involvement_data.keys())
    
    for group in groups:
        logging.info(f"\n--- Within-Group Stage Comparison for {group} Group (Overall) ---")
        
        # Prepare paired data for this group (subject-level means)
        subject_stage_data = {}
        for subject in results_by_treatment_group[group]:
            subject_stage_data[subject] = {}
            for stage in stages:
                # Collect all involvement values for this subject across all protocols for this stage
                subject_values = []
                for protocol in results_by_treatment_group[group][subject]:
                    if stage in results_by_treatment_group[group][subject][protocol]:
                        for wave_result in results_by_treatment_group[group][subject][protocol][stage]:
                            if isinstance(wave_result, dict) and 'involvement_percentage' in wave_result:
                                subject_values.append(wave_result['involvement_percentage'])
                
                if subject_values:
                    subject_stage_data[subject][stage] = np.mean(subject_values)
        
        # Create paired data arrays for each stage combination
        valid_stages = [s for s in stages if any(s in subject_stage_data[subj] for subj in subject_stage_data)]
        if len(valid_stages) >= 2:
            comparisons = list(itertools.combinations(valid_stages, 2))
            pairwise_results_raw = []
            
            for stage1, stage2 in comparisons:
                # Get paired data for subjects who have both stages
                paired_data_stage1 = []
                paired_data_stage2 = []
                
                for subject in subject_stage_data:
                    if stage1 in subject_stage_data[subject] and stage2 in subject_stage_data[subject]:
                        paired_data_stage1.append(subject_stage_data[subject][stage1])
                        paired_data_stage2.append(subject_stage_data[subject][stage2])
                
                if len(paired_data_stage1) >= 3:  # Need at least 3 paired observations
                    try:
                        # Use Wilcoxon signed-rank test for paired data
                        stat, p_val = scipy_stats.wilcoxon(paired_data_stage1, paired_data_stage2, alternative='two-sided')
                        pairwise_results_raw.append({
                            'stage1': stage1,
                            'stage2': stage2,
                            'statistic': stat,
                            'p_value_raw': p_val,
                            'n_subjects': len(paired_data_stage1)
                        })
                    except Exception as e:
                        logging.warning(f"Error performing Wilcoxon test for {group} {stage1} vs {stage2}: {str(e)}")
            
            # Apply FDR correction if we have multiple comparisons
            if pairwise_results_raw:
                pvals_raw = [res['p_value_raw'] for res in pairwise_results_raw]
                reject, pvals_corrected, _, _ = multipletests(pvals_raw, method='fdr_bh', alpha=0.05)
                
                for i, raw_res in enumerate(pairwise_results_raw):
                    within_group_tests.append({
                        'Analysis_Type': 'Overall',
                        'Comparison_Type': 'Within_Group',
                        'Test': 'Wilcoxon Signed-Rank',
                        'Metric': f'{group} Involvement: {raw_res["stage1"]} vs {raw_res["stage2"]}',
                        'Stage': f'{raw_res["stage1"]}_vs_{raw_res["stage2"]}',
                        'Protocol': 'Overall',
                        'Treatment_Group': group,
                        'Statistic': raw_res['statistic'],
                        'P_Value': pvals_corrected[i],
                        'Significant': reject[i],
                        'N_Subjects': raw_res['n_subjects']
                    })
                    
                    logging.info(f"Wilcoxon test for {group} {raw_res['stage1']} vs {raw_res['stage2']}: "
                          f"W={raw_res['statistic']:.2f}, p={pvals_corrected[i]:.4f} (n={raw_res['n_subjects']})")

    # Calculate origin data
    origin_data = {}
    for group in groups:
        origin_by_stage = {}
        for stage in stages:
            stage_results = all_waves_by_group[group][stage]
            total_waves = len(stage_results)
            region_counts = calculate_origin_statistics(stage_results, stage, total_waves)
            origin_by_stage[stage] = {
                'region_counts': region_counts,
                'total_waves': total_waves
            }
        origin_data[group] = origin_by_stage

    logging.info("\nOverall Origin Distribution by Treatment Group (All Protos Combined):")
    for group in groups:
        logging.info(f"\n{group} Group:")
        for stage in stages:
            rc = origin_data[group][stage]['region_counts']
            tw = origin_data[group][stage]['total_waves']
            if rc and tw > 0:
                logging.info(f"  {stage.capitalize()} (n={tw} waves):")
                # Show top 10 by count
                region_counts_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
                displayed = 0
                for (region, count) in region_counts_sorted:
                    if displayed >= 10:
                        break
                    percentage = (count / tw) * 100
                    logging.info(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                    displayed += 1
            else:
                logging.info(f"  {stage.capitalize()}: No origins detected")

    # Perform statistical tests on origin distribution
    origin_comparison_tests = []
    if 'Active' in origin_data and 'SHAM' in origin_data:
        for stage in stages:
            active_stage = origin_data['Active'][stage]
            sham_stage = origin_data['SHAM'][stage]
            if active_stage and sham_stage:
                active_rc = active_stage['region_counts']
                sham_rc = sham_stage['region_counts']
                all_regions = set(active_rc.keys()) | set(sham_rc.keys())
                if all_regions:
                    active_counts = [active_rc.get(r, 0) for r in all_regions]
                    sham_counts = [sham_rc.get(r, 0) for r in all_regions]
                    contingency_table = np.array([active_counts, sham_counts])
                    test_result = perform_chi_square_or_fisher_test(contingency_table)
                    if test_result:
                        test_result['Metric'] = f'Overall Origin Distribution: Active vs SHAM ({stage})'
                        test_result['Stage'] = stage
                        origin_comparison_tests.append(test_result)

                        if 'DF' in test_result:
                            logging.info(f"\n{test_result['Test']} for overall {stage} origin distribution (Active vs SHAM): "
                                  f"χ²={test_result['Statistic']:.2f}, df={test_result['DF']}, p={test_result['P_Value']:.4f}")
                        else:
                            logging.info(f"\n{test_result['Test']} for overall {stage} origin distribution (Active vs SHAM): "
                                  f"p={test_result['P_Value']:.4f}")
                        if test_result['Significant']:
                            logging.info(f"Significant difference detected in overall {stage} origin distribution "
                                  f"between Active and SHAM groups.")

    return {
        'overall_involvement_data': involvement_data,
        'overall_involvement_stats': involvement_stats,
        'overall_origin_data': origin_data,
        'overall_comparison_tests': between_group_tests + within_group_tests + origin_comparison_tests,
        'master_region_list': master_region_list
    }


def analyze_proto_specific_comparison(results_by_treatment_group, master_region_list, min_occurrence_threshold=2):
    """
    Perform proto-specific comparison between treatment groups for each proto.
    For proto1, also collect subject-specific involvement data.
    """
    logging.info("\n=== Proto-Specific Comparison ===")
    # Gather all protocols correctly
    all_protocols = set()
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            all_protocols.update(results_by_treatment_group[group][subject].keys())

    # Collect all possible stages across all groups, subjects, and protocols
    all_stages = set()
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            for protocol in results_by_treatment_group[group][subject]:
                all_stages.update(results_by_treatment_group[group][subject][protocol].keys())

    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order

    proto_specific_results = {}
    groups = list(results_by_treatment_group.keys())
    
    # Initialize subject-specific data container for proto1
    proto1_subject_specific_data = {}

    # Ensure the main loop iterates through PROTOCOLS, not subjects or other keys
    for protocol in sorted(list(all_protocols)): # Explicitly sort the list of protocols
        logging.info(f"\n--- Analysis for {protocol} ---")

        # Check if at least two groups have data for this specific protocol
        groups_with_data_for_protocol = []
        for group in groups:
            group_has_protocol = False
            for subject in results_by_treatment_group[group]:
                if protocol in results_by_treatment_group[group][subject]:
                    group_has_protocol = True
                    break
            if group_has_protocol:
                groups_with_data_for_protocol.append(group)

        if len(groups_with_data_for_protocol) < 2:
            logging.warning(f"Skipping {protocol} - not enough treatment groups with data for this protocol.")
            continue

        # Collect waves for each group, for this specific protocol, aggregating across subjects
        waves_by_group = {}
        for group in groups_with_data_for_protocol: # Use the correctly filtered list
            waves_by_group[group] = {stage: [] for stage in stages} # Initialize stages for the group/protocol
            for subject in results_by_treatment_group[group]:
                 # Check if subject has data for this protocol
                 if protocol in results_by_treatment_group[group][subject]:
                     for stage in stages:
                         # Check if stage exists for this subject/protocol
                         if stage in results_by_treatment_group[group][subject][protocol]:
                             # Extend the list with wave result dictionaries
                             waves_by_group[group][stage].extend(results_by_treatment_group[group][subject][protocol][stage])

        # Collect subject-specific involvement data for proto1 only
        if protocol.lower() == 'proto1':
            logging.info(f"Collecting subject-specific involvement data for {protocol}")
            proto1_subject_specific_data = {group: {stage: {} for stage in stages} for group in groups_with_data_for_protocol}
            
            for group in groups_with_data_for_protocol:
                for subject in results_by_treatment_group[group]:
                    if protocol in results_by_treatment_group[group][subject]:
                        for stage in stages:
                            if stage in results_by_treatment_group[group][subject][protocol]:
                                # Collect individual involvement percentages for this subject
                                subject_involvement_values = [
                                    res['involvement_percentage'] 
                                    for res in results_by_treatment_group[group][subject][protocol][stage] 
                                    if isinstance(res, dict) and 'involvement_percentage' in res
                                ]
                                if subject_involvement_values:
                                    proto1_subject_specific_data[group][stage][subject] = subject_involvement_values
                                    
            # Log summary of subject-specific data collected
            for group in groups_with_data_for_protocol:
                subject_count = len([s for s in proto1_subject_specific_data[group]['pre'].keys() if proto1_subject_specific_data[group]['pre'][s]])
                logging.info(f"Proto1 subject-specific data: {group} group has {subject_count} subjects with data")

        # Involvement (using subject-weighted averaging)
        involvement_stats = {}
        involvement_data = {}
        for group in waves_by_group: # Iterate through the corrected waves_by_group
            # Calculate subject means first, then group means
            subject_means_by_stage = {}
            for stage in stages:
                subject_means = []
                # Get subjects for this group that have this protocol
                for subject in results_by_treatment_group[group]:
                    if protocol in results_by_treatment_group[group][subject]:
                        if stage in results_by_treatment_group[group][subject][protocol]:
                            # Get all waves for this subject/protocol/stage
                            subject_waves = results_by_treatment_group[group][subject][protocol][stage]
                            subject_involvement_values = [
                                res['involvement_percentage'] 
                                for res in subject_waves 
                                if isinstance(res, dict) and 'involvement_percentage' in res
                            ]
                            if subject_involvement_values:
                                subject_mean = np.mean(subject_involvement_values)
                                subject_means.append(subject_mean)
                subject_means_by_stage[stage] = subject_means

            # Calculate group statistics from subject means (subject-weighted)
            stats_by_stage = {
                stage: calculate_involvement_statistics(subject_means_by_stage[stage])
                for stage in stages
            }
            involvement_stats[group] = stats_by_stage
            involvement_data[group] = subject_means_by_stage

        logging.info(f"\nInvolvement Statistics for {protocol} by Treatment Group:")
        for group in waves_by_group:
            logging.info(f"\n{group} Group:")
            for stage, stats in involvement_stats[group].items():
                logging.info(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']})")

        # Between-group statistical tests (Active vs SHAM)
        between_group_tests = []
        if 'Active' in involvement_data and 'SHAM' in involvement_data:
            for stage in stages:
                active_data = involvement_data['Active'].get(stage, [])
                sham_data = involvement_data['SHAM'].get(stage, [])
                if active_data and sham_data:
                    try:
                        u_stat, p_val = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                        between_group_tests.append({
                            'Analysis_Type': protocol,
                            'Comparison_Type': 'Between_Group',
                            'Test': 'Mann-Whitney U',
                            'Protocol': protocol,
                            'Metric': f'Involvement: Active vs SHAM ({stage})',
                            'Stage': stage,
                            'Statistic': u_stat,
                            'P_Value': p_val,
                            'Significant': p_val < 0.05
                        })
                        logging.info(f"\nMann-Whitney U Test for {stage} involvement in {protocol} (Active vs SHAM): "
                              f"U={u_stat:.2f}, p={p_val:.4f}")
                        if p_val < 0.05:
                            logging.info(f"Significant difference detected in {stage} involvement between Active and SHAM groups for {protocol}.")
                    except Exception as e:
                        logging.error(f"Error performing statistical test for {stage} involvement in {protocol}: {str(e)}")

        # Within-group statistical tests (paired comparisons across stages)
        within_group_tests = []
        
        for group in groups_with_data_for_protocol:
            logging.info(f"\n--- Within-Group Stage Comparison for {group} Group ({protocol}) ---")
            
            # Prepare paired data for this group and protocol
            subject_stage_data = {}
            for subject in results_by_treatment_group[group]:
                if protocol in results_by_treatment_group[group][subject]:
                    subject_stage_data[subject] = {}
                    for stage in stages:
                        if stage in results_by_treatment_group[group][subject][protocol]:
                            # Get all involvement values for this subject/protocol/stage
                            subject_values = []
                            for wave_result in results_by_treatment_group[group][subject][protocol][stage]:
                                if isinstance(wave_result, dict) and 'involvement_percentage' in wave_result:
                                    subject_values.append(wave_result['involvement_percentage'])
                            
                            if subject_values:
                                subject_stage_data[subject][stage] = np.mean(subject_values)
            
            # Create paired data arrays for each stage combination
            valid_stages = [s for s in stages if any(s in subject_stage_data[subj] for subj in subject_stage_data)]
            if len(valid_stages) >= 2:
                comparisons = list(itertools.combinations(valid_stages, 2))
                pairwise_results_raw = []
                
                for stage1, stage2 in comparisons:
                    # Get paired data for subjects who have both stages
                    paired_data_stage1 = []
                    paired_data_stage2 = []
                    
                    for subject in subject_stage_data:
                        if stage1 in subject_stage_data[subject] and stage2 in subject_stage_data[subject]:
                            paired_data_stage1.append(subject_stage_data[subject][stage1])
                            paired_data_stage2.append(subject_stage_data[subject][stage2])
                    
                    if len(paired_data_stage1) >= 3:  # Need at least 3 paired observations
                        try:
                            # Use Wilcoxon signed-rank test for paired data
                            stat, p_val = scipy_stats.wilcoxon(paired_data_stage1, paired_data_stage2, alternative='two-sided')
                            pairwise_results_raw.append({
                                'stage1': stage1,
                                'stage2': stage2,
                                'statistic': stat,
                                'p_value_raw': p_val,
                                'n_subjects': len(paired_data_stage1)
                            })
                        except Exception as e:
                            logging.warning(f"Error performing Wilcoxon test for {group} {stage1} vs {stage2} in {protocol}: {str(e)}")
                
                # Apply FDR correction if we have multiple comparisons
                if pairwise_results_raw:
                    pvals_raw = [res['p_value_raw'] for res in pairwise_results_raw]
                    reject, pvals_corrected, _, _ = multipletests(pvals_raw, method='fdr_bh', alpha=0.05)
                    
                    for i, raw_res in enumerate(pairwise_results_raw):
                        within_group_tests.append({
                            'Analysis_Type': protocol,
                            'Comparison_Type': 'Within_Group',
                            'Test': 'Wilcoxon Signed-Rank',
                            'Protocol': protocol,
                            'Metric': f'{group} Involvement: {raw_res["stage1"]} vs {raw_res["stage2"]}',
                            'Stage': f'{raw_res["stage1"]}_vs_{raw_res["stage2"]}',
                            'Treatment_Group': group,
                            'Statistic': raw_res['statistic'],
                            'P_Value': pvals_corrected[i],
                            'Significant': reject[i],
                            'N_Subjects': raw_res['n_subjects']
                        })
                        
                        logging.info(f"Wilcoxon test for {group} {raw_res['stage1']} vs {raw_res['stage2']} in {protocol}: "
                              f"W={raw_res['statistic']:.2f}, p={pvals_corrected[i]:.4f} (n={raw_res['n_subjects']})")

        # Origin data
        origin_data = {}
        for group in waves_by_group:
            origin_by_stage = {}
            for stage in stages:
                stage_results = waves_by_group[group][stage]
                total_waves = len(stage_results)
                region_counts = calculate_origin_statistics(stage_results, stage, total_waves)
                origin_by_stage[stage] = {
                    'region_counts': region_counts,
                    'total_waves': total_waves
                }
            origin_data[group] = origin_by_stage

        logging.info(f"\nOrigin Distribution for {protocol} by Treatment Group:")
        for group in waves_by_group:
            logging.info(f"\n{group} Group:")
            for stage, data_dict in origin_data[group].items():
                rc = data_dict['region_counts']
                tw = data_dict['total_waves']
                if rc and tw > 0:
                    logging.info(f"  {stage.capitalize()} (n={tw} waves):")
                    # Show top 5
                    region_counts_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
                    displayed = 0
                    for (region, count) in region_counts_sorted:
                        if displayed >= 5:
                            break
                        percentage = (count / tw) * 100
                        logging.info(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                        displayed += 1
                else:
                    logging.info(f"  {stage.capitalize()}: No origins detected")

        # Statistical tests on origin distribution
        origin_comparison_tests = []
        if 'Active' in origin_data and 'SHAM' in origin_data:
            for stage in stages:
                active_stage = origin_data['Active'][stage]
                sham_stage = origin_data['SHAM'][stage]
                if active_stage and sham_stage:
                    active_rc = active_stage['region_counts']
                    sham_rc = sham_stage['region_counts']
                    all_regions = set(active_rc.keys()) | set(sham_rc.keys())
                    if all_regions:
                        active_counts = [active_rc.get(r, 0) for r in all_regions]
                        sham_counts = [sham_rc.get(r, 0) for r in all_regions]
                        contingency_table = np.array([active_counts, sham_counts])
                        test_result = perform_chi_square_or_fisher_test(contingency_table)
                        if test_result:
                            test_result['Protocol'] = protocol
                            test_result['Metric'] = f'Origin Distribution: Active vs SHAM ({stage})'
                            test_result['Stage'] = stage
                            origin_comparison_tests.append(test_result)

                            if 'DF' in test_result:
                                logging.info(f"\n{test_result['Test']} for {stage} origin distribution in {protocol} "
                                      f"(Active vs SHAM): χ²={test_result['Statistic']:.2f}, df={test_result['DF']}, "
                                      f"p={test_result['P_Value']:.4f}")
                            else:
                                logging.info(f"\n{test_result['Test']} for {stage} origin distribution in {protocol} "
                                      f"(Active vs SHAM): p={test_result['P_Value']:.4f}")
                            if test_result['Significant']:
                                logging.info(f"Significant difference detected in {stage} origin distribution between Active and SHAM groups for {protocol}.")

        proto_specific_results[protocol] = {
            'involvement_data': involvement_data,
            'involvement_stats': involvement_stats,
            'origin_data': origin_data,
            # Combine all test types: between-group, within-group, and origin tests
            'comparison_tests': between_group_tests + within_group_tests + origin_comparison_tests
        }
        
        # Add subject-specific data for proto1 only
        if protocol.lower() == 'proto1' and proto1_subject_specific_data:
            proto_specific_results[protocol]['subject_specific_involvement'] = proto1_subject_specific_data

    return {
        'proto_specific_results': proto_specific_results,
        'master_region_list': master_region_list,
        'proto1_subject_specific_data': proto1_subject_specific_data if 'proto1' in [p.lower() for p in all_protocols] else None
    }


def analyze_within_group_stage_comparison(results_by_treatment_group, master_region_list):
    """
    Perform within-group stage comparison for each treatment group.
    """
    logging.info("\n=== Within-Group Stage Comparison ===")

    # Collect all possible stages across all groups, subjects, and protocols
    all_stages = set()
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            for protocol in results_by_treatment_group[group][subject]:
                all_stages.update(results_by_treatment_group[group][subject][protocol].keys())

    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order

    # Input structure: {group: {subject: {protocol: {stage: [wave_results]}}}}

    within_group_results = {}
    for group in results_by_treatment_group:
        logging.info(f"\n--- Analysis for {group} Group ---")

        group_data = results_by_treatment_group[group]
        subjects = list(group_data.keys())
        if not subjects:
            logging.warning(f"No subjects found for {group} group.")
            continue

        # --- Prepare Paired Involvement Data ---
        # We need one value per subject per stage (e.g., average involvement)
        subject_stage_involvement = {subject: {stage: [] for stage in stages} for subject in subjects}
        for subject in subjects:
            for protocol in group_data[subject]:
                for stage in stages:
                    if stage in group_data[subject][protocol]:
                        # Collect all involvement percentages for this subject/protocol/stage
                        inv_percentages = [res['involvement_percentage'] for res in group_data[subject][protocol][stage]]
                        if inv_percentages:
                            subject_stage_involvement[subject][stage].extend(inv_percentages)

        # Calculate average involvement per subject per stage
        paired_involvement_data = {stage: [] for stage in stages}
        valid_subjects_for_pairing = [] # Subjects with data for ALL stages being compared
        
        # Determine which subjects have data for all relevant stages
        stages_with_data = [s for s in stages if any(subject_stage_involvement[subj][s] for subj in subjects)]
        if len(stages_with_data) < 2:
             logging.warning(f"Not enough stages with data for paired comparison in {group} group.")
             # Continue to origin analysis, skip involvement tests
        else:
            logging.info(f"Preparing paired data for stages: {', '.join(stages_with_data)}")
            for subject in subjects:
                has_all_stages = True
                subject_averages = {}
                for stage in stages_with_data:
                    scores = subject_stage_involvement[subject][stage]
                    if scores:
                        subject_averages[stage] = np.mean(scores)
                    else:
                        has_all_stages = False
                        break # Subject missing data for this stage
                
                if has_all_stages:
                    valid_subjects_for_pairing.append(subject)
                    for stage in stages_with_data:
                        paired_involvement_data[stage].append(subject_averages[stage])

            logging.info(f"Found {len(valid_subjects_for_pairing)} subjects with complete data across stages for paired tests.")

            # --- Perform Wilcoxon Signed-Rank Tests on Involvement (Paired) ---
            involvement_test_results = []
            
            if len(valid_subjects_for_pairing) >= 3 and len(stages_with_data) >= 2:
                comparisons = list(itertools.combinations(stages_with_data, 2))
                pairwise_results_raw = []
                
                for stage1, stage2 in comparisons:
                    # Get paired data for subjects who have both stages
                    paired_data_stage1 = paired_involvement_data[stage1]
                    paired_data_stage2 = paired_involvement_data[stage2]
                    
                    if len(paired_data_stage1) >= 3:  # Need at least 3 paired observations
                        try:
                            # Use Wilcoxon signed-rank test for paired data
                            stat, p_val = scipy_stats.wilcoxon(paired_data_stage1, paired_data_stage2, alternative='two-sided')
                            pairwise_results_raw.append({
                                'stage1': stage1,
                                'stage2': stage2,
                                'statistic': stat,
                                'p_value_raw': p_val,
                                'n_subjects': len(paired_data_stage1)
                            })
                        except Exception as e:
                            logging.warning(f"Error performing Wilcoxon test for {group} {stage1} vs {stage2}: {str(e)}")
                
                # Apply FDR correction if we have multiple comparisons
                if pairwise_results_raw:
                    pvals_raw = [res['p_value_raw'] for res in pairwise_results_raw]
                    reject, pvals_corrected, _, _ = multipletests(pvals_raw, method='fdr_bh', alpha=0.05)
                    
                    for i, raw_res in enumerate(pairwise_results_raw):
                        involvement_test_results.append({
                            'Analysis_Type': 'Within_Group',
                            'Comparison_Type': 'Within_Group',
                            'Test': 'Wilcoxon Signed-Rank',
                            'Metric': f'{group} Involvement: {raw_res["stage1"]} vs {raw_res["stage2"]}',
                            'Stage': f'{raw_res["stage1"]}_vs_{raw_res["stage2"]}',
                            'Protocol': 'All_Combined',
                            'Treatment_Group': group,
                            'Statistic': raw_res['statistic'],
                            'P_Value': pvals_corrected[i],
                            'Significant': reject[i],
                            'N_Subjects': raw_res['n_subjects']
                        })
                        
                        logging.info(f"Wilcoxon test for {group} {raw_res['stage1']} vs {raw_res['stage2']}: "
                              f"W={raw_res['statistic']:.2f}, p={pvals_corrected[i]:.4f} (n={raw_res['n_subjects']})")
                
                # Log results summary
                if involvement_test_results:
                    logging.info(f"\nWithin-Group Paired Tests for {group} (FDR corrected):")
                    for res in involvement_test_results:
                        stage_comparison = res['Metric'].split(': ')[1]
                        logging.info(f"  {stage_comparison}: W={res['Statistic']:.2f}, p={res['P_Value']:.4f} {'*' if res['Significant'] else ''}")
                else:
                    logging.info(f"No within-group tests performed for {group} group due to insufficient data.")
            else:
                logging.warning(f"Not enough subjects ({len(valid_subjects_for_pairing)}) or stages ({len(stages_with_data)}) for paired tests in {group} group.")


        # --- Calculate Descriptive Involvement Stats (using subject-weighted averaging) ---
        involvement_stats = {}
        logging.info(f"\nOverall Involvement Statistics for {group} Group (Subject-Weighted):")
        for stage in stages:
            # Calculate subject means first, then group statistics
            subject_means = []
            for subject in group_data:
                if subject_stage_involvement[subject][stage]:
                    subject_mean = np.mean(subject_stage_involvement[subject][stage])
                    subject_means.append(subject_mean)
            involvement_stats[stage] = calculate_involvement_statistics(subject_means)
            stats = involvement_stats[stage]
            logging.info(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']} subjects)")


        # --- Calculate Origin Data (Aggregated across subjects for the group) ---
        origin_data = {}
        logging.info(f"\nAggregated Origin Statistics for {group} Group:")
        for stage in stages:
            all_stage_origins = []
            # Aggregate origins across all subjects/protocols for this group/stage
            for subject in group_data:
                for protocol in group_data[subject]:
                    if stage in group_data[subject][protocol]:
                         # Each item in the list is a wave result dict
                         all_stage_origins.extend(group_data[subject][protocol][stage])

            total_waves = len(all_stage_origins)
            # Pass the list of wave result dicts to calculate_origin_statistics
            region_counts = calculate_origin_statistics(all_stage_origins, stage, total_waves)
            origin_data[stage] = {
                'region_counts': region_counts,
                'total_waves': total_waves
            }

        for stage, data_dict in origin_data.items():
            rc = data_dict['region_counts']
            tw = data_dict['total_waves']
            if rc and tw > 0:
                logging.info(f"  {stage.capitalize()} (n={tw} waves):")
                ordered_regions = [r for r in master_region_list if r in rc]
                displayed = 0
                for region in ordered_regions:
                    if displayed >= 5:
                        break
                    count = rc[region]
                    percentage = (count / tw) * 100
                    logging.info(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                    displayed += 1
            else:
                logging.info(f"  {stage.capitalize()}: No origins detected")

        # Perform stats on origin distribution (comparing stages within the group)
        # This uses the aggregated origin counts per stage for the group
        origin_test_results = perform_origin_distribution_tests(origin_data, master_region_list)

        for result in origin_test_results:
            if 'DF' in result:
                logging.info(f"\n{result['Test']} for Origin Distribution in {group} Group: "
                      f"χ²={result['Statistic']:.2f}, df={result['DF']}, p={result['P_Value']:.4f}")
            else:
                logging.info(f"\n{result['Test']} for Origin Distribution in {group} Group: "
                      f"p={result['P_Value']:.4f}")
            if result['Significant']:
                logging.info(f"Significant differences detected in origin distribution between stages for {group} Group.")

        if not origin_test_results:
            logging.warning(f"\nNot enough data for statistical test of origin distribution in {group} Group.")

        within_group_results[group] = {
            # 'involvement_data' now refers to the paired data structure if needed later,
            # but the primary results are in 'involvement_test_results'.
            # We store descriptive stats separately.
            'paired_involvement_data_for_test': paired_involvement_data, # Store data used for tests
            'origin_data': origin_data, # Aggregated origin counts
            'involvement_stats': involvement_stats, # Descriptive stats
            'involvement_test_results': involvement_test_results, # Friedman/Wilcoxon results
            'origin_test_results': origin_test_results # Chi2/Fisher results for origins
        }

    return {
        'within_group_results': within_group_results,
        'master_region_list': master_region_list
    }


def consolidate_statistical_results(overall_results, proto_specific_results, within_group_results):
    """
    Consolidate all statistical test results into a unified format.
    
    Args:
        overall_results: Results from analyze_overall_treatment_comparison
        proto_specific_results: Results from analyze_proto_specific_comparison
        within_group_results: Results from analyze_within_group_stage_comparison
    
    Returns:
        List of all statistical test results in unified format
    """
    logging.info("\n=== Consolidating Statistical Results ===")
    
    all_tests = []
    
    # 1. Overall comparison tests (already includes between and within-group)
    if overall_results and 'overall_comparison_tests' in overall_results:
        overall_tests = overall_results['overall_comparison_tests']
        for test in overall_tests:
            # Ensure consistent format
            test_copy = test.copy()
            if 'Analysis_Type' not in test_copy:
                test_copy['Analysis_Type'] = 'Overall'
            # Standardize test type field
            if 'Test' in test_copy and 'Test_Type' not in test_copy:
                test_copy['Test_Type'] = test_copy['Test']
            all_tests.append(test_copy)
        logging.info(f"Added {len(overall_tests)} tests from overall comparison")
    
    # 2. Protocol-specific comparison tests (already includes between and within-group)
    if proto_specific_results and 'proto_specific_results' in proto_specific_results:
        proto_tests_count = 0
        for protocol, results in proto_specific_results['proto_specific_results'].items():
            if 'comparison_tests' in results:
                protocol_tests = results['comparison_tests']
                for test in protocol_tests:
                    # Ensure consistent format
                    test_copy = test.copy()
                    if 'Analysis_Type' not in test_copy:
                        test_copy['Analysis_Type'] = protocol
                    if 'Protocol' not in test_copy:
                        test_copy['Protocol'] = protocol
                    # Standardize test type field
                    if 'Test' in test_copy and 'Test_Type' not in test_copy:
                        test_copy['Test_Type'] = test_copy['Test']
                    all_tests.append(test_copy)
                proto_tests_count += len(protocol_tests)
        logging.info(f"Added {proto_tests_count} tests from protocol-specific comparison")
    
    # 3. Within-group comparison tests (these are different from the ones already included above)
    if within_group_results and 'within_group_results' in within_group_results:
        within_tests_count = 0
        for group, results in within_group_results['within_group_results'].items():
            # Involvement tests (paired comparisons)
            if 'involvement_test_results' in results:
                involvement_tests = results['involvement_test_results']
                for test in involvement_tests:
                    # These are already formatted correctly from the updated function
                    all_tests.append(test)
                within_tests_count += len(involvement_tests)
            
            # Origin tests (stage comparisons)
            if 'origin_test_results' in results:
                origin_tests = results['origin_test_results']
                for test in origin_tests:
                    # Add standardized fields for consistency
                    test_copy = test.copy()
                    test_copy['Analysis_Type'] = 'Within_Group_Detailed'
                    test_copy['Comparison_Type'] = 'Within_Group'
                    test_copy['Treatment_Group'] = group
                    test_copy['Protocol'] = 'All_Combined'
                    if 'Stage' not in test_copy:
                        test_copy['Stage'] = 'Multiple_Stages'
                    # Standardize test type field
                    if 'Test' in test_copy and 'Test_Type' not in test_copy:
                        test_copy['Test_Type'] = test_copy['Test']
                    all_tests.append(test_copy)
                within_tests_count += len(origin_tests)
        logging.info(f"Added {within_tests_count} tests from detailed within-group comparison")
    
    # Log summary
    total_tests = len(all_tests)
    logging.info(f"\nConsolidated {total_tests} total statistical tests")
    
    # Count by test type
    test_type_counts = {}
    comparison_type_counts = {}
    analysis_type_counts = {}
    
    for test in all_tests:
        test_type = test.get('Test', 'Unknown')
        comparison_type = test.get('Comparison_Type', 'Unknown')
        analysis_type = test.get('Analysis_Type', 'Unknown')
        
        test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        comparison_type_counts[comparison_type] = comparison_type_counts.get(comparison_type, 0) + 1
        analysis_type_counts[analysis_type] = analysis_type_counts.get(analysis_type, 0) + 1
    
    logging.info(f"\nTest type breakdown:")
    for test_type, count in test_type_counts.items():
        logging.info(f"  {test_type}: {count}")
    
    logging.info(f"\nComparison type breakdown:")
    for comp_type, count in comparison_type_counts.items():
        logging.info(f"  {comp_type}: {count}")
    
    logging.info(f"\nAnalysis type breakdown:")
    for anal_type, count in analysis_type_counts.items():
        logging.info(f"  {anal_type}: {count}")
    
    return all_tests


def calculate_involvement_percentage_changes(overall_results, proto_specific_results, results_by_treatment_group):
    """
    Calculate percentage changes in involvement for Stim-Pre and Post-Pre comparisons.
    Follows the approach used in plot_percentage_changes.py - uses aggregated involvement statistics.
    
    Args:
        overall_results: Results from overall treatment comparison
        proto_specific_results: Results from protocol-specific comparison  
        results_by_treatment_group: Raw data structure with subject-level results
    
    Returns:
        Dictionary containing percentage change data for overall and protocol-specific analyses
    """
    logging.info("\n=== Calculating Involvement Percentage Changes (Following plot_percentage_changes.py Logic) ===")
    
    # Use proto_specific_results to determine stages and comparison types
    proto_results = proto_specific_results.get('proto_specific_results', {})
    if not proto_results:
        logging.error("No protocol-specific results available for percentage change calculation")
        return {}
    
    # Determine stages from proto_specific_results
    all_stages = set()
    for protocol in proto_results:
        for group in proto_results[protocol].get('involvement_stats', {}):
            all_stages.update(proto_results[protocol]['involvement_stats'][group].keys())
    
    # Determine standard stage order and comparison types
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']
        comparison_types = ['Post-Pre']  # 4-stage scheme: compare post to pre
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  
        comparison_types = ['Stim-Pre', 'Post-Pre']  # 3-stage scheme: both comparisons
    else:
        stages = sorted(all_stages)
        # Default to Post-Pre if available, otherwise use the last stage vs pre
        if 'pre' in stages and 'post' in stages:
            comparison_types = ['Post-Pre']
        elif 'pre' in stages and 'stim' in stages:
            comparison_types = ['Stim-Pre']
        else:
            comparison_types = []
            logging.warning("No suitable stage combinations found for percentage change calculation")
    
    logging.info(f"Detected stages: {stages}")
    logging.info(f"Calculating percentage changes for: {comparison_types}")
    
    percentage_change_data = {
        'overall': {},
        'protocol_specific': {}
    }
    
    groups = list(proto_results[list(proto_results.keys())[0]].get('involvement_stats', {}).keys())
    
    # --- Overall Analysis (All Protocols Combined) ---
    logging.info("\n--- Overall Percentage Changes (All Protocols Combined) ---")
    
    overall_changes = {}
    overall_involvement_stats = overall_results.get('overall_involvement_stats', {})
    
    for group in groups:
        overall_changes[group] = {}
        
        if group in overall_involvement_stats:
            # Get aggregated stats for this group
            group_stats = overall_involvement_stats[group]
            
            # Get pre-stage stats
            if 'pre' in group_stats and group_stats['pre']['count'] > 0:
                pre_mean = group_stats['pre']['mean']
                pre_std = group_stats['pre']['std']
                pre_count = group_stats['pre']['count']
                
                for comp_type in comparison_types:
                    target_stage = comp_type.split('-')[0].lower()  # 'stim' or 'post'
                    
                    if target_stage in group_stats and group_stats[target_stage]['count'] > 0 and pre_mean > 0:
                        target_mean = group_stats[target_stage]['mean']
                        target_std = group_stats[target_stage]['std']
                        target_count = group_stats[target_stage]['count']
                        
                        # Calculate percentage change: ((target - pre) / pre) * 100
                        pct_change_mean = ((target_mean - pre_mean) / pre_mean) * 100
                        
                        # Error propagation for percentage change
                        # For f = ((B-A)/A)*100, error = 100 * sqrt((sB/A)^2 + ((B-A)*sA/A^2)^2)
                        pct_change_std = 100 * np.sqrt((target_std/pre_mean)**2 + ((target_mean - pre_mean)*pre_std/pre_mean**2)**2)
                        
                        overall_changes[group][comp_type] = {
                            'percentage_change_mean': pct_change_mean,
                            'percentage_change_std': pct_change_std,
                            'count': min(target_count, pre_count),
                            'pre_mean': pre_mean,
                            'target_mean': target_mean
                        }
                        
                        logging.info(f"Overall {group} {comp_type}: {pct_change_mean:.1f}% ± {pct_change_std:.1f}% (n={min(target_count, pre_count)})")
    
    percentage_change_data['overall'] = overall_changes
    
    # --- Protocol-Specific Analysis (Following plot_percentage_changes.py approach) ---
    logging.info("\n--- Protocol-Specific Percentage Changes ---")
    
    proto_changes = {}
    
    # Filter to only proto1-8 as requested
    protocols_to_include = [f'proto{i}' for i in range(1, 9)]
    available_protocols = [p for p in sorted(proto_results.keys()) if p in protocols_to_include]
    
    for protocol in available_protocols:
        logging.info(f"\n  Analyzing {protocol}...")
        proto_changes[protocol] = {}
        
        # Get involvement_stats for this protocol
        protocol_stats = proto_results[protocol].get('involvement_stats', {})
        
        for group in groups:
            proto_changes[protocol][group] = {}
            
            if group in protocol_stats:
                group_stats = protocol_stats[group]
                
                # Get pre-stage stats
                if 'pre' in group_stats and group_stats['pre']['count'] > 0:
                    pre_mean = group_stats['pre']['mean']
                    pre_std = group_stats['pre']['std']
                    pre_count = group_stats['pre']['count']
                    
                    for comp_type in comparison_types:
                        target_stage = comp_type.split('-')[0].lower()  # 'stim' or 'post'
                        
                        if target_stage in group_stats and group_stats[target_stage]['count'] > 0 and pre_mean > 0:
                            target_mean = group_stats[target_stage]['mean']
                            target_std = group_stats[target_stage]['std']
                            target_count = group_stats[target_stage]['count']
                            
                            # Calculate percentage change: ((target - pre) / pre) * 100
                            pct_change_mean = ((target_mean - pre_mean) / pre_mean) * 100
                            
                            # Error propagation for percentage change
                            pct_change_std = 100 * np.sqrt((target_std/pre_mean)**2 + ((target_mean - pre_mean)*pre_std/pre_mean**2)**2)
                            
                            proto_changes[protocol][group][comp_type] = {
                                'percentage_change_mean': pct_change_mean,
                                'percentage_change_std': pct_change_std,
                                'count': min(target_count, pre_count),
                                'pre_mean': pre_mean,
                                'target_mean': target_mean
                            }
                            
                            logging.info(f"  {protocol} {group} {comp_type}: {pct_change_mean:.1f}% ± {pct_change_std:.1f}% (n={min(target_count, pre_count)})")
    
    percentage_change_data['protocol_specific'] = proto_changes
    
    # Log summary
    logging.info(f"\nPercentage change calculation complete:")
    logging.info(f"  Overall analysis: {len(overall_changes)} groups")
    logging.info(f"  Protocol-specific: {len(available_protocols)} protocols (proto1-8)")
    logging.info(f"  Comparison types: {comparison_types}")
    
    return percentage_change_data
