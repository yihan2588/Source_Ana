import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from utils import (
    extract_region_name,
    calculate_origin_statistics,
    calculate_involvement_statistics,
    collect_data_by_treatment_group,
    read_subject_condition_mapping,
    scan_available_subjects_and_nights
)

from analysis import (
    validate_wave_result,
    analyze_slow_wave,
    process_directory,
    analyze_overall_treatment_comparison,
    analyze_proto_specific_comparison,
    analyze_within_group_stage_comparison,
    consolidate_statistical_results,
    calculate_involvement_percentage_changes
)

# Removed visualize import - visualization simplified/removed


def process_eeg_data_directory_subject_specific(directory_path, subject_condition_mapping, selected_subjects=None, selected_nights=None, visualize_regions=True, process_origins=True, source_dir=None):
    """
    Process the EEG data directory using subject-specific thresholds.
    Each subject's threshold is calculated from their own proto1-8 pre-stimulation data (95th percentile).
    
    Args:
        directory_path: Path to the EEG data directory
        subject_condition_mapping: Dictionary mapping subject IDs to conditions (Active/SHAM)
        selected_subjects: List of subject IDs to process (if None, process all)
        selected_nights: List of night IDs to process (if None, process all)
        visualize_regions: If True, generate region visualizations
        process_origins: If True, process origin analysis; if False, skip origin processing
        source_dir: Source directory where data is read from, used to construct output path
    """
    logging.info(f"Processing EEG data directory with SUBJECT-SPECIFIC thresholds: {directory_path}")

    # Define the expected treatment groups
    treatment_groups = ["Active", "SHAM"]
    # New structure: {group: {subject: {protocol: {stage: [wave_results]}}}}
    results_by_treatment_group = {group: {} for group in treatment_groups}
    total_processed_files = 0
    subject_thresholds = {}  # Store subject-specific thresholds
    
    # Get all subject directories
    subject_dirs = [d for d in Path(directory_path).iterdir() if d.is_dir() and d.name.startswith("Subject_")]
    
    # Filter subjects if specified
    if selected_subjects:
        subject_dirs = [d for d in subject_dirs if d.name in selected_subjects]

    if not subject_dirs:
        logging.warning(f"No subject directories found matching selection in {directory_path}.")
        return None

    logging.info(f"\n=== SUBJECT-SPECIFIC THRESHOLD ANALYSIS ===")
    logging.info(f"Processing {len(subject_dirs)} subjects with individualized thresholds...")
    logging.info(f"Each subject's threshold calculated from their own proto1-8 pre-stim data (95th percentile)")

    # Process each subject individually
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

        logging.info(f"\nProcessing {subject_id} (Group: {group})...")

        # Get night directories for this subject
        night_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("Night")]
        
        # Filter nights if specified
        if selected_nights:
            night_dirs = [d for d in night_dirs if d.name in selected_nights]

        if not night_dirs:
            logging.warning(f"No selected night directories found for {subject_id}.")
            continue

        # STEP 1: Calculate subject-specific threshold from their proto1-8 pre-stim data
        subject_prestim_values = []
        
        for night_dir in night_dirs:
            source_recon_dir = night_dir / "Output" / "SourceRecon"
            if not source_recon_dir.exists():
                continue
                
            # Collect this subject's proto1-8 pre-stim CSV files
            csv_files = list(source_recon_dir.glob('*proto*pre*.csv'))
            for csv_file in csv_files:
                filename = csv_file.name
                protocol_match = re.search(r'proto(\d+)', filename, re.IGNORECASE)
                if protocol_match:
                    proto_num = int(protocol_match.group(1))
                    if proto_num < 1 or proto_num > 8:
                        continue  # Skip protocols outside the 1-8 range
                else:
                    continue
                
                try:
                    df = pd.read_csv(csv_file)
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
                            subject_prestim_values.extend(data.flatten())
                except Exception as e:
                    logging.warning(f"Error reading pre-stim file {csv_file} for {subject_id}: {str(e)}")
        
        # Calculate subject-specific threshold
        if subject_prestim_values:
            subject_threshold = np.percentile(subject_prestim_values, 95)
            subject_thresholds[subject_id] = subject_threshold
            logging.info(f"  {subject_id} threshold: {subject_threshold:.6e} (95th percentile from {len(subject_prestim_values)} pre-stim data points)")
        else:
            logging.error(f"No proto1-8 pre-stim data found for {subject_id}! Skipping subject.")
            continue

        # STEP 2: Process all data for this subject using their specific threshold
        subject_processed_files = 0
        
        for night_dir in night_dirs:
            night_id = night_dir.name
            logging.info(f"  Processing {night_id} with subject-specific threshold...")

            source_recon_dir = night_dir / "Output" / "SourceRecon"
            if not source_recon_dir.exists():
                logging.warning(f"SourceRecon directory not found for {subject_id}/{night_id}. Skipping.")
                continue
                
            # Process the CSV files using subject-specific threshold
            subject_night_results = process_directory_subject_specific(
                source_recon_dir, 
                subject_id=subject_id,
                subject_threshold=subject_threshold,
                visualize_regions=visualize_regions, 
                process_origins=process_origins, 
                source_dir=source_dir
            )

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

                # Count processed files
                for protocol in subject_night_results:
                    for stage in subject_night_results[protocol]:
                        subject_processed_files += len(subject_night_results[protocol][stage])

        logging.info(f"  {subject_id} processed {subject_processed_files} waves with threshold {subject_threshold:.6e}")
        total_processed_files += subject_processed_files

    # Log summary of subject-specific thresholds
    logging.info(f"\n=== SUBJECT-SPECIFIC THRESHOLD SUMMARY ===")
    logging.info(f"Total subjects processed: {len(subject_thresholds)}")
    logging.info(f"Total waves processed: {total_processed_files}")
    
    if subject_thresholds:
        threshold_values = list(subject_thresholds.values())
        logging.info(f"Threshold range: {min(threshold_values):.6e} to {max(threshold_values):.6e}")
        logging.info(f"Mean threshold: {np.mean(threshold_values):.6e} ± {np.std(threshold_values):.6e}")
        
        # Log by treatment group
        for group in treatment_groups:
            group_thresholds = [subject_thresholds[subj] for subj in results_by_treatment_group[group] if subj in subject_thresholds]
            if group_thresholds:
                logging.info(f"{group} group thresholds (n={len(group_thresholds)}): {np.mean(group_thresholds):.6e} ± {np.std(group_thresholds):.6e}")

    if not any(results_by_treatment_group.values()):
        logging.error("No data was processed successfully.")
        return None

    return results_by_treatment_group, subject_thresholds


def process_directory_subject_specific(directory_path, subject_id, subject_threshold, visualize_regions=True, process_origins=True, source_dir=None):
    """
    Process all CSV files in the directory using a subject-specific threshold.
    
    Args:
        directory_path: Path to the directory containing CSV files
        subject_id: Subject identifier for logging
        subject_threshold: Subject-specific threshold to use
        visualize_regions: If True, generate region visualizations
        process_origins: If True, process origin analysis; if False, skip origin processing
        source_dir: Source directory where data is read from, used to construct output path
    """
    logging.info(f"Processing directory: {directory_path} (Subject: {subject_id}, Threshold: {subject_threshold:.6e})")

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
            protocol = protocol_match.group(1).lower()
            stage = stage_match.group(1).lower()
            
            # Filter to only process proto1-8
            protocol_number = protocol.replace('proto', '')
            try:
                proto_num = int(protocol_number)
                if proto_num < 1 or proto_num > 8:
                    continue
            except ValueError:
                continue
                
            if protocol not in results_by_protocol:
                results_by_protocol[protocol] = {}
            
            if stage not in results_by_protocol[protocol]:
                results_by_protocol[protocol][stage] = []

            try:
                df = pd.read_csv(csv_file)
                wave_name = csv_file.stem
                
                # Call analyze_slow_wave with subject-specific threshold
                result = analyze_slow_wave(
                    df, 
                    wave_name, 
                    process_origins=process_origins, 
                    fixed_threshold=subject_threshold  # Use subject-specific threshold
                )
                
                # Add subject_id to the result for tracking
                result['subject_id'] = subject_id
                result['subject_threshold'] = subject_threshold
                
                # Validate and log
                validate_wave_result_subject_specific(result, csv_file, subject_id)

                results_by_protocol[protocol][stage].append(result)
                processed_files += 1

                # Skip visualization - removed to simplify pipeline
                # if visualize_regions:
                #     visualize_region_time_series(result, csv_file, source_dir=source_dir)

            except Exception as e:
                error_files += 1
                logging.error(f"Error processing {filename} for {subject_id}: {str(e)}")

    logging.info(f"Subject {subject_id} directory processing summary:")
    logging.info(f"  Total files found: {total_files}")
    logging.info(f"  Successfully processed: {processed_files}")
    logging.info(f"  Errors: {error_files}")

    return results_by_protocol


def validate_wave_result_subject_specific(result, csv_file_path=None, subject_id=None):
    """
    Print and save validation information for a single wave file output with subject-specific details (simplified version).
    
    Args:
        result: Dictionary containing wave analysis results from analyze_slow_wave()
        csv_file_path: Path to the original CSV file (to save log file next to it)
        subject_id: Subject identifier
    """
    wave_name = result['wave_name']
    involvement_percentage = result['involvement_percentage']
    involvement_count = result['involvement_count']
    total_units = result.get('total_units', 0)
    subject_threshold = result.get('subject_threshold', 'Unknown')

    # Prepare validation output
    validation_lines = []
    validation_lines.append(f"[VALIDATION] Subject: {subject_id}")
    validation_lines.append(f"[VALIDATION] Wave: {wave_name}")
    validation_lines.append(f"[VALIDATION] Subject-specific threshold: {subject_threshold:.10e}")
    validation_lines.append(f"[VALIDATION] Involvement: {involvement_percentage:.2f}% ({involvement_count}/{total_units} units)")

    # Add window information
    window = result.get('window', (0, 0))
    validation_lines.append(f"[VALIDATION] Window: {window[0]}ms to {window[1]}ms")
    validation_lines.append("[VALIDATION] ------------------------------")

    # Log to main pipeline log
    for line in validation_lines:
        logging.info(line)

    # Save to individual log file if CSV path is provided
    if csv_file_path:
        try:
            log_file_path = str(csv_file_path).replace('.csv', f'_{subject_id}_subj_specific.log')
            with open(log_file_path, 'w') as log_file:
                for line in validation_lines:
                    log_file.write(f"{line}\n")
            logging.info(f"[VALIDATION] Saved subject-specific validation log to: {log_file_path}")
        except Exception as e:
            logging.error(f"[VALIDATION] Error saving validation log: {str(e)}")


def compare_threshold_approaches(directory_path, subject_condition_mapping, selected_subjects=None, selected_nights=None):
    """
    Compare global vs subject-specific threshold approaches.
    
    Args:
        directory_path: Path to the EEG data directory
        subject_condition_mapping: Dictionary mapping subject IDs to conditions
        selected_subjects: List of subject IDs to process
        selected_nights: List of night IDs to process
    
    Returns:
        Dictionary containing comparison results
    """
    logging.info("\n=== THRESHOLD APPROACH COMPARISON ===")
    
    # Import the original function
    from analysis import process_eeg_data_directory
    
    # Process with global threshold (original approach)
    logging.info("\n--- GLOBAL THRESHOLD APPROACH ---")
    global_results = process_eeg_data_directory(
        directory_path, 
        subject_condition_mapping, 
        selected_subjects, 
        selected_nights, 
        visualize_regions=False,  # Skip visualization for comparison
        process_origins=True
    )
    
    # Process with subject-specific thresholds
    logging.info("\n--- SUBJECT-SPECIFIC THRESHOLD APPROACH ---")
    subject_results, subject_thresholds = process_eeg_data_directory_subject_specific(
        directory_path, 
        subject_condition_mapping, 
        selected_subjects, 
        selected_nights, 
        visualize_regions=False,  # Skip visualization for comparison
        process_origins=True
    )
    
    # Compare results
    logging.info("\n=== COMPARISON ANALYSIS ===")
    comparison_results = analyze_threshold_differences(global_results, subject_results, subject_thresholds, subject_condition_mapping)
    
    return {
        'global_results': global_results,
        'subject_specific_results': subject_results,
        'subject_thresholds': subject_thresholds,
        'comparison_analysis': comparison_results
    }


def analyze_threshold_differences(global_results, subject_results, subject_thresholds, subject_condition_mapping):
    """
    Analyze differences between global and subject-specific threshold approaches.
    
    Args:
        global_results: Results from global threshold approach
        subject_results: Results from subject-specific threshold approach
        subject_thresholds: Dictionary of subject-specific thresholds
        subject_condition_mapping: Subject to treatment group mapping
    
    Returns:
        Dictionary containing comparison analysis
    """
    logging.info("Analyzing differences between threshold approaches...")
    
    comparison_data = {}
    
    # Compare involvement percentages
    for group in ['Active', 'SHAM']:
        if group in global_results and group in subject_results:
            comparison_data[group] = {}
            
            # Get subjects in both approaches
            global_subjects = set(global_results[group].keys())
            subject_subjects = set(subject_results[group].keys())
            common_subjects = global_subjects & subject_subjects
            
            logging.info(f"\n{group} group - comparing {len(common_subjects)} subjects:")
            
            subject_comparisons = {}
            
            for subject in common_subjects:
                subject_comparisons[subject] = {}
                subject_threshold = subject_thresholds.get(subject, 'Unknown')
                
                logging.info(f"  {subject} (threshold: {subject_threshold:.6e}):")
                
                # Compare across protocols and stages
                for protocol in global_results[group][subject]:
                    if protocol in subject_results[group][subject]:
                        subject_comparisons[subject][protocol] = {}
                        
                        for stage in global_results[group][subject][protocol]:
                            if stage in subject_results[group][subject][protocol]:
                                # Get involvement percentages
                                global_involvements = [w['involvement_percentage'] for w in global_results[group][subject][protocol][stage]]
                                subject_involvements = [w['involvement_percentage'] for w in subject_results[group][subject][protocol][stage]]
                                
                                if global_involvements and subject_involvements:
                                    global_mean = np.mean(global_involvements)
                                    subject_mean = np.mean(subject_involvements)
                                    difference = subject_mean - global_mean
                                    percent_change = (difference / global_mean * 100) if global_mean > 0 else 0
                                    
                                    subject_comparisons[subject][protocol][stage] = {
                                        'global_mean': global_mean,
                                        'subject_specific_mean': subject_mean,
                                        'difference': difference,
                                        'percent_change': percent_change,
                                        'n_waves': len(global_involvements)
                                    }
                                    
                                    logging.info(f"    {protocol}-{stage}: Global={global_mean:.1f}%, Subject-specific={subject_mean:.1f}%, "
                                          f"Δ={difference:+.1f}% ({percent_change:+.1f}%) [n={len(global_involvements)}]")
            
            comparison_data[group]['subject_comparisons'] = subject_comparisons
    
    # Calculate group-level statistics
    logging.info("\n=== GROUP-LEVEL COMPARISON ===")
    
    group_stats = {}
    for group in ['Active', 'SHAM']:
        if group in comparison_data:
            all_differences = []
            all_percent_changes = []
            
            for subject in comparison_data[group]['subject_comparisons']:
                for protocol in comparison_data[group]['subject_comparisons'][subject]:
                    for stage in comparison_data[group]['subject_comparisons'][subject][protocol]:
                        stage_data = comparison_data[group]['subject_comparisons'][subject][protocol][stage]
                        all_differences.append(stage_data['difference'])
                        all_percent_changes.append(stage_data['percent_change'])
            
            if all_differences:
                group_stats[group] = {
                    'mean_difference': np.mean(all_differences),
                    'std_difference': np.std(all_differences),
                    'mean_percent_change': np.mean(all_percent_changes),
                    'std_percent_change': np.std(all_percent_changes),
                    'n_comparisons': len(all_differences)
                }
                
                logging.info(f"{group} group summary:")
                logging.info(f"  Mean difference: {group_stats[group]['mean_difference']:.2f}% ± {group_stats[group]['std_difference']:.2f}%")
                logging.info(f"  Mean percent change: {group_stats[group]['mean_percent_change']:.1f}% ± {group_stats[group]['std_percent_change']:.1f}%")
                logging.info(f"  Number of comparisons: {group_stats[group]['n_comparisons']}")
    
    # Statistical test between approaches
    if 'Active' in group_stats and 'SHAM' in group_stats:
        active_differences = []
        sham_differences = []
        
        for subject in comparison_data['Active']['subject_comparisons']:
            for protocol in comparison_data['Active']['subject_comparisons'][subject]:
                for stage in comparison_data['Active']['subject_comparisons'][subject][protocol]:
                    active_differences.append(comparison_data['Active']['subject_comparisons'][subject][protocol][stage]['difference'])
        
        for subject in comparison_data['SHAM']['subject_comparisons']:
            for protocol in comparison_data['SHAM']['subject_comparisons'][subject]:
                for stage in comparison_data['SHAM']['subject_comparisons'][subject][protocol]:
                    sham_differences.append(comparison_data['SHAM']['subject_comparisons'][subject][protocol][stage]['difference'])
        
        if active_differences and sham_differences:
            try:
                u_stat, p_val = scipy_stats.mannwhitneyu(active_differences, sham_differences, alternative='two-sided')
                logging.info(f"\nMann-Whitney U test comparing threshold approach differences between groups:")
                logging.info(f"  U={u_stat:.2f}, p={p_val:.4f}")
                if p_val < 0.05:
                    logging.info("  Significant difference in how threshold approaches affect the two groups!")
            except Exception as e:
                logging.error(f"Error performing statistical test: {str(e)}")
    
    return {
        'group_comparisons': comparison_data,
        'group_statistics': group_stats
    }
