import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from scipy import stats as scipy_stats

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
    validation_lines.append(f"[VALIDATION] Window: {window[0]}ms to {window[1]}ms, Threshold: {threshold:.6f}")
    validation_lines.append("[VALIDATION] ------------------------------")
    
    # Print to console
    for line in validation_lines:
        print(line)
    
    # Save to log file if CSV path is provided
    if csv_file_path:
        try:
            # Create log file next to the original CSV
            log_file_path = str(csv_file_path).replace('.csv', '.log')
            with open(log_file_path, 'w') as log_file:
                for line in validation_lines:
                    log_file.write(f"{line}\n")
            print(f"[VALIDATION] Saved validation log to: {log_file_path}")
        except Exception as e:
            print(f"[VALIDATION] Error saving validation log: {str(e)}")
# Updated imports for stats_utils
from stats_utils import (
    perform_friedman_test,
    perform_wilcoxon_posthoc,
    perform_origin_distribution_tests,
    perform_chi_square_or_fisher_test
)
# Keep scipy stats import for Mann-Whitney U (group comparisons)
from scipy import stats as scipy_stats
from visualize import visualize_region_time_series, plot_voxel_waveforms


def analyze_slow_wave(df, wave_name, window_ms=100, threshold_percent=25, debug=True):
    """
    Analyze a single slow wave CSV file to extract origin and involvement metrics.
    """
    if debug:
        print(f"\nAnalyzing {wave_name}...")
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
        print(f"Warning: No time points found in window [{window_start}, {window_end}] ms")
        window_mask = np.ones_like(time_points, dtype=bool)

    window_data = data[:, window_mask]
    window_times = time_points[window_mask]

    if window_data.size > 0:
        max_current = np.max(window_data)
        threshold = max_current * (threshold_percent / 100)
    else:
        threshold = 0

    voxel_peak_times = []
    involved_voxels = []

    for voxel_idx in range(len(data)):
        if voxel_idx < window_data.shape[0]:
            voxel_data = window_data[voxel_idx]
            peaks = []
            for i in range(1, len(voxel_data)-1):
                if voxel_data[i] > voxel_data[i-1] and voxel_data[i] > voxel_data[i+1]:
                    if voxel_data[i] > threshold:
                        peaks.append((window_times[i], voxel_data[i]))

            if peaks:
                first_peak_time = min(peaks, key=lambda x: x[0])[0]
                region = extract_region_name(voxel_names[voxel_idx])
                voxel_peak_times.append({
                    'full_name': voxel_names[voxel_idx],
                    'region': region,
                    'peak_time': first_peak_time
                })
                involved_voxels.append(voxel_names[voxel_idx])

    total_voxels = len(data)
    num_involved = len(involved_voxels)
    involvement_percentage = (num_involved / total_voxels) * 100 if total_voxels > 0 else 0

    # Identify "origins" as the earliest 10% of involved voxels
    origins = []
    if voxel_peak_times:
        peak_times_df = pd.DataFrame(voxel_peak_times).sort_values('peak_time')
        n_origins = max(1, int(len(peak_times_df) * 0.1))
        origins = peak_times_df.head(n_origins)

    if debug:
        print(f"Involvement: {involvement_percentage:.1f}% ({num_involved}/{total_voxels} voxels)")
        if voxel_peak_times:
            print("Top origin regions:")
            for _, row in origins.iterrows():
                print(f"- {row['region']} at {row['peak_time']:.1f}ms")

    return {
        'wave_name': wave_name,
        'origins': origins,  # a small DataFrame of earliest 10% voxel-peak times
        'involvement_count': num_involved,
        'involvement_percentage': involvement_percentage,
        'involved_voxels': involved_voxels,
        'window': (window_start, window_end),
        'threshold': threshold
    }


def process_eeg_data_directory(directory_path, subject_condition_mapping, selected_subjects=None, selected_nights=None, visualize_regions=True, source_dir=None):
    """
    Process the EEG data directory using subject-condition mapping and optional subject/night filtering.
    
    Args:
        directory_path: Path to the EEG data directory
        subject_condition_mapping: Dictionary mapping subject IDs to conditions (Active/SHAM)
        selected_subjects: List of subject IDs to process (if None, process all)
        selected_nights: List of night IDs to process (if None, process all)
        visualize_regions: If True, generate region visualizations
        source_dir: Source directory where data is read from, used to construct output path
    """
    print(f"Processing EEG data directory: {directory_path}")
    
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
        print(f"Warning: No subject directories found.")
        return None
    
    print(f"\nProcessing {len(subject_dirs)} subjects...")
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        # Skip if subject not in mapping
        if subject_id not in subject_condition_mapping:
            print(f"Warning: Subject {subject_id} not found in condition mapping. Skipping.")
            continue
        # Get treatment group for this subject
        group = subject_condition_mapping[subject_id]
        if group not in treatment_groups:
            print(f"Warning: Unknown treatment group '{group}' for {subject_id}. Skipping.")
            continue

        # Initialize subject entry if not present
        if subject_id not in results_by_treatment_group[group]:
            results_by_treatment_group[group][subject_id] = {}

        print(f"Processing {subject_id} (Group: {group})...")

        # Get night directories
        night_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith("Night")]
        
        # Filter nights if specified
        if selected_nights:
            night_dirs = [d for d in night_dirs if d.name in selected_nights]
        
        if not night_dirs:
            print(f"Warning: No night directories found for {subject_id}.")
            continue
        
        for night_dir in night_dirs:
            night_id = night_dir.name
            print(f"  Processing {night_id}...")
            
            # Navigate to the SourceRecon directory
            source_recon_dir = night_dir / "Output" / "SourceRecon"
            if not source_recon_dir.exists():
                print(f"Warning: SourceRecon directory not found for {subject_id}/{night_id}.")
                continue
            # Process the CSV files in the SourceRecon directory for this specific night
            # subject_night_results has format {protocol: {stage: [wave_results]}}
            subject_night_results = process_directory(source_recon_dir, quiet=True, visualize_regions=visualize_regions, source_dir=source_dir)

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

    # Recalculate total_files based on collected data for accuracy
    total_files = processed_files # Approximation for now

    print(f"\nProcessing summary:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Errors: {error_files}")
    
    if not any(results_by_treatment_group.values()):
        print("No data was processed successfully.")
        return None
    
    return results_by_treatment_group


def process_directory(directory_path, quiet=False, visualize_regions=True, source_dir=None):
    """
    Process all CSV files in the directory and organize by protocol and stage.
    
    Args:
        directory_path: Path to the directory containing CSV files
        quiet: If True, suppress most output
        visualize_regions: If True, generate region visualizations
        source_dir: Source directory where data is read from, used to construct output path
    """
    if not quiet:
        print(f"Processing directory: {directory_path}")

    csv_files = list(Path(directory_path).glob('*.csv'))
    if not csv_files:
        if not quiet:
            print("No CSV files found in the specified directory.")
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
            if protocol not in results_by_protocol:
                results_by_protocol[protocol] = {}
            
            # Make sure the stage exists in the protocol's results
            if stage not in results_by_protocol[protocol]:
                results_by_protocol[protocol][stage] = []

            try:
                df = pd.read_csv(csv_file)
                wave_name = csv_file.stem
                result = analyze_slow_wave(df, wave_name, debug=False)
                
                # Validate, print, and save wave result information
                validate_wave_result(result, csv_file)
                
                results_by_protocol[protocol][stage].append(result)
                processed_files += 1
                
                # Generate region time series visualization
                if visualize_regions:
                    # Use source_dir if provided, otherwise use the standard 'results' directory
                    visualize_region_time_series(result, csv_file, output_dir="results", source_dir=source_dir)
                    # Add the new call for voxel waveforms
                    plot_voxel_waveforms(csv_file, wave_name, output_dir="results", source_dir=source_dir)
                
                if not quiet:
                    print(f"Processed {filename} - Protocol: {protocol}, Stage: {stage}")
            except Exception as e:
                error_files += 1
                if not quiet:
                    print(f"Error processing {filename}: {str(e)}")

    if not quiet:
        print(f"\nProcessing summary:")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Errors: {error_files}")

    return results_by_protocol


def analyze_protocol_results(protocol_name, protocol_results, master_region_list):
    """
    Analyze results for a single protocol across all stages.
    """
    print(f"\n=== Analysis for {protocol_name} ===")
    # Note: This function operates on data aggregated across subjects for a single protocol.
    # It cannot perform paired tests as subject info is lost at this stage.
    # We will only perform origin analysis here. Paired involvement tests are done elsewhere.

    stages = list(protocol_results.keys()) # Use list for potential ordering later if needed

    # --- Involvement Analysis Removed ---
    # The previous involvement analysis using Kruskal-Wallis/Mann-Whitney U on aggregated data
    # is removed because the user requested paired tests (Friedman/Wilcoxon), which require
    # subject-level data not available in the `protocol_results` structure passed here.
    # Paired involvement tests across stages are now handled in `analyze_within_group_stage_comparison`.

    print("\nInvolvement Statistics (Aggregated - Descriptive Only):")
    # Calculate and print descriptive stats, but no inferential tests here.
    involvement_stats = {}
    for stage in stages:
        involvement_list = [res['involvement_percentage'] for res in protocol_results.get(stage, [])]
        involvement_stats[stage] = calculate_involvement_statistics(involvement_list)
        stats_data = involvement_stats[stage]
        print(f"{stage.capitalize()}: {stats_data['mean']:.1f}% ± {stats_data['std']:.1f}% (n={stats_data['count']})")

    # Calculate origin statistics
    origin_data = {}
    for stage in stages:
        stage_results = protocol_results[stage]
        total_waves = len(stage_results)
        region_counts = calculate_origin_statistics(stage_results, stage, total_waves)
        origin_data[stage] = {
            'region_counts': region_counts,
            'total_waves': total_waves
        }

    print("\nOrigin Statistics:")
    for stage, data_dict in origin_data.items():
        rc = data_dict['region_counts']
        tw = data_dict['total_waves']
        if rc and tw > 0:
            print(f"{stage.capitalize()} (n={tw} waves):")
            # Show top 5 in the order of master_region_list
            ordered_regions = [r for r in master_region_list if r in rc]
            displayed = 0
            for region in ordered_regions:
                if displayed >= 5:
                    break
                count = rc[region]
                percentage = (count / tw) * 100
                print(f"  {region}: {count}/{tw} waves ({percentage:.1f}%)")
                displayed += 1
        else:
            print(f"{stage.capitalize()}: No origins detected")

    # Perform statistical tests on origin distribution
    origin_test_results = perform_origin_distribution_tests(origin_data, master_region_list)

    for result in origin_test_results:
        if 'DF' in result:
            print(f"\n{result['Test']} for Origin Distribution: χ²={result['Statistic']:.2f}, df={result['DF']}, p={result['P_Value']:.4f}")
        else:
            print(f"\n{result['Test']} for Origin Distribution: p={result['P_Value']:.4f}")
        if result['Significant']:
            print("Significant differences detected in origin distribution between stages.")

    if not origin_test_results:
        print("\nNot enough data for statistical test of origin distribution.")

    return {
        # 'involvement_data' and 'involvement_test_results' removed as they are no longer calculated here.
        'origin_data': origin_data,
        'involvement_stats': involvement_stats, # Keep descriptive stats
        'origin_test_results': origin_test_results
    }



def analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list, min_occurrence_threshold=3):
    """
    Perform overall treatment group comparison by collapsing subjects, nights, and protos.
    """
    print("\n=== Overall Treatment Group Comparison (Collapsing Subjects, Nights, and Protos) ===")

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
    print(f"DEBUG: Starting aggregation for all_waves_by_group. Groups: {groups}") # DEBUG
    for group in groups:
        print(f"DEBUG: Processing group: {group}") # DEBUG
        all_waves_by_group[group] = {stage: [] for stage in stages} # Initialize stages for the group
        subjects_in_group = list(results_by_treatment_group[group].keys())
        print(f"DEBUG: Subjects in group '{group}': {subjects_in_group}") # DEBUG
        for subject in subjects_in_group:
            print(f"DEBUG:  Processing subject: {subject}") # DEBUG
            protocols_for_subject = list(results_by_treatment_group[group][subject].keys())
            print(f"DEBUG:   Protocols for subject '{subject}': {protocols_for_subject}") # DEBUG
            for protocol in protocols_for_subject:
                print(f"DEBUG:    Processing protocol: {protocol}") # DEBUG
                stages_for_protocol = list(results_by_treatment_group[group][subject][protocol].keys())
                print(f"DEBUG:     Stages for protocol '{protocol}': {stages_for_protocol}") # DEBUG
                for stage in stages: # Iterate through ALL possible stages
                    # Check if the stage exists for this subject/protocol
                    if stage in stages_for_protocol: # Check against actual stages for this proto/subj
                        wave_list = results_by_treatment_group[group][subject][protocol][stage]
                        print(f"DEBUG:      Extending stage '{stage}' for group '{group}' with {len(wave_list)} items. Source: {group}/{subject}/{protocol}/{stage}") # DEBUG
                        # Check type before extending
                        if not isinstance(wave_list, list):
                             print(f"ERROR: Expected list but got {type(wave_list)} for {group}/{subject}/{protocol}/{stage}")
                             continue
                        if wave_list and not all(isinstance(item, dict) for item in wave_list):
                             print(f"ERROR: Expected list of dicts but found other types in {group}/{subject}/{protocol}/{stage}")
                             # Print the non-dict items
                             for idx, item in enumerate(wave_list):
                                 if not isinstance(item, dict):
                                     print(f"ERROR item index {idx}: type={type(item)}, value='{item}'") # Added quotes for clarity
                             continue
                        # If checks pass, extend
                        all_waves_by_group[group][stage].extend(wave_list)
                    # else: # DEBUG - uncomment if needed
                    #    print(f"DEBUG:      Stage '{stage}' not found for {group}/{subject}/{protocol}") # DEBUG


    # Calculate involvement (using the correctly aggregated all_waves_by_group)
    involvement_stats = {}
    involvement_data = {}
    groups = list(all_waves_by_group.keys())
    # stages already defined above with detected stages

    for group in groups:
        # Simplified Debugging loop V3
        involvement_by_stage = {}
        print(f"\nDEBUG: Calculating involvement for group '{group}'...") # Keep this outer debug print
        for stage in stages:
            stage_list = []
            items_in_stage = all_waves_by_group[group].get(stage, [])
            print(f"DEBUG:  Processing stage '{stage}', length: {len(items_in_stage)}") # Keep this inner debug print
            item_counter = 0
            for res_item in items_in_stage: # Use distinct variable name
                item_counter += 1
                try:
                    # Check type right before access attempt
                    if isinstance(res_item, dict):
                        value = res_item['involvement_percentage'] # Attempt access
                        stage_list.append(value)
                    else:
                        # Log unexpected type if not dict
                        print(f"DEBUG:   Item {item_counter} is NOT a dict. Type: {type(res_item)}, Value: '{res_item}'")
                except TypeError as te:
                    # Catch the specific error during access
                    print(f"DEBUG:   *** TypeError on item {item_counter}. Type was {type(res_item)}. Error: {te}. Item: '{res_item}'")
                except KeyError as ke:
                    # Catch missing key if it's a dict but key is missing
                    print(f"DEBUG:   *** KeyError on item {item_counter}. Key: {ke}. Item: '{res_item}'")
                except Exception as e:
                    # Catch any other error during access
                    print(f"DEBUG:   *** Unexpected Error on item {item_counter}. Error: {e}. Item: '{res_item}'")
            involvement_by_stage[stage] = stage_list

        # Safety check (optional, can be removed if debug loop is sufficient)
        for stage in stages:
            original_count = len(all_waves_by_group[group].get(stage, []))
            processed_count = len(involvement_by_stage.get(stage, []))
            if original_count != processed_count:
                 print(f"Warning in analyze_overall_treatment_comparison ({group}, {stage}): Skipped {original_count - processed_count} non-dictionary items during involvement calculation.")

        stats_by_stage = {
            stage: calculate_involvement_statistics(involvement_by_stage[stage])
            for stage in stages
        }
        involvement_stats[group] = stats_by_stage
        involvement_data[group] = involvement_by_stage

    print("\nOverall Involvement Statistics by Treatment Group (All Protos Combined):")
    for group in groups:
        print(f"\n{group} Group:")
        for stage in stages:
            sdata = involvement_stats[group][stage]
            print(f"  {stage.capitalize()}: {sdata['mean']:.1f}% ± {sdata['std']:.1f}% (n={sdata['count']})")

    # Perform statistical tests comparing involvement
    involvement_comparison_tests = []
    if 'Active' in involvement_data and 'SHAM' in involvement_data:
        for stage in stages:
            active_data = involvement_data['Active'][stage]
            sham_data = involvement_data['SHAM'][stage]
            if active_data and sham_data:
                try:
                    u_stat, p_val = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                    involvement_comparison_tests.append({
                        'Test': 'Mann-Whitney U',
                        'Metric': f'Overall Involvement: Active vs SHAM ({stage})',
                        'Stage': stage,
                        'Statistic': u_stat,
                        'P_Value': p_val,
                        'Significant': p_val < 0.05
                    })
                    print(f"\nMann-Whitney U Test for overall {stage} involvement (Active vs SHAM): "
                          f"U={u_stat:.2f}, p={p_val:.4f}")
                    if p_val < 0.05:
                        print(f"Significant difference detected in overall {stage} involvement between Active and SHAM groups.")
                except Exception as e:
                    print(f"Error performing statistical test for overall {stage} involvement: {str(e)}")

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

    print("\nOverall Origin Distribution by Treatment Group (All Protos Combined):")
    for group in groups:
        print(f"\n{group} Group:")
        for stage in stages:
            rc = origin_data[group][stage]['region_counts']
            tw = origin_data[group][stage]['total_waves']
            if rc and tw > 0:
                print(f"  {stage.capitalize()} (n={tw} waves):")
                # Filter or show top 10
                # But user wants to see partial. We'll keep the top 10 approach:
                # We'll also combine "Other" if below threshold, etc.
                # For simplicity, just show top 10 by count
                region_counts_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
                displayed = 0
                for (region, count) in region_counts_sorted:
                    if displayed >= 10:
                        break
                    percentage = (count / tw) * 100
                    print(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                    displayed += 1
            else:
                print(f"  {stage.capitalize()}: No origins detected")

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
                            print(f"\n{test_result['Test']} for overall {stage} origin distribution (Active vs SHAM): "
                                  f"χ²={test_result['Statistic']:.2f}, df={test_result['DF']}, p={test_result['P_Value']:.4f}")
                        else:
                            print(f"\n{test_result['Test']} for overall {stage} origin distribution (Active vs SHAM): "
                                  f"p={test_result['P_Value']:.4f}")
                        if test_result['Significant']:
                            print(f"Significant difference detected in overall {stage} origin distribution "
                                  f"between Active and SHAM groups.")

    return {
        'overall_involvement_data': involvement_data,
        'overall_involvement_stats': involvement_stats,
        'overall_origin_data': origin_data,
        'overall_comparison_tests': involvement_comparison_tests + origin_comparison_tests,
        'master_region_list': master_region_list
    }


def analyze_proto_specific_comparison(results_by_treatment_group, master_region_list, min_occurrence_threshold=2):
    """
    Perform proto-specific comparison between treatment groups for each proto.
    """
    print("\n=== Proto-Specific Comparison ===")
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

    # Ensure the main loop iterates through PROTOCOLS, not subjects or other keys
    for protocol in sorted(list(all_protocols)): # Explicitly sort the list of protocols
        print(f"\n--- Analysis for {protocol} ---")

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
            print(f"Skipping {protocol} - not enough treatment groups with data for this protocol.")
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

        # Involvement (using the correctly aggregated waves_by_group)
        involvement_stats = {}
        involvement_data = {}
        for group in waves_by_group: # Iterate through the corrected waves_by_group
            involvement_by_stage = {
                # Add safety check: only process if res is a dictionary
                stage: [res['involvement_percentage'] for res in waves_by_group[group].get(stage, []) if isinstance(res, dict)]
                for stage in stages
            }
            # Check if any non-dict items were skipped and print a warning
            for stage in stages:
                original_count = len(waves_by_group[group].get(stage, []))
                processed_count = len(involvement_by_stage.get(stage, []))
                if original_count != processed_count:
                    print(f"Warning in analyze_proto_specific_comparison ({protocol}, {group}, {stage}): Skipped {original_count - processed_count} non-dictionary items during involvement calculation.")

            stats_by_stage = {
                stage: calculate_involvement_statistics(involvement_by_stage[stage])
                for stage in stages
            }
            involvement_stats[group] = stats_by_stage
            involvement_data[group] = involvement_by_stage

        print(f"\nInvolvement Statistics for {protocol} by Treatment Group:")
        for group in waves_by_group:
            print(f"\n{group} Group:")
            for stage, stats in involvement_stats[group].items():
                print(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']})")

        # Statistical tests between groups
        involvement_comparison_tests = []
        if 'Active' in involvement_data and 'SHAM' in involvement_data:
            for stage in stages:
                active_data = involvement_data['Active'].get(stage, [])
                sham_data = involvement_data['SHAM'].get(stage, [])
                if active_data and sham_data:
                    try:
                        u_stat, p_val = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                        involvement_comparison_tests.append({
                            'Test': 'Mann-Whitney U',
                            'Protocol': protocol,
                            'Metric': f'Involvement: Active vs SHAM ({stage})',
                            'Stage': stage,
                            'Statistic': u_stat,
                            'P_Value': p_val,
                            'Significant': p_val < 0.05
                        })
                        print(f"\nMann-Whitney U Test for {stage} involvement in {protocol} (Active vs SHAM): "
                              f"U={u_stat:.2f}, p={p_val:.4f}")
                        if p_val < 0.05:
                            print(f"Significant difference detected in {stage} involvement between Active and SHAM groups for {protocol}.")
                    except Exception as e:
                        print(f"Error performing statistical test for {stage} involvement in {protocol}: {str(e)}")

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

        print(f"\nOrigin Distribution for {protocol} by Treatment Group:")
        for group in waves_by_group:
            print(f"\n{group} Group:")
            for stage, data_dict in origin_data[group].items():
                rc = data_dict['region_counts']
                tw = data_dict['total_waves']
                if rc and tw > 0:
                    print(f"  {stage.capitalize()} (n={tw} waves):")
                    # Filter or show top 5
                    region_counts_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
                    displayed = 0
                    for (region, count) in region_counts_sorted:
                        if displayed >= 5:
                            break
                        percentage = (count / tw) * 100
                        print(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                        displayed += 1
                else:
                    print(f"  {stage.capitalize()}: No origins detected")

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
                                print(f"\n{test_result['Test']} for {stage} origin distribution in {protocol} "
                                      f"(Active vs SHAM): χ²={test_result['Statistic']:.2f}, df={test_result['DF']}, "
                                      f"p={test_result['P_Value']:.4f}")
                            else:
                                print(f"\n{test_result['Test']} for {stage} origin distribution in {protocol} "
                                      f"(Active vs SHAM): p={test_result['P_Value']:.4f}")
                            if test_result['Significant']:
                                print(f"Significant difference detected in {stage} origin distribution between Active and SHAM groups for {protocol}.")

        proto_specific_results[protocol] = {
            'involvement_data': involvement_data,
            'involvement_stats': involvement_stats,
            'origin_data': origin_data,
            # Corrected: Combine involvement and origin tests, not involvement twice
            'comparison_tests': involvement_comparison_tests + origin_comparison_tests
        }

    return {
        'proto_specific_results': proto_specific_results,
        'master_region_list': master_region_list
    }


def analyze_within_group_stage_comparison(results_by_treatment_group, master_region_list):
    """
    Perform within-group stage comparison for each treatment group.
    """
    print("\n=== Within-Group Stage Comparison ===")

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
        print(f"\n--- Analysis for {group} Group ---")

        group_data = results_by_treatment_group[group]
        subjects = list(group_data.keys())
        if not subjects:
            print(f"No subjects found for {group} group.")
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
             print(f"Not enough stages with data for paired comparison in {group} group.")
             # Continue to origin analysis, skip involvement tests
        else:
            print(f"Preparing paired data for stages: {', '.join(stages_with_data)}")
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

            print(f"Found {len(valid_subjects_for_pairing)} subjects with complete data across stages for paired tests.")

            # --- Perform Paired Statistical Tests on Involvement ---
            involvement_test_results = []
            if len(valid_subjects_for_pairing) >= 2: # Need at least 2 subjects for paired tests
                # Prepare data for Friedman/Wilcoxon: list of lists/arrays
                friedman_data_input = [paired_involvement_data[stage] for stage in stages_with_data]

                # Friedman Test (if 3+ stages)
                if len(stages_with_data) >= 3:
                    friedman_result = perform_friedman_test(*friedman_data_input)
                    if friedman_result:
                        involvement_test_results.append(friedman_result)
                        print(f"\nFriedman Test for Involvement in {group} Group: "
                              f"Statistic={friedman_result['Statistic']:.2f}, p={friedman_result['P_Value']:.4f}")
                        if friedman_result['Significant']:
                            print(f"Significant differences detected in involvement between stages for {group} Group.")
                            # Perform post-hoc Wilcoxon only if Friedman is significant
                            print("\nPost-hoc Wilcoxon Signed-Rank Tests (FDR corrected):")
                            wilcoxon_results = perform_wilcoxon_posthoc(friedman_data_input, stages_with_data)
                            involvement_test_results.extend(wilcoxon_results)
                            for res in wilcoxon_results:
                                print(f"  {res['Metric'].split(': ')[1]}: p={res['P_Value']:.4f} {'*' if res['Significant'] else ''}")
                        else:
                             print("Friedman test not significant, skipping post-hoc tests.")
                    else:
                        print("Could not perform Friedman test.")
                # Wilcoxon Test (if exactly 2 stages)
                elif len(stages_with_data) == 2:
                     print("\nPerforming Wilcoxon Signed-Rank Test (only 2 stages):")
                     wilcoxon_results = perform_wilcoxon_posthoc(friedman_data_input, stages_with_data)
                     involvement_test_results.extend(wilcoxon_results)
                     for res in wilcoxon_results:
                         print(f"  {res['Metric'].split(': ')[1]}: Stat={res['Statistic']:.2f}, p={res['P_Value']:.4f} {'*' if res['Significant'] else ''}")
                else:
                    print("Not enough stages for comparison.")

            else:
                print(f"Not enough subjects ({len(valid_subjects_for_pairing)}) with complete data across stages for paired tests in {group} group.")

            if not involvement_test_results:
                 print(f"No paired statistical tests performed for involvement in {group} Group.")


        # --- Calculate Descriptive Involvement Stats (using all available data per stage) ---
        involvement_stats = {}
        print(f"\nOverall Involvement Statistics for {group} Group (Descriptive):")
        for stage in stages:
            # Aggregate all scores for this stage across all subjects and protocols
            all_stage_scores = []
            for subject in group_data:
                 all_stage_scores.extend(subject_stage_involvement[subject][stage])
            involvement_stats[stage] = calculate_involvement_statistics(all_stage_scores)
            stats = involvement_stats[stage]
            print(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']})")


        # --- Calculate Origin Data (Aggregated across subjects for the group) ---
        origin_data = {}
        print(f"\nAggregated Origin Statistics for {group} Group:")
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
                print(f"  {stage.capitalize()} (n={tw} waves):")
                ordered_regions = [r for r in master_region_list if r in rc]
                displayed = 0
                for region in ordered_regions:
                    if displayed >= 5:
                        break
                    count = rc[region]
                    percentage = (count / tw) * 100
                    print(f"    {region}: {count}/{tw} waves ({percentage:.1f}%)")
                    displayed += 1
            else:
                print(f"  {stage.capitalize()}: No origins detected")

        # Perform stats on origin distribution (comparing stages within the group)
        # This uses the aggregated origin counts per stage for the group
        origin_test_results = perform_origin_distribution_tests(origin_data, master_region_list)

        for result in origin_test_results:
            if 'DF' in result:
                print(f"\n{result['Test']} for Origin Distribution in {group} Group: "
                      f"χ²={result['Statistic']:.2f}, df={result['DF']}, p={result['P_Value']:.4f}")
            else:
                print(f"\n{result['Test']} for Origin Distribution in {group} Group: "
                      f"p={result['P_Value']:.4f}")
            if result['Significant']:
                print(f"Significant differences detected in origin distribution between stages for {group} Group.")

        if not origin_test_results:
            print(f"\nNot enough data for statistical test of origin distribution in {group} Group.")

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
