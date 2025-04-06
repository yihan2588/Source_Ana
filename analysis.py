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
from stats_utils import (
    perform_involvement_tests,
    perform_origin_distribution_tests,
    perform_chi_square_or_fisher_test
)


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
            
            # Process the CSV files in the SourceRecon directory
            subject_results = process_directory(source_recon_dir, quiet=True, visualize_regions=visualize_regions, source_dir=source_dir)
            if subject_results:
                # Merge the subject results into the treatment group results
                for protocol in subject_results:
                    if protocol not in results_by_treatment_group[group]:
                        results_by_treatment_group[group][protocol] = {'pre': [], 'early': [], 'late': [], 'post': []}
                    for stage in ['pre', 'early', 'late', 'post']:
                        results_by_treatment_group[group][protocol][stage].extend(subject_results[protocol][stage])
                
                # Update file counts
                for protocol in subject_results:
                    for stage in subject_results[protocol]:
                        total_files += len(subject_results[protocol][stage])
                        processed_files += len(subject_results[protocol][stage])
    
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
    stage_pattern = r'(pre|early|late|post)-stim'

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
                results_by_protocol[protocol] = {'pre': [], 'early': [], 'late': [], 'post': []}

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
                    from visualize import visualize_region_time_series
                    # Use source_dir if provided, otherwise use the standard 'results' directory
                    visualize_region_time_series(result, csv_file, output_dir="results", source_dir=source_dir)
                
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
    stages = protocol_results.keys()

    # Calculate involvement data and statistics
    involvement_data = {
        stage: [res['involvement_percentage'] for res in protocol_results[stage]]
        for stage in stages
    }
    involvement_stats = {
        stage: calculate_involvement_statistics(involvement_data[stage])
        for stage in stages
    }

    print("\nInvolvement Statistics:")
    for stage, stats_data in involvement_stats.items():
        print(f"{stage.capitalize()}: {stats_data['mean']:.1f}% ± {stats_data['std']:.1f}% (n={stats_data['count']})")

    # Perform statistical tests on involvement data
    involvement_test_results = perform_involvement_tests(involvement_data)

    # Print involvement test results
    has_kw = False
    for result in involvement_test_results:
        if result['Test'] == 'Kruskal-Wallis':
            has_kw = True
            print(f"\nKruskal-Wallis Test for Involvement: H={result['Statistic']:.2f}, p={result['P_Value']:.4f}")
            if result['Significant']:
                print("Significant differences detected in involvement between stages.")
                print("\nPost-hoc Mann-Whitney U Tests:")
        elif result['Test'] == 'Mann-Whitney U':
            metric_parts = result['Metric'].split(': ')[1].split(' vs ')
            print(f"  {metric_parts[0]} vs {metric_parts[1]}: p={result['P_Value']:.4f} {'*' if result['Significant'] else ''}")

    if (not has_kw) and (not involvement_test_results):
        print("Not enough stages with data for statistical comparison of involvement.")

    # Calculate origin statistics
    #  -- We'll store each stage's 'region_counts' and 'total_waves'
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
        'involvement_data': involvement_data,
        'origin_data': origin_data,
        'involvement_stats': involvement_stats,
        'involvement_test_results': involvement_test_results,
        'origin_test_results': origin_test_results
    }


def analyze_treatment_groups(results_by_treatment_group, master_region_list):
    """
    Analyze and compare results between treatment groups (Active vs. SHAM).
    Returns a dictionary with treatment comparison results.
    """
    print("\n=== Treatment Group Comparison Analysis ===")

    # Collect data by treatment group
    treatment_data = collect_data_by_treatment_group(results_by_treatment_group)

    # Calculate involvement statistics
    treatment_involvement_stats = {}
    treatment_involvement_data = {}

    for group in treatment_data:
        involvement_by_stage = {
            stage: [res['involvement_percentage'] for res in treatment_data[group][stage]]
            for stage in treatment_data[group]
        }
        stats_by_stage = {
            stage: calculate_involvement_statistics(involvement_by_stage[stage])
            for stage in involvement_by_stage
        }
        treatment_involvement_stats[group] = stats_by_stage
        treatment_involvement_data[group] = involvement_by_stage

    print("\nInvolvement Statistics by Treatment Group:")
    for group in treatment_involvement_stats:
        print(f"\n{group} Group:")
        for stage, stats in treatment_involvement_stats[group].items():
            print(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']})")

    # Perform statistical tests comparing treatment groups for each stage
    treatment_comparison_tests = []
    stages = ['pre', 'early', 'late', 'post']
    if 'Active' in treatment_involvement_data and 'SHAM' in treatment_involvement_data:
        for stage in stages:
            active_data = treatment_involvement_data.get('Active', {}).get(stage, [])
            sham_data = treatment_involvement_data.get('SHAM', {}).get(stage, [])
            if active_data and sham_data:
                try:
                    u_stat, p_val = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                    treatment_comparison_tests.append({
                        'Test': 'Mann-Whitney U',
                        'Metric': f'Involvement: Active vs SHAM ({stage})',
                        'Stage': stage,
                        'Statistic': u_stat,
                        'P_Value': p_val,
                        'Significant': p_val < 0.05
                    })
                    print(f"\nMann-Whitney U Test for {stage} stage (Active vs SHAM): U={u_stat:.2f}, p={p_val:.4f}")
                    if p_val < 0.05:
                        print(f"Significant difference detected in {stage} involvement between Active and SHAM groups.")
                except Exception as e:
                    print(f"Error performing statistical test for {stage} stage: {str(e)}")

    # Calculate origin statistics
    treatment_origin_data = {}
    for group in treatment_data:
        origin_by_stage = {}
        for stage in treatment_data[group]:
            stage_results = treatment_data[group][stage]
            total_waves = len(stage_results)
            region_counts = calculate_origin_statistics(stage_results, stage, total_waves)
            origin_by_stage[stage] = {
                'region_counts': region_counts,
                'total_waves': total_waves
            }
        treatment_origin_data[group] = origin_by_stage

    print("\nOrigin Distribution by Treatment Group:")
    for group in treatment_origin_data:
        print(f"\n{group} Group:")
        for stage, data_dict in treatment_origin_data[group].items():
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

    # Perform statistical tests on origin distribution between treatment groups
    origin_comparison_tests = []
    if 'Active' in treatment_origin_data and 'SHAM' in treatment_origin_data:
        for stage in stages:
            active_stage = treatment_origin_data['Active'].get(stage, {})
            sham_stage = treatment_origin_data['SHAM'].get(stage, {})
            if active_stage and sham_stage:
                active_origins = active_stage['region_counts']
                sham_origins = sham_stage['region_counts']
                # Create 2-row contingency table for top or all regions
                all_regions = set(active_origins.keys()) | set(sham_origins.keys())
                # Build the table
                if all_regions:
                    contingency_table = []
                    for region in all_regions:
                        # We'll fill these after we pivot to shape (2 x N)
                        pass
                    # Actually, let's do separate arrays for "active" and "sham"
                    # then pass them to the test function.
                    # Filter out empty?
                    active_counts = [active_origins.get(r, 0) for r in all_regions]
                    sham_counts = [sham_origins.get(r, 0) for r in all_regions]
                    contingency_table = np.array([active_counts, sham_counts])

                    # Perform the test
                    test_result = perform_chi_square_or_fisher_test(contingency_table)
                    if test_result:
                        test_result['Metric'] = f'Origin Distribution: Active vs SHAM ({stage})'
                        test_result['Stage'] = stage
                        origin_comparison_tests.append(test_result)

                        # Print results
                        if 'DF' in test_result:
                            print(f"\n{test_result['Test']} for {stage} origin distribution (Active vs SHAM): "
                                  f"χ²={test_result['Statistic']:.2f}, df={test_result['DF']}, p={test_result['P_Value']:.4f}")
                        else:
                            print(f"\n{test_result['Test']} for {stage} origin distribution (Active vs SHAM): "
                                  f"p={test_result['P_Value']:.4f}")
                        if test_result['Significant']:
                            print(f"Significant difference detected in {stage} origin distribution between Active and SHAM groups.")

    return {
        'treatment_involvement_data': treatment_involvement_data,
        'treatment_involvement_stats': treatment_involvement_stats,
        'treatment_origin_data': treatment_origin_data,
        'treatment_comparison_tests': treatment_comparison_tests + origin_comparison_tests,
        'master_region_list': master_region_list
    }


def analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list, min_occurrence_threshold=3):
    """
    Perform overall treatment group comparison by collapsing subjects, nights, and protos.
    """
    print("\n=== Overall Treatment Group Comparison (Collapsing Subjects, Nights, and Protos) ===")

    # Collect all waves for each treatment group across all protocols
    all_waves_by_group = {'Active': {}, 'SHAM': {}}
    for group in results_by_treatment_group:
        all_waves_by_group[group] = {'pre': [], 'early': [], 'late': [], 'post': []}
        for protocol in results_by_treatment_group[group]:
            for stage in ['pre', 'early', 'late', 'post']:
                all_waves_by_group[group][stage].extend(results_by_treatment_group[group][protocol][stage])

    # Calculate involvement
    involvement_stats = {}
    involvement_data = {}
    groups = list(all_waves_by_group.keys())
    stages = ['pre', 'early', 'late', 'post']

    for group in groups:
        involvement_by_stage = {
            stage: [res['involvement_percentage'] for res in all_waves_by_group[group][stage]]
            for stage in stages
        }
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
    # Gather all protocols
    all_protocols = set()
    for group in results_by_treatment_group:
        all_protocols.update(results_by_treatment_group[group].keys())

    proto_specific_results = {}
    stages = ['pre', 'early', 'late', 'post']
    groups = list(results_by_treatment_group.keys())

    for protocol in sorted(all_protocols):
        print(f"\n--- Analysis for {protocol} ---")
        groups_with_data = [g for g in groups if protocol in results_by_treatment_group[g]]
        if len(groups_with_data) < 2:
            print(f"Skipping {protocol} - not enough treatment groups with data.")
            continue

        # Collect waves for each group, for this protocol
        waves_by_group = {}
        for group in groups_with_data:
            waves_by_group[group] = results_by_treatment_group[group][protocol]

        # Involvement
        involvement_stats = {}
        involvement_data = {}
        for group in waves_by_group:
            involvement_by_stage = {
                stage: [res['involvement_percentage'] for res in waves_by_group[group][stage]]
                for stage in stages
            }
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
            'comparison_tests': involvement_comparison_tests + involvement_comparison_tests
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

    # Collect all waves for each treatment group across all protocols
    all_waves_by_group = {}
    for group in results_by_treatment_group:
        all_waves_by_group[group] = {'pre': [], 'early': [], 'late': [], 'post': []}
        for protocol in results_by_treatment_group[group]:
            for stage in ['pre', 'early', 'late', 'post']:
                all_waves_by_group[group][stage].extend(results_by_treatment_group[group][protocol][stage])

    within_group_results = {}
    for group in all_waves_by_group:
        print(f"\n--- Analysis for {group} Group ---")
        # Calculate involvement data
        involvement_data = {
            stage: [res['involvement_percentage'] for res in all_waves_by_group[group][stage]]
            for stage in ['pre', 'early', 'late', 'post']
        }
        involvement_stats = {
            stage: calculate_involvement_statistics(involvement_data[stage])
            for stage in involvement_data
        }

        print(f"\nInvolvement Statistics for {group} Group:")
        for stage, stats in involvement_stats.items():
            print(f"  {stage.capitalize()}: {stats['mean']:.1f}% ± {stats['std']:.1f}% (n={stats['count']})")

        # Perform stats on involvement
        involvement_test_results = perform_involvement_tests(involvement_data)
        has_kw = False
        for result in involvement_test_results:
            if result['Test'] == 'Kruskal-Wallis':
                has_kw = True
                print(f"\nKruskal-Wallis Test for Involvement in {group} Group: "
                      f"H={result['Statistic']:.2f}, p={result['P_Value']:.4f}")
                if result['Significant']:
                    print(f"Significant differences detected in involvement between stages for {group} Group.")
                    print("\nPost-hoc Mann-Whitney U Tests:")
            elif result['Test'] == 'Mann-Whitney U':
                metric_parts = result['Metric'].split(': ')[1].split(' vs ')
                print(f"  {metric_parts[0]} vs {metric_parts[1]}: p={result['P_Value']:.4f} "
                      f"{'*' if result['Significant'] else ''}")

        if (not has_kw) and (not involvement_test_results):
            print(f"Not enough stages with data for statistical comparison of involvement in {group} Group.")

        # Calculate origin data
        origin_data = {}
        for stage in ['pre', 'early', 'late', 'post']:
            stage_results = all_waves_by_group[group][stage]
            total_waves = len(stage_results)
            region_counts = calculate_origin_statistics(stage_results, stage, total_waves)
            origin_data[stage] = {
                'region_counts': region_counts,
                'total_waves': total_waves
            }

        print(f"\nOrigin Statistics for {group} Group:")
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

        # Perform stats on origin distribution
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
            'involvement_data': involvement_data,
            'origin_data': origin_data,
            'involvement_stats': involvement_stats,
            'involvement_test_results': involvement_test_results,
            'origin_test_results': origin_test_results
        }

    return {
        'within_group_results': within_group_results,
        'master_region_list': master_region_list
    }
