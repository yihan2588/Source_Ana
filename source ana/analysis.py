import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from utils import extract_region_name, compute_master_region_list, calculate_origin_statistics, calculate_involvement_statistics, collect_wave_level_data
from stats_utils import perform_involvement_tests, perform_origin_distribution_tests

def analyze_slow_wave(df, wave_name, window_ms=100, threshold_percent=25, debug=True):
    """Analyze a single slow wave CSV file to extract origin and involvement metrics"""
    if debug:
        print(f"\nAnalyzing {wave_name}...")
    
    try:
        if 'Time' in df.columns:
            numeric_cols = []
            for col in df.columns[1:]:
                if not str(col).startswith('Unnamed'):
                    try:
                        float(col)
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
        'origins': origins,
        'involvement_count': num_involved,
        'involvement_percentage': involvement_percentage,
        'involved_voxels': involved_voxels,
        'window': (window_start, window_end),
        'threshold': threshold
    }

def process_directory(directory_path):
    """Process all CSV files in the directory and organize by protocol and stage"""
    print(f"Processing directory: {directory_path}")
    csv_files = list(Path(directory_path).glob('*.csv'))
    if not csv_files:
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
        protocol_match = re.search(protocol_pattern, filename)
        stage_match = re.search(stage_pattern, filename)
        
        if protocol_match and stage_match:
            protocol = protocol_match.group(1)
            stage = stage_match.group(1)
            if protocol not in results_by_protocol:
                results_by_protocol[protocol] = {'pre': [], 'early': [], 'late': [], 'post': []}
            try:
                df = pd.read_csv(csv_file)
                wave_name = csv_file.stem
                result = analyze_slow_wave(df, wave_name, debug=False)
                results_by_protocol[protocol][stage].append(result)
                processed_files += 1
                print(f"Processed {filename} - Protocol: {protocol}, Stage: {stage}")
            except Exception as e:
                error_files += 1
                print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing summary:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Errors: {error_files}")
    
    return results_by_protocol

# This function has been moved to utils.py

def analyze_protocol_results(protocol_name, protocol_results, master_region_list):
    """Analyze results for a single protocol across all stages"""
    print(f"\n=== Analysis for {protocol_name} ===")
    stages = protocol_results.keys()
    
    # Calculate involvement data and statistics
    involvement_data = {stage: [res['involvement_percentage'] for res in protocol_results[stage]]
                        for stage in stages}
    involvement_stats = {stage: calculate_involvement_statistics(involvement_data[stage])
                        for stage in stages}
    
    print("\nInvolvement Statistics:")
    for stage, stats_data in involvement_stats.items():
        print(f"{stage.capitalize()}: {stats_data['mean']:.1f}% ± {stats_data['std']:.1f}% (n={stats_data['count']})")
    
    # Perform statistical tests on involvement data
    involvement_test_results = perform_involvement_tests(involvement_data)
    
    # Print involvement test results
    for result in involvement_test_results:
        if result['Test'] == 'Kruskal-Wallis':
            print(f"\nKruskal-Wallis Test for Involvement: H={result['Statistic']:.2f}, p={result['P_Value']:.4f}")
            if result['Significant']:
                print("Significant differences detected in involvement between stages.")
                print("\nPost-hoc Mann-Whitney U Tests:")
        elif result['Test'] == 'Mann-Whitney U':
            metric_parts = result['Metric'].split(': ')[1].split(' vs ')
            print(f"  {metric_parts[0]} vs {metric_parts[1]}: p={result['P_Value']:.4f} {'*' if result['Significant'] else ''}")
    
    if not involvement_test_results:
        print("Not enough stages with data for statistical comparison of involvement.")
    
    # Calculate origin statistics
    origin_data = {}
    for stage in stages:
        stage_results = protocol_results[stage]
        total_waves = len(stage_results)
        origin_data[stage] = calculate_origin_statistics(stage_results, stage, total_waves)
    
    print("\nOrigin Statistics:")
    for stage, counts in origin_data.items():
        if counts:
            total_waves = len(protocol_results[stage])
            print(f"{stage.capitalize()} (n={total_waves} waves):")
            ordered_regions = [r for r in master_region_list if r in counts]
            displayed = 0
            for region in ordered_regions:
                if displayed >= 5:
                    break
                count = counts[region]
                percentage = (count / total_waves) * 100
                print(f"  {region}: {count}/{total_waves} waves ({percentage:.1f}%)")
                displayed += 1
        else:
            print(f"{stage.capitalize()}: No origins detected")
    
    # Perform statistical tests on origin distribution
    origin_test_results = perform_origin_distribution_tests(origin_data, master_region_list)
    
    # Print origin distribution test results
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

def analyze_meta_protocol(results_by_protocol, master_region_list):
    """
    Perform meta analysis by collapsing data across all protocols and comparing
    the four stages (pre, early, late, post).
    """
    print("\n=== Meta Protocol Analysis: Collapsing All Protocols ===")
    
    # Use the utility function to collect wave-level data across all protocols
    meta_protocol_waves = collect_wave_level_data(results_by_protocol)
    
    # Calculate involvement data
    meta_involvement_data = {
        stage: [res['involvement_percentage'] for res in meta_protocol_waves[stage]]
        for stage in meta_protocol_waves.keys()
    }
    
    # Calculate involvement statistics
    meta_involvement_stats = {
        stage: calculate_involvement_statistics(meta_involvement_data[stage])
        for stage in meta_involvement_data.keys()
    }
    
    print("\nMeta Involvement Statistics:")
    for stage, stats_data in meta_involvement_stats.items():
        print(f"{stage.capitalize()}: {stats_data['mean']:.1f}% ± {stats_data['std']:.1f}% (n={stats_data['count']})")
    
    # Calculate origin statistics
    meta_origin_data = {}
    for stage in meta_protocol_waves.keys():
        stage_results = meta_protocol_waves[stage]
        total_waves = len(stage_results)
        meta_origin_data[stage] = calculate_origin_statistics(stage_results, stage, total_waves)
    
    # Perform statistical tests on involvement data
    involvement_test_results = perform_involvement_tests(meta_involvement_data)
    
    # Add protocol information to test results
    for result in involvement_test_results:
        result['Protocol'] = 'meta'
    
    # Print involvement test results
    for result in involvement_test_results:
        if result['Test'] == 'Kruskal-Wallis':
            print(f"\nMeta Protocol: Kruskal-Wallis Test for Involvement: H={result['Statistic']:.2f}, p={result['P_Value']:.4f}")
            if result['Significant']:
                print("Significant differences detected in meta involvement between stages.")
                print("\nMeta Protocol: Post-hoc Mann-Whitney U Tests:")
        elif result['Test'] == 'Mann-Whitney U':
            metric_parts = result['Metric'].split(': ')[1].split(' vs ')
            print(f"  {metric_parts[0]} vs {metric_parts[1]}: p={result['P_Value']:.4f} {'*' if result['Significant'] else ''}")
    
    if not involvement_test_results:
        print("Not enough data for meta statistical comparison of involvement.")
    
    print("\nMeta Origin Distribution Analysis:")
    for stage, counts in meta_origin_data.items():
        total_waves = meta_involvement_stats[stage]['count']
        if counts:
            print(f"{stage.capitalize()} (n={total_waves} waves):")
            ordered_regions = [r for r in master_region_list if r in counts]
            displayed = 0
            for region in ordered_regions:
                if displayed >= 5:
                    break
                c = counts[region]
                percentage = (c / total_waves * 100) if total_waves > 0 else 0
                print(f"  {region}: {c}/{total_waves} waves ({percentage:.1f}%)")
                displayed += 1
        else:
            print(f"{stage.capitalize()}: No origins detected")
    
    # Perform statistical tests on origin distribution
    origin_test_results = perform_origin_distribution_tests(meta_origin_data, master_region_list)
    
    # Add protocol information to test results
    for result in origin_test_results:
        result['Protocol'] = 'meta'
    
    # Print origin distribution test results
    for result in origin_test_results:
        if 'DF' in result:
            print(f"\nMeta Protocol: {result['Test']} for Origin Distribution: χ²={result['Statistic']:.2f}, df={result['DF']}, p={result['P_Value']:.4f}")
        else:
            print(f"\nMeta Protocol: {result['Test']} for Origin Distribution: p={result['P_Value']:.4f}")
        
        if result['Significant']:
            print("Significant differences detected in meta origin distribution between stages.")
    
    if not origin_test_results:
        print("\nNot enough data for meta statistical test of origin distribution.")
    
    return {
        'meta_involvement_data': meta_involvement_data,
        'meta_origin_data': meta_origin_data,
        'meta_involvement_stats': meta_involvement_stats,
        'master_region_list': master_region_list,
        'meta_protocol_waves': meta_protocol_waves,
        'meta_stats_results': involvement_test_results + origin_test_results
    }
