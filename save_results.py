import os
import logging # Added
import pandas as pd
import numpy as np
from collections import Counter
# Removed import of collect_wave_level_data as it's unused

# Removed save_statistical_results function as requested

# Removed save_treatment_comparison_results function as requested

def save_overall_treatment_comparison_results(overall_comparison_results, source_dir=None): # Removed output_dir
    """
    Save overall treatment comparison results to CSV files

    Args:
        overall_comparison_results: Results from overall treatment comparison
        source_dir: Source directory where data is read from, used to construct output path
    """
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")

    # Define specific directories for this comparison type
    involvement_dir = os.path.join(base_output_dir, "involvement", "overall_comparison")
    origin_dir = os.path.join(base_output_dir, "origin", "overall_comparison")
    # Use a general output_dir for tests that might span both metrics
    tests_dir = os.path.join(base_output_dir, "stats", "overall_comparison")
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)
    logging.info("\n=== Saving Overall Treatment Comparison Results ===")

    # Save overall involvement statistics
    involvement_stats = overall_comparison_results['overall_involvement_stats']
    involvement_data = []
    
    for group, stats_by_stage in involvement_stats.items():
        for stage, stats in stats_by_stage.items():
            involvement_data.append({
                'Treatment_Group': group,
                'Stage': stage,
                'Mean_Involvement': stats['mean'],
                'Median_Involvement': stats['median'],
                'Std_Involvement': stats['std'],
                'Count': stats['count']
            })
    
    involvement_df = pd.DataFrame(involvement_data)
    # Save to the new involvement directory
    involvement_file = os.path.join(involvement_dir, "overall_involvement_statistics.csv")
    involvement_df.to_csv(involvement_file, index=False)
    logging.info(f"Saved overall involvement statistics to {involvement_file}")

    # Save overall origin statistics
    origin_data = overall_comparison_results['overall_origin_data']
    origin_list = []
    
    for group, origin_by_stage in origin_data.items():
        for stage, stage_info in origin_by_stage.items():
            region_counts = stage_info['region_counts']
            total_waves = stage_info['total_waves']
            for region, count in region_counts.items():
                percentage = (count / total_waves * 100) if total_waves > 0 else 0
                origin_list.append({
                    'Treatment_Group': group,
                    'Stage': stage,
                    'Region': region,
                    'Count': count,
                    'Total_Waves': total_waves,
                    'Percentage': percentage
                })
    
    if origin_list:
        origin_df = pd.DataFrame(origin_list)
        # Save to the new origin directory
        origin_file = os.path.join(origin_dir, "overall_origin_statistics.csv")
        origin_df.to_csv(origin_file, index=False)
        logging.info(f"Saved overall origin statistics to {origin_file}")
    else:
        origin_file = None

    # Save overall comparison test results
    comparison_tests = overall_comparison_results.get('overall_comparison_tests', [])
    tests_file = None
    
    if comparison_tests:
        tests_df = pd.DataFrame(comparison_tests)
        # Save to the new tests directory
        tests_file = os.path.join(tests_dir, "overall_comparison_test_results.csv")
        tests_df.to_csv(tests_file, index=False)
        logging.info(f"Saved overall comparison test results to {tests_file}")

    return {
        'overall_involvement_stats': involvement_file,
        'overall_origin_stats': origin_file,
        'overall_comparison_tests': tests_file
    }

def save_proto_specific_comparison_results(proto_specific_results, source_dir=None): # Removed output_dir
    """
    Save proto-specific comparison results to CSV files

    Args:
        proto_specific_results: Results from proto-specific comparison
        source_dir: Source directory where data is read from, used to construct output path
    """
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")

    # Define specific directories for this comparison type
    involvement_dir = os.path.join(base_output_dir, "involvement", "proto_specific_comparison")
    origin_dir = os.path.join(base_output_dir, "origin", "proto_specific_comparison")
    tests_dir = os.path.join(base_output_dir, "stats", "proto_specific_comparison")
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)
    logging.info("\n=== Saving Proto-Specific Comparison Results ===")

    # Save proto-specific involvement statistics
    involvement_data = []
    
    for protocol, results in proto_specific_results['proto_specific_results'].items():
        for group, stats_by_stage in results['involvement_stats'].items():
            for stage, stats in stats_by_stage.items():
                involvement_data.append({
                    'Protocol': protocol,
                    'Treatment_Group': group,
                    'Stage': stage,
                    'Mean_Involvement': stats['mean'],
                    'Median_Involvement': stats['median'],
                    'Std_Involvement': stats['std'],
                    'Count': stats['count']
                })
    
    involvement_df = pd.DataFrame(involvement_data)
    # Save to the new involvement directory
    involvement_file = os.path.join(involvement_dir, "proto_specific_involvement_statistics.csv")
    involvement_df.to_csv(involvement_file, index=False)
    logging.info(f"Saved proto-specific involvement statistics to {involvement_file}")

    # Save proto-specific origin statistics
    origin_list = []
    
    for protocol, results in proto_specific_results['proto_specific_results'].items():
        for group, origin_by_stage in results['origin_data'].items():
            for stage, stage_info in origin_by_stage.items():
                region_counts = stage_info['region_counts']
                total_waves = stage_info['total_waves']
                for region, count in region_counts.items():
                    percentage = (count / total_waves * 100) if total_waves > 0 else 0
                    origin_list.append({
                        'Protocol': protocol,
                        'Treatment_Group': group,
                        'Stage': stage,
                        'Region': region,
                        'Count': count,
                        'Total_Waves': total_waves,
                        'Percentage': percentage
                    })
    
    if origin_list:
        origin_df = pd.DataFrame(origin_list)
        # Save to the new origin directory
        origin_file = os.path.join(origin_dir, "proto_specific_origin_statistics.csv")
        origin_df.to_csv(origin_file, index=False)
        logging.info(f"Saved proto-specific origin statistics to {origin_file}")
    else:
        origin_file = None

    # Save proto-specific comparison test results
    all_tests = []
    
    for protocol, results in proto_specific_results['proto_specific_results'].items():
        for test in results.get('comparison_tests', []):
            test_copy = test.copy()
            if 'Protocol' not in test_copy:
                test_copy['Protocol'] = protocol
            all_tests.append(test_copy)
    
    tests_file = None
    if all_tests:
        tests_df = pd.DataFrame(all_tests)
        # Save to the new tests directory
        tests_file = os.path.join(tests_dir, "proto_specific_comparison_test_results.csv")
        tests_df.to_csv(tests_file, index=False)
        logging.info(f"Saved proto-specific comparison test results to {tests_file}")

    return {
        'proto_specific_involvement_stats': involvement_file,
        'proto_specific_origin_stats': origin_file,
        'proto_specific_comparison_tests': tests_file
    }

def save_within_group_stage_comparison_results(within_group_results, source_dir=None): # Removed output_dir
    """
    Save within-group stage comparison results to CSV files

    Args:
        within_group_results: Results from within-group stage comparison
        source_dir: Source directory where data is read from, used to construct output path
    """
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")

    # Define specific directories for this comparison type
    involvement_dir = os.path.join(base_output_dir, "involvement", "within_group_comparison")
    origin_dir = os.path.join(base_output_dir, "origin", "within_group_comparison")
    tests_dir = os.path.join(base_output_dir, "stats", "within_group_comparison")
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)
    logging.info("\n=== Saving Within-Group Stage Comparison Results ===")

    # Save within-group involvement statistics
    involvement_data = []
    
    for group, results in within_group_results['within_group_results'].items():
        for stage, stats in results['involvement_stats'].items():
            involvement_data.append({
                'Treatment_Group': group,
                'Stage': stage,
                'Mean_Involvement': stats['mean'],
                'Median_Involvement': stats['median'],
                'Std_Involvement': stats['std'],
                'Count': stats['count']
                })
    
    involvement_df = pd.DataFrame(involvement_data)
    # Save to the new involvement directory
    involvement_file = os.path.join(involvement_dir, "within_group_involvement_statistics.csv")
    involvement_df.to_csv(involvement_file, index=False)
    logging.info(f"Saved within-group involvement statistics to {involvement_file}")

    # Save within-group origin statistics
    origin_list = []
    
    for group, results in within_group_results['within_group_results'].items():
        for stage, stage_info in results['origin_data'].items():
            region_counts = stage_info['region_counts']
            total_waves = stage_info['total_waves']
            for region, count in region_counts.items():
                percentage = (count / total_waves * 100) if total_waves > 0 else 0
                origin_list.append({
                    'Treatment_Group': group,
                    'Stage': stage,
                    'Region': region,
                    'Count': count,
                    'Total_Waves': total_waves,
                    'Percentage': percentage
                })
    
    if origin_list:
        origin_df = pd.DataFrame(origin_list)
        # Save to the new origin directory
        origin_file = os.path.join(origin_dir, "within_group_origin_statistics.csv")
        origin_df.to_csv(origin_file, index=False)
        logging.info(f"Saved within-group origin statistics to {origin_file}")
    else:
        origin_file = None

    # Save within-group test results
    all_tests = []
    
    for group, results in within_group_results['within_group_results'].items():
        for test in results.get('involvement_test_results', []) + results.get('origin_test_results', []):
            test_copy = test.copy()
            test_copy['Treatment_Group'] = group
            all_tests.append(test_copy)
    
    tests_file = None
    if all_tests:
        tests_df = pd.DataFrame(all_tests)
        # Save to the new tests directory
        tests_file = os.path.join(tests_dir, "within_group_test_results.csv")
        tests_df.to_csv(tests_file, index=False)
        logging.info(f"Saved within-group test results to {tests_file}")

    return {
        'within_group_involvement_stats': involvement_file,
        'within_group_origin_stats': origin_file,
        'within_group_test_results': tests_file
    }

# The save_meta_results function has been removed as it doesn't separate treatment groups
