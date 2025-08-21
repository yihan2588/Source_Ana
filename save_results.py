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
    For proto1, also saves subject-specific involvement data

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

    # Save proto1 subject-specific involvement data if available
    proto1_subject_file = None
    proto1_summary_file = None
    proto1_subject_data = proto_specific_results.get('proto1_subject_specific_data')
    if proto1_subject_data:
        logging.info("Saving proto1 subject-specific involvement data...")
        subject_specific_list = []
        
        for group in proto1_subject_data:
            for stage in proto1_subject_data[group]:
                for subject_id, involvement_values in proto1_subject_data[group][stage].items():
                    for wave_idx, involvement_value in enumerate(involvement_values, 1):
                        subject_specific_list.append({
                            'Treatment_Group': group,
                            'Stage': stage,
                            'Subject_ID': subject_id,
                            'Wave_Number': wave_idx,
                            'Involvement_Percentage': involvement_value
                        })
        
        if subject_specific_list:
            subject_df = pd.DataFrame(subject_specific_list)
            proto1_subject_file = os.path.join(involvement_dir, "proto1_subject_specific_involvement.csv")
            subject_df.to_csv(proto1_subject_file, index=False)
            logging.info(f"Saved proto1 subject-specific involvement data to {proto1_subject_file}")
            
            # Also save a summary version with subject averages
            summary_list = []
            for group in proto1_subject_data:
                for stage in proto1_subject_data[group]:
                    for subject_id, involvement_values in proto1_subject_data[group][stage].items():
                        if involvement_values:
                            summary_list.append({
                                'Treatment_Group': group,
                                'Stage': stage,
                                'Subject_ID': subject_id,
                                'Mean_Involvement': np.mean(involvement_values),
                                'Std_Involvement': np.std(involvement_values) if len(involvement_values) > 1 else 0,
                                'Wave_Count': len(involvement_values)
                            })
            
            if summary_list:
                summary_df = pd.DataFrame(summary_list)
                proto1_summary_file = os.path.join(involvement_dir, "proto1_subject_averages_involvement.csv")
                summary_df.to_csv(proto1_summary_file, index=False)
                logging.info(f"Saved proto1 subject summary data to {proto1_summary_file}")

    return {
        'proto_specific_involvement_stats': involvement_file,
        'proto_specific_origin_stats': origin_file,
        'proto_specific_comparison_tests': tests_file,
        'proto1_subject_specific_file': proto1_subject_file,
        'proto1_subject_summary_file': proto1_summary_file if proto1_subject_data else None
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

def save_consolidated_statistical_results(consolidated_tests, source_dir=None):
    """
    Save consolidated statistical test results from all analysis types to a single CSV file
    
    Args:
        consolidated_tests: List of consolidated statistical test results from consolidate_statistical_results()
        source_dir: Source directory where data is read from, used to construct output path
    """
    if not consolidated_tests:
        logging.warning("No consolidated statistical tests to save")
        return None
    
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")
    
    # Define consolidated stats directory
    consolidated_dir = os.path.join(base_output_dir, "stats", "consolidated")
    os.makedirs(consolidated_dir, exist_ok=True)
    
    logging.info("\n=== Saving Consolidated Statistical Results ===")
    
    # Convert to DataFrame and save
    consolidated_df = pd.DataFrame(consolidated_tests)
    consolidated_file = os.path.join(consolidated_dir, "all_statistical_test_results.csv")
    consolidated_df.to_csv(consolidated_file, index=False)
    
    # Log summary statistics
    total_tests = len(consolidated_tests)
    significant_tests = len([t for t in consolidated_tests if t.get('Significant', False)])
    
    # Count by analysis type
    analysis_counts = Counter([t.get('Analysis_Type', 'Unknown') for t in consolidated_tests])
    comparison_counts = Counter([t.get('Comparison_Type', 'Unknown') for t in consolidated_tests])
    test_counts = Counter([t.get('Test_Type', 'Unknown') for t in consolidated_tests])
    
    logging.info(f"Saved consolidated statistical results to {consolidated_file}")
    logging.info(f"Total tests: {total_tests}, Significant: {significant_tests} ({significant_tests/total_tests*100:.1f}%)")
    logging.info(f"Tests by analysis type: {dict(analysis_counts)}")
    logging.info(f"Tests by comparison type: {dict(comparison_counts)}")
    logging.info(f"Tests by test type: {dict(test_counts)}")
    
    return consolidated_file


def save_involvement_percentage_changes(percentage_change_data, source_dir=None):
    """
    Save involvement percentage change data to CSV files and return file paths
    
    Args:
        percentage_change_data: Dictionary containing percentage change data from calculate_involvement_percentage_changes()
        source_dir: Source directory where data is read from, used to construct output path
    
    Returns:
        Dictionary containing paths to saved CSV files
    """
    if not percentage_change_data:
        logging.warning("No percentage change data to save")
        return {}
    
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")
    
    # Define percentage change directory
    change_dir = os.path.join(base_output_dir, "involvement", "percentage_changes")
    os.makedirs(change_dir, exist_ok=True)
    
    logging.info("\n=== Saving Involvement Percentage Changes ===")
    
    saved_files = {}
    
    # --- Save Overall Percentage Changes ---
    if 'overall' in percentage_change_data and percentage_change_data['overall']:
        overall_data = []
        overall_changes = percentage_change_data['overall']
        
        for group in overall_changes:
            for comp_type in overall_changes[group]:
                # Handle aggregated statistics format (not subject-level)
                if isinstance(overall_changes[group][comp_type], dict):
                    stats = overall_changes[group][comp_type]
                    overall_data.append({
                        'Analysis_Type': 'Overall',
                        'Protocol': 'All_Combined',
                        'Treatment_Group': group,
                        'Comparison_Type': comp_type,
                        'Percentage_Change_Mean': stats.get('percentage_change_mean', 0),
                        'Percentage_Change_Std': stats.get('percentage_change_std', 0),
                        'Pre_Mean': stats.get('pre_mean', 0),
                        'Target_Mean': stats.get('target_mean', 0),
                        'Count': stats.get('count', 0)
                    })
        
        if overall_data:
            overall_df = pd.DataFrame(overall_data)
            overall_file = os.path.join(change_dir, "overall_percentage_changes.csv")
            overall_df.to_csv(overall_file, index=False)
            saved_files['overall'] = overall_file
            
            logging.info(f"Saved overall percentage changes to {overall_file}")
            # Log summary from the data
            for row in overall_data:
                logging.info(f"  Overall {row['Treatment_Group']} {row['Comparison_Type']}: {row['Percentage_Change_Mean']:.1f}% ± {row['Percentage_Change_Std']:.1f}% (n={row['Count']})")
    
    # --- Save Protocol-Specific Percentage Changes ---
    if 'protocol_specific' in percentage_change_data and percentage_change_data['protocol_specific']:
        proto_data = []
        proto_changes = percentage_change_data['protocol_specific']
        
        for protocol in proto_changes:
            for group in proto_changes[protocol]:
                for comp_type in proto_changes[protocol][group]:
                    # Handle aggregated statistics format (not subject-level)
                    if isinstance(proto_changes[protocol][group][comp_type], dict):
                        stats = proto_changes[protocol][group][comp_type]
                        proto_data.append({
                            'Analysis_Type': 'Protocol_Specific',
                            'Protocol': protocol,
                            'Treatment_Group': group,
                            'Comparison_Type': comp_type,
                            'Percentage_Change_Mean': stats.get('percentage_change_mean', 0),
                            'Percentage_Change_Std': stats.get('percentage_change_std', 0),
                            'Pre_Mean': stats.get('pre_mean', 0),
                            'Target_Mean': stats.get('target_mean', 0),
                            'Count': stats.get('count', 0)
                        })
        
        if proto_data:
            proto_df = pd.DataFrame(proto_data)
            proto_file = os.path.join(change_dir, "protocol_specific_percentage_changes.csv")
            proto_df.to_csv(proto_file, index=False)
            saved_files['protocol_specific'] = proto_file
            
            logging.info(f"Saved protocol-specific percentage changes to {proto_file}")
            # Log summary from the data
            for row in proto_data:
                logging.info(f"  {row['Protocol']} {row['Treatment_Group']} {row['Comparison_Type']}: {row['Percentage_Change_Mean']:.1f}% ± {row['Percentage_Change_Std']:.1f}% (n={row['Count']})")
    
    # --- Save Combined Summary File ---
    overall_data_exists = 'overall_data' in locals() and overall_data
    proto_data_exists = 'proto_data' in locals() and proto_data
    
    if overall_data_exists or proto_data_exists:
        combined_data = []
        if overall_data_exists:
            combined_data.extend(overall_data)
        if proto_data_exists:
            combined_data.extend(proto_data)
            
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            combined_file = os.path.join(change_dir, "all_percentage_changes.csv")
            combined_df.to_csv(combined_file, index=False)
            saved_files['combined'] = combined_file
            logging.info(f"Saved combined percentage changes to {combined_file}")
    
    # Log final summary
    total_records = (len(overall_data) if overall_data_exists else 0) + (len(proto_data) if proto_data_exists else 0)
    logging.info(f"\nPercentage change saving complete:")
    logging.info(f"  Total records saved: {total_records}")
    logging.info(f"  Files created: {len(saved_files)}")
    
    return saved_files

# The save_meta_results function has been removed as it doesn't separate treatment groups
