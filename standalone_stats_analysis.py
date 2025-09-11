#!/usr/bin/env python3
"""
Standalone Statistical Analysis for Wave-Level CSV Data

This script takes existing wave-level involvement CSV data and performs comprehensive statistical analysis:
- Overall analysis (all protocols combined)
- Proto-specific analysis (each protocol separately)
- Within-group comparisons (Wilcoxon signed-rank test)
- Between-group comparisons (Mann-Whitney U test)
- Descriptive statistics for all comparisons
- All results consolidated into one CSV file

Usage:
    python standalone_stats_analysis.py <wave_csv_file> <output_directory>
    
Expected CSV columns:
    - Subject_ID
    - Treatment_Group  
    - Protocol
    - Stage
    - Involvement_Percentage
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import itertools


def setup_logging(output_dir):
    """Set up logging for the analysis."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "standalone_stats_analysis.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def load_and_validate_wave_data(csv_file_path):
    """
    Load and validate wave-level CSV data.
    
    Args:
        csv_file_path: Path to the wave-level CSV file
        
    Returns:
        DataFrame with validated wave data
    """
    
    logging.info(f"Loading wave data from: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    
    # Validate required columns
    required_columns = ['Subject_ID', 'Treatment_Group', 'Protocol', 'Stage', 'Involvement_Percentage']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean and validate data
    df = df.dropna(subset=['Involvement_Percentage'])
    df = df[df['Involvement_Percentage'].between(0, 100)]  # Valid percentage range
    
    logging.info(f"Loaded {len(df)} valid wave records")
    logging.info(f"Subjects: {df['Subject_ID'].nunique()}")
    logging.info(f"Treatment groups: {df['Treatment_Group'].value_counts().to_dict()}")
    logging.info(f"Protocols: {sorted(df['Protocol'].unique())}")
    logging.info(f"Stages: {sorted(df['Stage'].unique())}")
    
    return df


def create_subject_averaged_data(wave_df, group_by_protocol=False):
    """
    Create subject-averaged involvement data.
    
    Args:
        wave_df: DataFrame with wave-level data
        group_by_protocol: If True, create separate averages for each protocol
        
    Returns:
        DataFrame with subject-averaged data
    """
    
    if group_by_protocol:
        # Group by protocol as well
        grouping_columns = ['Subject_ID', 'Treatment_Group', 'Protocol', 'Stage']
        logging.info("Creating subject-averaged data (protocol-specific)")
    else:
        # Overall analysis - combine all protocols
        grouping_columns = ['Subject_ID', 'Treatment_Group', 'Stage'] 
        logging.info("Creating subject-averaged data (overall)")
    
    subject_means = wave_df.groupby(grouping_columns).agg({
        'Involvement_Percentage': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    if group_by_protocol:
        subject_means.columns = ['Subject_ID', 'Treatment_Group', 'Protocol', 'Stage', 
                               'Mean_Involvement', 'Std_Involvement', 'N_Waves']
    else:
        subject_means.columns = ['Subject_ID', 'Treatment_Group', 'Stage',
                               'Mean_Involvement', 'Std_Involvement', 'N_Waves']
    
    logging.info(f"Created {len(subject_means)} subject-averaged records")
    
    return subject_means


def perform_within_group_analysis(subject_means_df, analysis_level, protocol=None):
    """
    Perform within-group statistical analysis using Wilcoxon signed-rank test.
    
    Args:
        subject_means_df: DataFrame with subject-averaged data
        analysis_level: 'Overall' or 'Proto_Specific'
        protocol: Protocol name (if proto-specific)
        
    Returns:
        List of result dictionaries
    """
    
    results = []
    groups = sorted(subject_means_df['Treatment_Group'].unique())
    stages = sorted(subject_means_df['Stage'].unique())
    
    logging.info(f"Performing within-group analysis - {analysis_level}")
    if protocol:
        logging.info(f"Protocol: {protocol}")
    
    for group in groups:
        logging.info(f"\n{group} Group:")
        group_data = subject_means_df[subject_means_df['Treatment_Group'] == group]
        
        # Get data for each stage
        stage_data = {}
        for stage in stages:
            stage_subjects = group_data[group_data['Stage'] == stage]
            if len(stage_subjects) > 0:
                stage_data[stage] = stage_subjects[['Subject_ID', 'Mean_Involvement']].set_index('Subject_ID')['Mean_Involvement']
        
        # Perform pairwise comparisons
        stage_pairs = list(itertools.combinations(stages, 2))
        
        for stage1, stage2 in stage_pairs:
            if stage1 not in stage_data or stage2 not in stage_data:
                continue
            
            # Get paired data (subjects who have both stages)
            common_subjects = stage_data[stage1].index.intersection(stage_data[stage2].index)
            
            if len(common_subjects) < 3:
                logging.warning(f"  {stage1} vs {stage2}: Only {len(common_subjects)} paired subjects - insufficient")
                continue
            
            paired_data1 = stage_data[stage1][common_subjects].values
            paired_data2 = stage_data[stage2][common_subjects].values
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = scipy_stats.wilcoxon(paired_data1, paired_data2, alternative='two-sided')
                
                # Calculate effect size (Cohen's d for paired data)
                differences = paired_data2 - paired_data1
                effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
                
                logging.info(f"  {stage1} vs {stage2} (n={len(common_subjects)} paired subjects):")
                logging.info(f"    {stage1}: {np.mean(paired_data1):.2f}% ± {np.std(paired_data1):.2f}%")
                logging.info(f"    {stage2}: {np.mean(paired_data2):.2f}% ± {np.std(paired_data2):.2f}%")
                logging.info(f"    Wilcoxon: W={w_stat:.3f}, p={w_p:.6f} {'***' if w_p < 0.05 else ''}")
                
                results.append({
                    'Analysis_Level': analysis_level,
                    'Protocol': protocol if protocol else 'All',
                    'Test_Type': 'Within_Group',
                    'Group': group,
                    'Stage1': stage1,
                    'Stage2': stage2,
                    'Comparison': f"{stage2} vs {stage1}",
                    'Test_Method': 'Wilcoxon Signed-Rank',
                    'N_Subjects': len(common_subjects),
                    'Mean1': np.mean(paired_data1),
                    'Std1': np.std(paired_data1),
                    'Mean2': np.mean(paired_data2),
                    'Std2': np.std(paired_data2),
                    'Mean_Difference': np.mean(differences),
                    'Statistic': w_stat,
                    'P_Value': w_p,
                    'Significant': w_p < 0.05,
                    'Effect_Size': effect_size
                })
                
            except Exception as e:
                logging.error(f"  {stage1} vs {stage2}: Wilcoxon test failed - {e}")
    
    return results


def perform_between_group_analysis(subject_means_df, analysis_level, protocol=None):
    """
    Perform between-group statistical analysis using Mann-Whitney U test.
    
    Args:
        subject_means_df: DataFrame with subject-averaged data
        analysis_level: 'Overall' or 'Proto_Specific' 
        protocol: Protocol name (if proto-specific)
        
    Returns:
        List of result dictionaries
    """
    
    results = []
    groups = sorted(subject_means_df['Treatment_Group'].unique())
    stages = sorted(subject_means_df['Stage'].unique())
    
    logging.info(f"Performing between-group analysis - {analysis_level}")
    if protocol:
        logging.info(f"Protocol: {protocol}")
    
    for stage in stages:
        logging.info(f"\n{stage} Stage:")
        stage_data = subject_means_df[subject_means_df['Stage'] == stage]
        
        # Get data for each group
        group_data = {}
        for group in groups:
            group_stage_data = stage_data[stage_data['Treatment_Group'] == group]
            if len(group_stage_data) > 0:
                group_data[group] = group_stage_data['Mean_Involvement'].values
        
        # Perform between-group comparison
        if 'Active' in group_data and 'SHAM' in group_data:
            active_data = group_data['Active']
            sham_data = group_data['SHAM']
            
            if len(active_data) >= 3 and len(sham_data) >= 3:
                try:
                    u_stat, u_p = scipy_stats.mannwhitneyu(active_data, sham_data, alternative='two-sided')
                    
                    # Calculate effect size (Cohen's d for independent groups)
                    pooled_std = np.sqrt(((len(active_data) - 1) * np.var(active_data, ddof=1) + 
                                        (len(sham_data) - 1) * np.var(sham_data, ddof=1)) / 
                                       (len(active_data) + len(sham_data) - 2))
                    effect_size = (np.mean(active_data) - np.mean(sham_data)) / pooled_std if pooled_std > 0 else 0
                    
                    logging.info(f"  Active (n={len(active_data)}): {np.mean(active_data):.2f}% ± {np.std(active_data):.2f}%")
                    logging.info(f"  SHAM (n={len(sham_data)}): {np.mean(sham_data):.2f}% ± {np.std(sham_data):.2f}%")
                    logging.info(f"  Mann-Whitney U: U={u_stat:.3f}, p={u_p:.6f} {'***' if u_p < 0.05 else ''}")
                    
                    results.append({
                        'Analysis_Level': analysis_level,
                        'Protocol': protocol if protocol else 'All',
                        'Test_Type': 'Between_Group',
                        'Group': 'Active_vs_SHAM',
                        'Stage1': stage,
                        'Stage2': None,
                        'Comparison': f'Active vs SHAM ({stage})',
                        'Test_Method': 'Mann-Whitney U',
                        'N_Subjects': f"{len(active_data)},{len(sham_data)}",
                        'Mean1': np.mean(active_data),
                        'Std1': np.std(active_data),
                        'Mean2': np.mean(sham_data),
                        'Std2': np.std(sham_data),
                        'Mean_Difference': np.mean(active_data) - np.mean(sham_data),
                        'Statistic': u_stat,
                        'P_Value': u_p,
                        'Significant': u_p < 0.05,
                        'Effect_Size': effect_size
                    })
                    
                except Exception as e:
                    logging.error(f"  Mann-Whitney U test failed for {stage}: {e}")
            else:
                logging.warning(f"  Insufficient data: Active n={len(active_data)}, SHAM n={len(sham_data)}")
    
    return results


def analyze_wave_csv_comprehensive(csv_file_path, output_dir):
    """
    Main function to perform comprehensive statistical analysis on wave-level CSV data.
    
    Args:
        csv_file_path: Path to wave-level CSV file
        output_dir: Output directory for results
        
    Returns:
        Path to consolidated results CSV file
    """
    
    # Set up logging
    log_file = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("STANDALONE STATISTICAL ANALYSIS FOR WAVE-LEVEL CSV DATA")
    logging.info("="*80)
    logging.info(f"Input CSV: {csv_file_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file}")
    
    # Load and validate data
    wave_df = load_and_validate_wave_data(csv_file_path)
    
    all_results = []
    
    # ========================================
    # OVERALL ANALYSIS (All Protocols Combined)
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("OVERALL ANALYSIS (ALL PROTOCOLS COMBINED)")
    logging.info("="*60)
    
    # Create subject-averaged data (overall)
    overall_subject_means = create_subject_averaged_data(wave_df, group_by_protocol=False)
    
    # Within-group analysis (overall)
    overall_within_results = perform_within_group_analysis(
        overall_subject_means, 
        analysis_level='Overall', 
        protocol=None
    )
    all_results.extend(overall_within_results)
    
    # Between-group analysis (overall)
    overall_between_results = perform_between_group_analysis(
        overall_subject_means,
        analysis_level='Overall',
        protocol=None
    )
    all_results.extend(overall_between_results)
    
    # ========================================
    # PROTO-SPECIFIC ANALYSIS
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("PROTO-SPECIFIC ANALYSIS")
    logging.info("="*60)
    
    protocols = sorted(wave_df['Protocol'].unique())
    logging.info(f"Analyzing protocols: {protocols}")
    
    for protocol in protocols:
        logging.info(f"\n--- {protocol.upper()} ---")
        
        # Filter data for this protocol
        protocol_data = wave_df[wave_df['Protocol'] == protocol]
        
        if len(protocol_data) == 0:
            logging.warning(f"No data for protocol {protocol}")
            continue
        
        # Create subject-averaged data (protocol-specific)
        protocol_subject_means = create_subject_averaged_data(protocol_data, group_by_protocol=True)
        
        # Within-group analysis (protocol-specific)
        protocol_within_results = perform_within_group_analysis(
            protocol_subject_means,
            analysis_level='Proto_Specific', 
            protocol=protocol
        )
        all_results.extend(protocol_within_results)
        
        # Between-group analysis (protocol-specific)
        protocol_between_results = perform_between_group_analysis(
            protocol_subject_means,
            analysis_level='Proto_Specific',
            protocol=protocol
        )
        all_results.extend(protocol_between_results)
    
    # ========================================
    # SAVE CONSOLIDATED RESULTS
    # ========================================
    logging.info("\n" + "="*60)
    logging.info("SAVING CONSOLIDATED RESULTS")
    logging.info("="*60)
    
    if not all_results:
        logging.warning("No statistical results generated!")
        return None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Round numerical columns for readability
    numeric_columns = ['Mean1', 'Std1', 'Mean2', 'Std2', 'Mean_Difference', 'Statistic', 'P_Value', 'Effect_Size']
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(6)
    
    # Sort results logically
    sort_columns = ['Analysis_Level', 'Protocol', 'Test_Type', 'Group', 'Stage1', 'Stage2']
    available_sort_columns = [col for col in sort_columns if col in results_df.columns]
    results_df = results_df.sort_values(available_sort_columns).reset_index(drop=True)
    
    # Save consolidated results
    output_file = os.path.join(output_dir, "comprehensive_statistical_results.csv")
    results_df.to_csv(output_file, index=False)
    
    logging.info(f"Comprehensive results saved: {output_file}")
    logging.info(f"Total statistical tests performed: {len(results_df)}")
    
    # Summary statistics
    significant_tests = results_df[results_df['Significant'] == True]
    logging.info(f"Significant results: {len(significant_tests)}/{len(results_df)} ({100*len(significant_tests)/len(results_df):.1f}%)")
    
    if len(significant_tests) > 0:
        logging.info("\nSignificant Results Summary:")
        for _, row in significant_tests.iterrows():
            logging.info(f"  {row['Analysis_Level']} - {row['Comparison']}: p={row['P_Value']:.6f}")
    
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    
    return output_file


def main():
    """Main entry point for the script."""
    
    if len(sys.argv) != 3:
        print("Usage: python standalone_stats_analysis.py <wave_csv_file> <output_directory>")
        print("\nExpected CSV columns:")
        print("  - Subject_ID")
        print("  - Treatment_Group") 
        print("  - Protocol")
        print("  - Stage")
        print("  - Involvement_Percentage")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        result_file = analyze_wave_csv_comprehensive(csv_file_path, output_dir)
        
        if result_file:
            print(f"\n{'='*60}")
            print("COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED")
            print(f"{'='*60}")
            print(f"Results saved to: {result_file}")
            print(f"Log file: {os.path.join(output_dir, 'standalone_stats_analysis.log')}")
            print("\nThis analysis includes:")
            print("• Overall analysis (all protocols combined)")
            print("• Proto-specific analysis (each protocol separately)")
            print("• Within-group comparisons (Wilcoxon signed-rank)")
            print("• Between-group comparisons (Mann-Whitney U)")
            print("• Descriptive statistics for all comparisons")
            print("• All results consolidated in one CSV file")
        else:
            print("Analysis failed - check log file for details")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
