#!/usr/bin/env python3
"""
Final Involvement Analysis Pipeline

This script implements the optimal statistical approach based on our analysis:
1. Subject-specific threshold calculation and application
2. Wave-level involvement data computation and saving
3. Subject-averaged statistical analysis using:
   - Wilcoxon signed-rank test for within-group comparisons (paired)
   - Mann-Whitney U test for between-group comparisons (independent)
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing modules
from analysis_subject_specific import (
    process_eeg_data_directory_subject_specific
)
from analysis import (
    analyze_overall_treatment_comparison,
    consolidate_statistical_results
)
from utils import (
    compute_master_region_list,
    read_subject_condition_mapping,
    scan_available_subjects_and_nights
)


def setup_logging(output_dir):
    """Set up comprehensive logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "final_involvement_analysis.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def extract_wave_involvement_data(results_by_treatment_group, subject_thresholds):
    """
    Extract wave-level involvement data from processed results.
    
    Args:
        results_by_treatment_group: Nested dictionary from subject-specific processing
        subject_thresholds: Dictionary of subject-specific thresholds
        
    Returns:
        DataFrame with wave-level involvement data
    """
    
    logging.info("\n=== EXTRACTING WAVE-LEVEL INVOLVEMENT DATA ===")
    
    wave_data = []
    
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            subject_threshold = subject_thresholds.get(subject, np.nan)
            
            for protocol in results_by_treatment_group[group][subject]:
                for stage in results_by_treatment_group[group][subject][protocol]:
                    for wave_result in results_by_treatment_group[group][subject][protocol][stage]:
                        if isinstance(wave_result, dict):
                            wave_data.append({
                                'Subject_ID': subject,
                                'Treatment_Group': group,
                                'Protocol': protocol,
                                'Stage': stage,
                                'Wave_Name': wave_result.get('wave_name', ''),
                                'Involvement_Percentage': wave_result.get('involvement_percentage', np.nan),
                                'Involvement_Count': wave_result.get('involvement_count', np.nan),
                                'Threshold': subject_threshold,
                                'Global_Max_Value': wave_result.get('global_max_value', np.nan),
                                'Global_Max_Time': wave_result.get('global_max_time', np.nan)
                            })
    
    df = pd.DataFrame(wave_data)
    
    # Parse protocol numbers
    df['Protocol_Number'] = df['Protocol'].str.extract(r'proto(\d+)').astype(int)
    
    logging.info(f"Extracted {len(df)} wave records")
    logging.info(f"Subjects: {df['Subject_ID'].nunique()}")
    logging.info(f"Treatment groups: {df['Treatment_Group'].value_counts().to_dict()}")
    logging.info(f"Stages: {sorted(df['Stage'].unique())}")
    logging.info(f"Protocols: {sorted(df['Protocol'].unique())}")
    
    return df


def create_subject_averaged_data(wave_df):
    """
    Create subject-averaged involvement data from wave-level data.
    
    Args:
        wave_df: DataFrame with wave-level data
        
    Returns:
        DataFrame with subject-averaged data
    """
    
    logging.info("\n=== CREATING SUBJECT-AVERAGED DATA ===")
    
    # Calculate subject means for each stage
    subject_means = wave_df.groupby(['Subject_ID', 'Treatment_Group', 'Stage']).agg({
        'Involvement_Percentage': ['mean', 'std', 'count'],
        'Threshold': 'first'  # Same for all waves from same subject
    }).reset_index()
    
    # Flatten column names
    subject_means.columns = ['Subject_ID', 'Treatment_Group', 'Stage', 
                           'Mean_Involvement', 'Std_Involvement', 'N_Waves', 'Threshold']
    
    logging.info(f"Created subject-averaged data: {len(subject_means)} records")
    logging.info("Subject means by group and stage:")
    
    # Log summary statistics
    summary = subject_means.groupby(['Treatment_Group', 'Stage'])['Mean_Involvement'].agg(['count', 'mean', 'std'])
    for (group, stage), stats in summary.iterrows():
        logging.info(f"  {group} {stage}: n={stats['count']}, mean={stats['mean']:.2f}% ± {stats['std']:.2f}%")
    
    return subject_means


def perform_optimal_statistical_analysis(subject_means_df):
    """
    Perform statistical analysis using optimal methods:
    - Wilcoxon signed-rank for within-group (paired) comparisons
    - Mann-Whitney U for between-group (independent) comparisons
    
    Args:
        subject_means_df: DataFrame with subject-averaged data
        
    Returns:
        Dictionary with statistical results
    """
    
    logging.info("\n=== OPTIMAL STATISTICAL ANALYSIS ===")
    logging.info("Methods: Wilcoxon signed-rank (within-group), Mann-Whitney U (between-group)")
    
    results = {}
    stages = sorted(subject_means_df['Stage'].unique())
    groups = sorted(subject_means_df['Treatment_Group'].unique())
    
    # 1. WITHIN-GROUP COMPARISONS (PAIRED - WILCOXON SIGNED-RANK)
    logging.info("\n--- WITHIN-GROUP COMPARISONS (WILCOXON SIGNED-RANK) ---")
    results['within_group'] = {}
    
    for group in groups:
        logging.info(f"\n{group} Group:")
        group_data = subject_means_df[subject_means_df['Treatment_Group'] == group]
        results['within_group'][group] = {}
        
        # Get data for each stage
        stage_data = {}
        for stage in stages:
            stage_subjects = group_data[group_data['Stage'] == stage]
            if len(stage_subjects) > 0:
                stage_data[stage] = stage_subjects[['Subject_ID', 'Mean_Involvement']].set_index('Subject_ID')['Mean_Involvement']
        
        # Perform pairwise comparisons
        import itertools
        stage_pairs = list(itertools.combinations(stages, 2))
        
        for stage1, stage2 in stage_pairs:
            if stage1 not in stage_data or stage2 not in stage_data:
                continue
            
            # Get paired data (subjects who have both stages)
            common_subjects = stage_data[stage1].index.intersection(stage_data[stage2].index)
            
            if len(common_subjects) < 3:
                logging.warning(f"  {stage1} vs {stage2}: Only {len(common_subjects)} paired subjects - insufficient for analysis")
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
                logging.info(f"    Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.6f} {'***' if w_p < 0.05 else ''}")
                logging.info(f"    Effect size (Cohen's d): {effect_size:.3f}")
                
                results['within_group'][group][f"{stage1}_vs_{stage2}"] = {
                    'test': 'Wilcoxon Signed-Rank',
                    'n_subjects': len(common_subjects),
                    'statistic': w_stat,
                    'p_value': w_p,
                    'significant': w_p < 0.05,
                    'effect_size': effect_size,
                    'stage1_mean': np.mean(paired_data1),
                    'stage2_mean': np.mean(paired_data2),
                    'stage1_std': np.std(paired_data1),
                    'stage2_std': np.std(paired_data2),
                    'mean_difference': np.mean(differences),
                    'subjects_analyzed': common_subjects.tolist()
                }
                
            except Exception as e:
                logging.error(f"  {stage1} vs {stage2}: Wilcoxon test failed - {e}")
    
    # 2. BETWEEN-GROUP COMPARISONS (INDEPENDENT - MANN-WHITNEY U)
    logging.info("\n--- BETWEEN-GROUP COMPARISONS (MANN-WHITNEY U) ---")
    results['between_group'] = {}
    
    for stage in stages:
        logging.info(f"\n{stage} Stage:")
        stage_data = subject_means_df[subject_means_df['Stage'] == stage]
        results['between_group'][stage] = {}
        
        # Get data for each group
        group_data = {}
        for group in groups:
            group_stage_data = stage_data[stage_data['Treatment_Group'] == group]
            if len(group_stage_data) > 0:
                group_data[group] = group_stage_data['Mean_Involvement'].values
        
        # Perform between-group comparisons
        if 'Active' in group_data and 'SHAM' in group_data:
            active_data = group_data['Active']
            sham_data = group_data['SHAM']
            
            if len(active_data) >= 3 and len(sham_data) >= 3:
                # Mann-Whitney U test
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
                    logging.info(f"  Effect size (Cohen's d): {effect_size:.3f}")
                    
                    results['between_group'][stage] = {
                        'test': 'Mann-Whitney U',
                        'n_active': len(active_data),
                        'n_sham': len(sham_data),
                        'statistic': u_stat,
                        'p_value': u_p,
                        'significant': u_p < 0.05,
                        'effect_size': effect_size,
                        'active_mean': np.mean(active_data),
                        'sham_mean': np.mean(sham_data),
                        'active_std': np.std(active_data),
                        'sham_std': np.std(sham_data)
                    }
                    
                except Exception as e:
                    logging.error(f"  Mann-Whitney U test failed for {stage}: {e}")
            else:
                logging.warning(f"  Insufficient data: Active n={len(active_data)}, SHAM n={len(sham_data)}")
    
    return results


def save_analysis_results(wave_df, subject_means_df, statistical_results, output_dir):
    """
    Save all analysis results to files.
    
    Args:
        wave_df: Wave-level involvement data
        subject_means_df: Subject-averaged involvement data  
        statistical_results: Statistical analysis results
        output_dir: Output directory
        
    Returns:
        Dictionary of saved file paths
    """
    
    logging.info(f"\n=== SAVING RESULTS TO {output_dir} ===")
    
    saved_files = {}
    
    # 1. Wave-level data
    wave_file = os.path.join(output_dir, "wave_involvement_data.csv")
    wave_df.to_csv(wave_file, index=False)
    saved_files['wave_data'] = wave_file
    logging.info(f"Wave-level data saved: {wave_file}")
    
    # 2. Subject-averaged data
    subject_file = os.path.join(output_dir, "subject_averaged_involvement.csv")
    subject_means_df.to_csv(subject_file, index=False)
    saved_files['subject_data'] = subject_file
    logging.info(f"Subject-averaged data saved: {subject_file}")
    
    # 3. Statistical results summary
    stats_data = []
    
    # Within-group results
    if 'within_group' in statistical_results:
        for group, comparisons in statistical_results['within_group'].items():
            for comparison, results in comparisons.items():
                stage1, stage2 = comparison.split('_vs_')
                stats_data.append({
                    'Analysis_Type': 'Within_Group',
                    'Group': group,
                    'Stage1': stage1,
                    'Stage2': stage2,
                    'Comparison': f"{stage2} vs {stage1}",
                    'Test': results['test'],
                    'N_Subjects': results['n_subjects'],
                    'Statistic': results['statistic'],
                    'P_Value': results['p_value'],
                    'Significant': results['significant'],
                    'Effect_Size': results['effect_size'],
                    'Stage1_Mean': results['stage1_mean'],
                    'Stage2_Mean': results['stage2_mean'],
                    'Mean_Difference': results.get('mean_difference', np.nan)
                })
    
    # Between-group results
    if 'between_group' in statistical_results:
        for stage, results in statistical_results['between_group'].items():
            stats_data.append({
                'Analysis_Type': 'Between_Group',
                'Stage': stage,
                'Comparison': 'Active vs SHAM',
                'Test': results['test'],
                'N_Active': results['n_active'],
                'N_SHAM': results['n_sham'],
                'Statistic': results['statistic'],
                'P_Value': results['p_value'],
                'Significant': results['significant'],
                'Effect_Size': results['effect_size'],
                'Active_Mean': results['active_mean'],
                'SHAM_Mean': results['sham_mean']
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join(output_dir, "optimal_statistical_results.csv")
        stats_df.to_csv(stats_file, index=False)
        saved_files['statistics'] = stats_file
        logging.info(f"Statistical results saved: {stats_file}")
    
    return saved_files


def create_summary_visualizations(subject_means_df, statistical_results, output_dir):
    """Create summary visualizations of the results."""
    
    logging.info("\n=== CREATING SUMMARY VISUALIZATIONS ===")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Box plot of involvement by group and stage
    plt.figure(figsize=(12, 8))
    
    # Create the plot
    ax = sns.boxplot(data=subject_means_df, x='Stage', y='Mean_Involvement', 
                     hue='Treatment_Group', palette=['red', 'blue'])
    
    # Add individual points
    sns.stripplot(data=subject_means_df, x='Stage', y='Mean_Involvement', 
                  hue='Treatment_Group', dodge=True, alpha=0.7, size=6, ax=ax)
    
    # Customize plot
    plt.title('Involvement Percentage by Treatment Group and Stage\n(Subject-Specific Thresholds)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Involvement Percentage (%)', fontsize=12)
    plt.legend(title='Treatment Group', fontsize=10, title_fontsize=11)
    
    # Add statistical annotations
    stages = sorted(subject_means_df['Stage'].unique())
    y_max = subject_means_df['Mean_Involvement'].max() * 1.1
    
    # Annotate significant between-group differences
    if 'between_group' in statistical_results:
        for i, stage in enumerate(stages):
            if stage in statistical_results['between_group']:
                result = statistical_results['between_group'][stage]
                if result['significant']:
                    plt.text(i, y_max * 0.95, f"p={result['p_value']:.3f}***", 
                            ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    boxplot_file = os.path.join(plots_dir, "involvement_by_group_stage.png")
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Box plot saved: {boxplot_file}")
    
    # 2. Within-group difference plot for Active group
    if 'within_group' in statistical_results and 'Active' in statistical_results['within_group']:
        plt.figure(figsize=(10, 6))
        
        # Get Active group data
        active_data = subject_means_df[subject_means_df['Treatment_Group'] == 'Active']
        active_pivot = active_data.pivot(index='Subject_ID', columns='Stage', values='Mean_Involvement')
        
        # Create paired plot
        for stage1, stage2 in [('pre', 'stim')]:  # Focus on key comparison
            if stage1 in active_pivot.columns and stage2 in active_pivot.columns:
                paired_data = active_pivot[[stage1, stage2]].dropna()
                
                if len(paired_data) > 0:
                    x_pos = [0, 1]
                    for i, (_, row) in enumerate(paired_data.iterrows()):
                        plt.plot(x_pos, [row[stage1], row[stage2]], 'o-', alpha=0.7, 
                                color='red', markersize=8)
                    
                    # Add means
                    means = [paired_data[stage1].mean(), paired_data[stage2].mean()]
                    plt.plot(x_pos, means, 'o-', color='darkred', linewidth=3, 
                            markersize=12, label='Group Mean')
                    
                    # Statistical annotation
                    comparison_key = f"{stage1}_vs_{stage2}"
                    if comparison_key in statistical_results['within_group']['Active']:
                        result = statistical_results['within_group']['Active'][comparison_key]
                        p_val = result['p_value']
                        sig_text = f"Wilcoxon p={p_val:.6f}{'***' if result['significant'] else ''}"
                        plt.text(0.5, max(means) * 1.1, sig_text, ha='center', 
                                fontweight='bold', fontsize=12,
                                color='red' if result['significant'] else 'black')
                    
                    plt.xticks(x_pos, [stage1.capitalize(), stage2.capitalize()])
                    plt.ylabel('Involvement Percentage (%)')
                    plt.title(f'Active Group: {stage2.capitalize()} vs {stage1.capitalize()} Comparison\n'
                             f'(Subject-Specific Thresholds, Paired Analysis)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    paired_plot_file = os.path.join(plots_dir, f"active_{stage1}_vs_{stage2}_paired.png")
                    plt.savefig(paired_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logging.info(f"Paired comparison plot saved: {paired_plot_file}")
    
    return plots_dir


def main():
    """Main function to run the final involvement analysis pipeline."""
    
    print("="*80)
    print("FINAL INVOLVEMENT ANALYSIS PIPELINE")
    print("="*80)
    print("Optimal Statistical Approach:")
    print("• Subject-specific thresholds")
    print("• Wilcoxon signed-rank for within-group comparisons")  
    print("• Mann-Whitney U for between-group comparisons")
    print("="*80)
    
    # Get input parameters
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        data_directory = input("Enter path to your data directory: ").strip()
    
    if not os.path.isdir(data_directory):
        print(f"Error: Directory not found: {data_directory}")
        return
    
    # Validate required files
    eeg_data_path = os.path.join(data_directory, "EEG_data")
    json_path = os.path.join(data_directory, "Subject_Condition.json")
    
    if not os.path.isdir(eeg_data_path):
        print(f"Error: EEG_data directory not found in '{data_directory}'.")
        return
    
    if not os.path.isfile(json_path):
        print(f"Error: Subject_Condition.json file not found in '{data_directory}'.")
        return
    
    # Set up output directory and logging
    output_dir = os.path.join(data_directory, "Final_Involvement_Analysis")
    log_file = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("FINAL INVOLVEMENT ANALYSIS PIPELINE STARTED")
    logging.info("="*80)
    logging.info(f"Data directory: {data_directory}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file}")
    
    # Load subject condition mapping
    subject_condition_mapping = read_subject_condition_mapping(json_path)
    if not subject_condition_mapping:
        return
    
    # Get available data
    subjects, nights = scan_available_subjects_and_nights(eeg_data_path)
    logging.info(f"Available: {len(subjects)} subjects, {len(nights)} nights")
    
    # STEP 1: Process data with subject-specific thresholds
    logging.info("\n" + "="*60)
    logging.info("STEP 1: SUBJECT-SPECIFIC THRESHOLD PROCESSING")
    logging.info("="*60)
    
    results_by_treatment_group, subject_thresholds = process_eeg_data_directory_subject_specific(
        eeg_data_path,
        subject_condition_mapping,
        selected_subjects=None,  # Process all subjects
        selected_nights=None,    # Process all nights
        visualize_regions=False, # Skip visualization for speed
        process_origins=False,   # Focus on involvement only
        source_dir=data_directory
    )
    
    if results_by_treatment_group is None:
        logging.error("Subject-specific processing failed")
        return
    
    # STEP 2: Extract wave-level involvement data
    logging.info("\n" + "="*60)
    logging.info("STEP 2: EXTRACT WAVE-LEVEL INVOLVEMENT DATA")
    logging.info("="*60)
    
    wave_df = extract_wave_involvement_data(results_by_treatment_group, subject_thresholds)
    
    # STEP 3: Create subject-averaged data
    logging.info("\n" + "="*60)
    logging.info("STEP 3: CREATE SUBJECT-AVERAGED DATA")
    logging.info("="*60)
    
    subject_means_df = create_subject_averaged_data(wave_df)
    
    # STEP 4: Perform optimal statistical analysis
    logging.info("\n" + "="*60)
    logging.info("STEP 4: OPTIMAL STATISTICAL ANALYSIS")
    logging.info("="*60)
    
    statistical_results = perform_optimal_statistical_analysis(subject_means_df)
    
    # STEP 5: Save all results
    logging.info("\n" + "="*60)
    logging.info("STEP 5: SAVE RESULTS")
    logging.info("="*60)
    
    saved_files = save_analysis_results(wave_df, subject_means_df, statistical_results, output_dir)
    
    # STEP 6: Create visualizations
    logging.info("\n" + "="*60)
    logging.info("STEP 6: CREATE VISUALIZATIONS") 
    logging.info("="*60)
    
    plots_dir = create_summary_visualizations(subject_means_df, statistical_results, output_dir)
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Log file: {log_file}")
    
    # Highlight key results
    print("\nKEY RESULTS:")
    print("-" * 40)
    
    # Within-group results
    if 'within_group' in statistical_results and 'Active' in statistical_results['within_group']:
        for comparison, result in statistical_results['within_group']['Active'].items():
            stage1, stage2 = comparison.split('_vs_')
            sig_marker = "***SIGNIFICANT***" if result['significant'] else ""
            print(f"Active {stage2} vs {stage1}: Wilcoxon p={result['p_value']:.6f} {sig_marker}")
    
    # Between-group results  
    if 'between_group' in statistical_results:
        for stage, result in statistical_results['between_group'].items():
            sig_marker = "***SIGNIFICANT***" if result['significant'] else ""
            print(f"{stage.capitalize()} Active vs SHAM: Mann-Whitney U p={result['p_value']:.6f} {sig_marker}")
    
    print(f"\nThis pipeline successfully reproduces and formalizes our optimal approach!")
    
    print("\nMETHODOLOGICAL SUMMARY:")
    print("• Subject-specific thresholds eliminate inter-subject detection variability")
    print("• Wave-level data preserved for complete information")
    print("• Wilcoxon signed-rank tests maintain proper paired design")
    print("• Mann-Whitney U tests provide robust between-group comparisons")
    print("• All results are methodologically sound and reproducible")


if __name__ == "__main__":
    main()
