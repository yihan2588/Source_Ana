#!/usr/bin/env python3
"""
Create Final Clean Plots for Involvement Analysis

Simple, clean publication plots with fixed layout issues.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_analysis_data(results_dir):
    """Load the saved analysis data."""
    subject_file = os.path.join(results_dir, "subject_averaged_involvement.csv")
    wave_file = os.path.join(results_dir, "wave_involvement_data.csv")
    
    if not os.path.isfile(subject_file):
        raise FileNotFoundError(f"Subject-averaged data not found: {subject_file}")
    
    subject_df = pd.read_csv(subject_file)
    wave_df = pd.read_csv(wave_file)
    
    print(f"Loaded subject data: {len(subject_df)} records")
    print(f"Loaded wave data: {len(wave_df)} records")
    
    return subject_df, wave_df


def create_simple_boxplot(subject_df, title_suffix="", output_path=None):
    """Create simple, clean box plot like protocol-specific style."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors
    colors = {'Active': 'lightblue', 'SHAM': 'lightgreen'}
    point_colors = {'Active': 'blue', 'SHAM': 'green'}
    
    # Ensure pre-stim-post order
    subject_df['Stage'] = pd.Categorical(subject_df['Stage'], categories=['pre', 'stim', 'post'], ordered=True)
    
    # Simple seaborn plot (same as protocol style)
    sns.boxplot(data=subject_df, x='Stage', y='Mean_Involvement', 
                hue='Treatment_Group', palette=colors,
                showmeans=True, order=['pre', 'stim', 'post'],
                meanprops={'marker': 'D', 'markerfacecolor': 'white', 
                          'markeredgecolor': 'black', 'markersize': 7})
    
    sns.stripplot(data=subject_df, x='Stage', y='Mean_Involvement', 
                  hue='Treatment_Group', palette=point_colors,
                  order=['pre', 'stim', 'post'],
                  dodge=True, size=7, alpha=0.8)
    
    # Add mean values right above upper quartile line, slightly to the right
    stages = ['pre', 'stim', 'post']
    groups = ['Active', 'SHAM']
    
    for i, stage in enumerate(stages):
        for j, group in enumerate(groups):
            stage_data = subject_df[(subject_df['Stage'] == stage) & 
                                  (subject_df['Treatment_Group'] == group)]
            if len(stage_data) > 0:
                mean_val = stage_data['Mean_Involvement'].mean()
                q3_val = np.percentile(stage_data['Mean_Involvement'], 75)  # Upper quartile
                
                # Position text right above Q3 line (centered)
                x_pos = i + (j - 0.5) * 0.2  # Group offset only, no right shift
                y_pos = q3_val + 0.05  # Just above Q3
                
                plt.text(x_pos, y_pos, f'μ = {mean_val:.2f}%\n(n={len(stage_data)})', 
                        ha='center', va='bottom', fontsize=6, fontweight='bold',
                        color=point_colors[group],
                        bbox=dict(boxstyle='round,pad=0.15', 
                                 facecolor=colors[group], alpha=0.8))
    
    # Simple title like protocol style
    plt.title('Involvement - Overall', fontsize=14, fontweight='bold')
    plt.xlabel('Experimental Stage', fontsize=12)
    plt.ylabel('Involvement Percentage (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Clean legend (same as protocol style)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Active', 'SHAM'], title='Treatment Group')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved: {output_path}")


def create_simple_change_plot(changes_df, title_suffix="", output_path=None, global_y_lim=None):
    """Create simple percentage change plot with Stim-Pre left, Post-Pre right."""
    
    if len(changes_df) == 0:
        return None
    
    # Colors
    bar_colors = {'Active': 'lightblue', 'SHAM': 'lightgreen'}
    point_colors = {'Active': 'blue', 'SHAM': 'green'}
    
    # Fixed order: Stim-Pre left, Post-Pre right
    comparisons = ['Stim-Pre', 'Post-Pre']
    available_comps = [c for c in comparisons if c in changes_df['Comparison'].unique()]
    
    fig, axes = plt.subplots(1, len(available_comps), figsize=(6*len(available_comps), 8))
    if len(available_comps) == 1:
        axes = [axes]
    
    # Simple title without "overall" to avoid overlap
    fig.suptitle('Involvement Percentage Changes - Overall', 
                 fontsize=12, fontweight='bold', y=0.95)
    
    for i, comparison in enumerate(available_comps):
        ax = axes[i]
        comp_data = changes_df[changes_df['Comparison'] == comparison]
        
        # Plot each group
        for j, group in enumerate(['Active', 'SHAM']):
            group_data = comp_data[comp_data['Treatment_Group'] == group]
            if len(group_data) > 0:
                values = group_data['Percentage_Change'].values
                mean = np.mean(values)
                std = np.std(values)
                
                # Simple bar
                ax.bar(j, mean, yerr=std, color=bar_colors[group], 
                      alpha=0.8, capsize=5, ecolor='black', edgecolor='black')
                
                # Individual points
                np.random.seed(42)
                x_jitter = np.random.normal(j, 0.05, len(values))
                ax.scatter(x_jitter, values, color=point_colors[group], 
                          s=60, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Always show mean value on bars
                label_y = mean + std + 3 if mean >= 0 else mean - std - 3
                ax.text(j, label_y, f'{mean:.1f}%\n(n={len(values)})', 
                       ha='center', va='bottom' if mean >= 0 else 'top',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor=bar_colors[group], alpha=0.8))
        
        # Format subplot
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{comparison}', fontsize=12, fontweight='bold', pad=8)  # Smaller title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Active', 'SHAM'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Percentage Change (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if global_y_lim:
            ax.set_ylim(global_y_lim)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Change plot saved: {output_path}")


def determine_protocol_specific_limits(wave_df):
    """Determine y-axis limits specific to protocol data (not same as overall)."""
    
    protocols = sorted([p for p in wave_df['Protocol'].unique() if p.startswith('proto')])
    
    # Get all protocol involvement values
    all_proto_involvement = []
    all_proto_changes = []
    
    for protocol in protocols:
        protocol_data = wave_df[wave_df['Protocol'] == protocol]
        protocol_means = protocol_data.groupby(['Subject_ID', 'Treatment_Group', 'Stage']).agg({
            'Involvement_Percentage': 'mean'
        }).reset_index()
        protocol_means.columns = ['Subject_ID', 'Treatment_Group', 'Stage', 'Mean_Involvement']
        
        if len(protocol_means) >= 6:
            # Involvement values
            all_proto_involvement.extend(protocol_means['Mean_Involvement'].values)
            
            # Percentage changes
            changes = calculate_percentage_changes(protocol_means)
            if len(changes) > 0:
                all_proto_changes.extend(changes['Percentage_Change'].values)
    
    # Calculate limits
    if all_proto_involvement:
        inv_min, inv_max = np.min(all_proto_involvement), np.max(all_proto_involvement)
        inv_range = inv_max - inv_min
        proto_involvement_lim = [inv_min - inv_range*0.1, inv_max + inv_range*0.2]
    else:
        proto_involvement_lim = [0, 5]
    
    if all_proto_changes:
        change_min, change_max = np.min(all_proto_changes), np.max(all_proto_changes)
        change_range = change_max - change_min
        proto_change_lim = [change_min - change_range*0.1, change_max + change_range*0.2]
    else:
        proto_change_lim = [-50, 50]
    
    return proto_involvement_lim, proto_change_lim


def create_protocol_plots(wave_df, results_dir, proto_y_lim, proto_change_lim):
    """Create protocol-specific plots with their own consistent scaling."""
    
    protocols = sorted([p for p in wave_df['Protocol'].unique() if p.startswith('proto')])
    
    for protocol in protocols:
        protocol_data = wave_df[wave_df['Protocol'] == protocol]
        protocol_means = protocol_data.groupby(['Subject_ID', 'Treatment_Group', 'Stage']).agg({
            'Involvement_Percentage': 'mean'
        }).reset_index()
        protocol_means.columns = ['Subject_ID', 'Treatment_Group', 'Stage', 'Mean_Involvement']
        
        if len(protocol_means) < 6:
            continue
        
        # Create directory
        protocol_dir = os.path.join(results_dir, "plots", f"{protocol}_plots")
        os.makedirs(protocol_dir, exist_ok=True)
        
        # Box plot
        boxplot_path = os.path.join(protocol_dir, f"{protocol}_involvement_boxplot.png")
        create_simple_protocol_boxplot(protocol_means, protocol.upper(), 
                                     boxplot_path, proto_y_lim)
        
        # Change plot with ±100% fixed bounds
        changes = calculate_percentage_changes(protocol_means)
        if len(changes) > 0:
            change_path = os.path.join(protocol_dir, f"{protocol}_percentage_changes.png")
            create_simple_protocol_change_plot(changes, protocol.upper(),
                                             change_path)


def create_simple_protocol_boxplot(subject_df, protocol, output_path, y_lim):
    """Create simple protocol box plot with pre-stim-post order."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Active': 'lightblue', 'SHAM': 'lightgreen'}
    point_colors = {'Active': 'blue', 'SHAM': 'green'}
    
    # Ensure pre-stim-post order
    subject_df['Stage'] = pd.Categorical(subject_df['Stage'], categories=['pre', 'stim', 'post'], ordered=True)
    
    # Simple seaborn plot
    sns.boxplot(data=subject_df, x='Stage', y='Mean_Involvement', 
                hue='Treatment_Group', palette=colors,
                showmeans=True, order=['pre', 'stim', 'post'],
                meanprops={'marker': 'D', 'markerfacecolor': 'white', 
                          'markeredgecolor': 'black', 'markersize': 7})
    
    sns.stripplot(data=subject_df, x='Stage', y='Mean_Involvement', 
                  hue='Treatment_Group', palette=point_colors,
                  order=['pre', 'stim', 'post'],
                  dodge=True, size=7, alpha=0.8)
    
    # Add mean values right above upper quartile line, slightly to the right
    stages = ['pre', 'stim', 'post']
    groups = ['Active', 'SHAM']
    
    for i, stage in enumerate(stages):
        for j, group in enumerate(groups):
            stage_data = subject_df[(subject_df['Stage'] == stage) & 
                                  (subject_df['Treatment_Group'] == group)]
            if len(stage_data) > 0:
                mean_val = stage_data['Mean_Involvement'].mean()
                q3_val = np.percentile(stage_data['Mean_Involvement'], 75)  # Upper quartile
                
                # Position text right above Q3 line (centered)
                x_pos = i + (j - 0.5) * 0.2  # Group offset only, no right shift
                y_pos = q3_val + (y_lim[1] - y_lim[0]) * 0.01  # Just above Q3
                
                plt.text(x_pos, y_pos, f'μ = {mean_val:.2f}%\n(n={len(stage_data)})', 
                        ha='center', va='bottom', fontsize=6, fontweight='bold',
                        color=point_colors[group],
                        bbox=dict(boxstyle='round,pad=0.1', 
                                 facecolor=colors[group], alpha=0.8))
    
    plt.title(f'Involvement - {protocol}', fontsize=14, fontweight='bold')
    plt.xlabel('Experimental Stage', fontsize=12)
    plt.ylabel('Involvement Percentage (%)', fontsize=12)
    plt.ylim(y_lim)
    plt.grid(True, alpha=0.3)
    
    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Active', 'SHAM'], title='Treatment Group')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {protocol} box plot saved")


def create_simple_protocol_change_plot(changes_df, protocol, output_path, y_lim=None):
    """Create simple protocol percentage change plot with only mean bars (no error bars or individual points)."""
    
    if len(changes_df) == 0:
        return None
    
    bar_colors = {'Active': 'lightblue', 'SHAM': 'lightgreen'}
    
    # Fixed order: Stim-Pre left, Post-Pre right
    comparisons = ['Stim-Pre', 'Post-Pre']
    available_comps = [c for c in comparisons if c in changes_df['Comparison'].unique()]
    
    # Fixed reasonable figure size
    fig, axes = plt.subplots(1, len(available_comps), figsize=(6*len(available_comps), 5))
    if len(available_comps) == 1:
        axes = [axes]
    
    fig.suptitle(f'Percentage Changes - {protocol}', fontsize=14, fontweight='bold', y=0.95)
    
    for i, comparison in enumerate(available_comps):
        ax = axes[i]
        comp_data = changes_df[changes_df['Comparison'] == comparison]
        
        for j, group in enumerate(['Active', 'SHAM']):
            group_data = comp_data[comp_data['Treatment_Group'] == group]
            if len(group_data) > 0:
                values = group_data['Percentage_Change'].values
                mean = np.mean(values)
                
                # Simple bar - no error bars, no individual points
                ax.bar(j, mean, color=bar_colors[group], 
                      alpha=0.8, edgecolor='black')
                
                # Show mean value above bar
                label_y = mean + 5 if mean >= 0 else mean - 5
                ax.text(j, label_y, f'{mean:.1f}%\n(n={len(values)})', 
                       ha='center', va='bottom' if mean >= 0 else 'top',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor=bar_colors[group], alpha=0.8))
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{comparison}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Active', 'SHAM'], fontsize=11)
        ax.set_ylabel('Percentage Change (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Fixed ±100% bounds for protocol plots
        ax.set_ylim([-100, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {protocol} change plot saved")


def create_protocol_plots(wave_df, results_dir, proto_y_lim, proto_change_lim):
    """Create protocol-specific plots with their own consistent scaling."""
    
    protocols = sorted([p for p in wave_df['Protocol'].unique() if p.startswith('proto')])
    
    for protocol in protocols:
        protocol_data = wave_df[wave_df['Protocol'] == protocol]
        protocol_means = protocol_data.groupby(['Subject_ID', 'Treatment_Group', 'Stage']).agg({
            'Involvement_Percentage': 'mean'
        }).reset_index()
        protocol_means.columns = ['Subject_ID', 'Treatment_Group', 'Stage', 'Mean_Involvement']
        
        if len(protocol_means) < 6:
            continue
        
        # Create directory
        protocol_dir = os.path.join(results_dir, "plots", f"{protocol}_plots")
        os.makedirs(protocol_dir, exist_ok=True)
        
        # Box plot
        boxplot_path = os.path.join(protocol_dir, f"{protocol}_involvement_boxplot.png")
        create_simple_protocol_boxplot(protocol_means, protocol.upper(), 
                                     boxplot_path, proto_y_lim)
        
        # Change plot
        changes = calculate_percentage_changes(protocol_means)
        if len(changes) > 0:
            change_path = os.path.join(protocol_dir, f"{protocol}_percentage_changes.png")
            create_simple_protocol_change_plot(changes, protocol.upper(),
                                             change_path, proto_change_lim)


def calculate_percentage_changes(subject_df):
    """Calculate percentage changes for within-group comparisons."""
    
    pivot_df = subject_df.pivot_table(
        index=['Subject_ID', 'Treatment_Group'], 
        columns='Stage', 
        values='Mean_Involvement'
    ).reset_index()
    
    changes = []
    
    for _, row in pivot_df.iterrows():
        subject_id = row['Subject_ID']
        group = row['Treatment_Group']
        pre_val = row.get('pre', np.nan)
        stim_val = row.get('stim', np.nan)
        post_val = row.get('post', np.nan)
        
        if not np.isnan(pre_val) and pre_val > 0:
            if not np.isnan(stim_val):
                stim_change = ((stim_val - pre_val) / pre_val) * 100
                changes.append({
                    'Subject_ID': subject_id,
                    'Treatment_Group': group,
                    'Comparison': 'Stim-Pre',
                    'Percentage_Change': stim_change
                })
            
            if not np.isnan(post_val):
                post_change = ((post_val - pre_val) / pre_val) * 100
                changes.append({
                    'Subject_ID': subject_id,
                    'Treatment_Group': group,
                    'Comparison': 'Post-Pre',
                    'Percentage_Change': post_change
                })
    
    return pd.DataFrame(changes)


def plot_subject_line_charts(report_df, output_dir):
    """
    Create only 2 specific line charts from subject detailed report data.
    Each line represents a subject, with different colors for Active vs SHAM groups.
    
    Args:
        report_df: DataFrame with subject detailed report data
        output_dir: Output directory
    """
    
    print("\n5. Creating subject line charts...")
    
    # Color schemes for groups
    active_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(report_df[report_df['Treatment_Group'] == 'Active'])))
    sham_colors = plt.cm.Greens(np.linspace(0.3, 0.8, len(report_df[report_df['Treatment_Group'] == 'SHAM'])))
    
    protocols = [f'Proto{i}' for i in range(1, 9)]
    stages = ['Pre', 'Stim', 'Post']
    
    # 1. Protocol-Averaged Involvement Across Stages
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_cols = ['ProtoAvg_Pre', 'ProtoAvg_Stim', 'ProtoAvg_Post']
    x_positions = [0, 1, 2]
    x_labels = ['Pre', 'Stim', 'Post']
    
    # Plot Active subjects
    active_subjects = report_df[report_df['Treatment_Group'] == 'Active']
    for i, (_, subject) in enumerate(active_subjects.iterrows()):
        values = [subject[col] for col in avg_cols]
        # Filter out NaN values for plotting
        valid_data = [(x, y) for x, y in zip(x_positions, values) if not pd.isna(y)]
        if valid_data:
            x_vals, y_vals = zip(*valid_data)
            ax.plot(x_vals, y_vals, 'o-', color=active_colors[i % len(active_colors)], 
                   alpha=0.7, linewidth=2, markersize=6,
                   label='Active' if i == 0 else "")
    
    # Plot SHAM subjects
    sham_subjects = report_df[report_df['Treatment_Group'] == 'SHAM']
    for i, (_, subject) in enumerate(sham_subjects.iterrows()):
        values = [subject[col] for col in avg_cols]
        # Filter out NaN values for plotting
        valid_data = [(x, y) for x, y in zip(x_positions, values) if not pd.isna(y)]
        if valid_data:
            x_vals, y_vals = zip(*valid_data)
            ax.plot(x_vals, y_vals, 's-', color=sham_colors[i % len(sham_colors)], 
                   alpha=0.7, linewidth=2, markersize=6,
                   label='SHAM' if i == 0 else "")
    
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Involvement Percentage (%)', fontsize=12)
    ax.set_title('Subject-Level Protocol-Averaged Involvement Across Stages', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    avg_plot_path = os.path.join(output_dir, "plots", "subject_lines_protocol_averages.png")
    plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Subject line chart saved: protocol averages")
    
    # 2. Individual Continuous Sequential Charts for each subject
    # Create sequence of columns for continuous trajectory
    sequential_cols = []
    x_labels = []
    for proto in protocols:
        for stage in stages:
            sequential_cols.append(f'{proto}_{stage}')
            x_labels.append(f'{proto}-{stage}')
    
    x_positions = list(range(len(sequential_cols)))
    
    # Create directory for individual subject plots
    individual_plots_dir = os.path.join(output_dir, "plots", "individual_subject_sequential")
    os.makedirs(individual_plots_dir, exist_ok=True)
    
    # Plot each subject individually
    all_subjects = report_df.copy()
    
    for _, subject_row in all_subjects.iterrows():
        subject_id = subject_row['Subject_ID']
        treatment_group = subject_row['Treatment_Group']
        
        # Choose color based on treatment group
        if treatment_group == 'Active':
            line_color = 'blue'
            marker_style = 'o-'
        else:  # SHAM
            line_color = 'green'
            marker_style = 's-'
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Get values for this subject
        values = [subject_row[col] for col in sequential_cols]
        
        # Filter out NaN values for plotting but keep x positions aligned
        x_vals, y_vals = [], []
        for x, y in zip(x_positions, values):
            if not pd.isna(y):
                x_vals.append(x)
                y_vals.append(y)
        
        if x_vals and y_vals:
            ax.plot(x_vals, y_vals, marker_style, color=line_color, 
                   linewidth=2, markersize=6, alpha=0.8)
            
            # Add vertical lines to separate protocols
            for i in range(1, len(protocols)):
                protocol_boundary = i * 3 - 0.5  # Between Proto(i-1)-Post and Proto(i)-Pre
                ax.axvline(x=protocol_boundary, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Protocol Sequence (Pre → Stim → Post)', fontsize=12)
            ax.set_ylabel('Involvement Percentage (%)', fontsize=12)
            ax.set_title(f'Subject {subject_id} ({treatment_group}) - Continuous Sequential Involvement', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save individual plot
            individual_plot_path = os.path.join(individual_plots_dir, f"subject_{subject_id}_sequential.png")
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Close empty figure if no valid data
            plt.close()
    
    print(f"  Individual subject sequential charts saved: {len(all_subjects)} subjects")


def create_subject_detailed_report(wave_df, subject_df, output_dir):
    """
    Create a detailed subject-level CSV report with protocol-specific and protocol-averaged data.
    
    Args:
        wave_df: DataFrame with wave-level data
        subject_df: DataFrame with subject-averaged data
        output_dir: Output directory
        
    Returns:
        Path to the saved CSV file
    """
    
    print("\n4. Creating detailed subject report...")
    
    # Get unique subjects and treatment groups
    subjects = wave_df['Subject_ID'].unique()
    protocols = [f'proto{i}' for i in range(1, 9)]  # proto1-8
    stages = ['pre', 'stim', 'post']
    
    # Initialize the report data
    report_data = []
    
    for subject in subjects:
        subject_wave_data = wave_df[wave_df['Subject_ID'] == subject]
        subject_avg_data = subject_df[subject_df['Subject_ID'] == subject]
        
        if len(subject_avg_data) == 0:
            continue
            
        # Get treatment group
        treatment_group = subject_avg_data['Treatment_Group'].iloc[0]
        
        # Initialize row data
        row_data = {
            'Subject_ID': subject,
            'Treatment_Group': treatment_group
        }
        
        # Protocol-specific data
        proto_averages = {'pre': [], 'stim': [], 'post': []}
        
        for protocol in protocols:
            protocol_data = subject_wave_data[subject_wave_data['Protocol'] == protocol]
            
            # Calculate protocol means for each stage
            protocol_means = {}
            for stage in stages:
                stage_data = protocol_data[protocol_data['Stage'] == stage]
                if len(stage_data) > 0:
                    mean_val = stage_data['Involvement_Percentage'].mean()
                    protocol_means[stage] = mean_val
                    proto_averages[stage].append(mean_val)
                    row_data[f'{protocol.capitalize()}_{stage.capitalize()}'] = mean_val
                else:
                    protocol_means[stage] = np.nan
                    row_data[f'{protocol.capitalize()}_{stage.capitalize()}'] = np.nan
            
            # Calculate percentage changes for this protocol
            if not np.isnan(protocol_means.get('pre', np.nan)) and protocol_means['pre'] > 0:
                # Stim-Pre change
                if not np.isnan(protocol_means.get('stim', np.nan)):
                    stim_change = ((protocol_means['stim'] - protocol_means['pre']) / protocol_means['pre']) * 100
                    row_data[f'{protocol.capitalize()}_Stim_Pre_Change%'] = stim_change
                else:
                    row_data[f'{protocol.capitalize()}_Stim_Pre_Change%'] = np.nan
                
                # Post-Pre change
                if not np.isnan(protocol_means.get('post', np.nan)):
                    post_change = ((protocol_means['post'] - protocol_means['pre']) / protocol_means['pre']) * 100
                    row_data[f'{protocol.capitalize()}_Post_Pre_Change%'] = post_change
                else:
                    row_data[f'{protocol.capitalize()}_Post_Pre_Change%'] = np.nan
            else:
                row_data[f'{protocol.capitalize()}_Stim_Pre_Change%'] = np.nan
                row_data[f'{protocol.capitalize()}_Post_Pre_Change%'] = np.nan
        
        # Calculate protocol averages (mean across proto1-8)
        for stage in stages:
            if proto_averages[stage]:
                avg_val = np.mean(proto_averages[stage])
                row_data[f'ProtoAvg_{stage.capitalize()}'] = avg_val
            else:
                row_data[f'ProtoAvg_{stage.capitalize()}'] = np.nan
        
        # Calculate percentage changes for protocol averages
        pre_avg = row_data.get('ProtoAvg_Pre', np.nan)
        stim_avg = row_data.get('ProtoAvg_Stim', np.nan)
        post_avg = row_data.get('ProtoAvg_Post', np.nan)
        
        if not np.isnan(pre_avg) and pre_avg > 0:
            if not np.isnan(stim_avg):
                row_data['ProtoAvg_Stim_Pre_Change%'] = ((stim_avg - pre_avg) / pre_avg) * 100
            else:
                row_data['ProtoAvg_Stim_Pre_Change%'] = np.nan
                
            if not np.isnan(post_avg):
                row_data['ProtoAvg_Post_Pre_Change%'] = ((post_avg - pre_avg) / pre_avg) * 100
            else:
                row_data['ProtoAvg_Post_Pre_Change%'] = np.nan
        else:
            row_data['ProtoAvg_Stim_Pre_Change%'] = np.nan
            row_data['ProtoAvg_Post_Pre_Change%'] = np.nan
        
        report_data.append(row_data)
    
    # Create DataFrame and save
    report_df = pd.DataFrame(report_data)
    
    # Reorder columns for better readability
    base_cols = ['Subject_ID', 'Treatment_Group']
    proto_cols = []
    
    for protocol in protocols:
        proto_cols.extend([
            f'{protocol.capitalize()}_Pre',
            f'{protocol.capitalize()}_Stim', 
            f'{protocol.capitalize()}_Post',
            f'{protocol.capitalize()}_Stim_Pre_Change%',
            f'{protocol.capitalize()}_Post_Pre_Change%'
        ])
    
    avg_cols = [
        'ProtoAvg_Pre', 'ProtoAvg_Stim', 'ProtoAvg_Post',
        'ProtoAvg_Stim_Pre_Change%', 'ProtoAvg_Post_Pre_Change%'
    ]
    
    # Reorder columns
    ordered_cols = base_cols + proto_cols + avg_cols
    existing_cols = [col for col in ordered_cols if col in report_df.columns]
    report_df = report_df[existing_cols]
    
    # Save to CSV
    report_path = os.path.join(output_dir, "subject_detailed_report.csv")
    report_df.to_csv(report_path, index=False)
    
    print(f"  Detailed subject report saved: {report_path}")
    print(f"  Report contains {len(report_df)} subjects with {len(report_df.columns)} columns")
    
    return report_path


def main():
    """Main function to create clean, simple plots."""
    
    print("="*80)
    print("CREATING CLEAN PUBLICATION PLOTS")
    print("="*80)
    
    # Get directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = input("Enter path to Final_Involvement_Analysis results directory: ").strip()
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return
    
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    try:
        subject_df, wave_df = load_analysis_data(results_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nCreating plots...")
    
    # Determine protocol-specific limits (NOT same as overall)
    proto_y_lim, proto_change_lim = determine_protocol_specific_limits(wave_df)
    
    # Overall limits (separate from protocol)
    all_involvement = subject_df['Mean_Involvement'].values
    overall_y_lim = [np.min(all_involvement) - 0.2, np.max(all_involvement) + 0.3]
    
    all_changes = calculate_percentage_changes(subject_df)
    if len(all_changes) > 0:
        change_vals = all_changes['Percentage_Change'].values
        overall_change_lim = [np.min(change_vals) - 10, np.max(change_vals) + 15]
    else:
        overall_change_lim = [-50, 50]
    
    # Protocol change limits fixed to ±100%
    proto_change_lim = [-100, 100]
    
    print(f"  Overall y-limits: {overall_y_lim[0]:.1f} to {overall_y_lim[1]:.1f}")
    print(f"  Protocol y-limits: {proto_y_lim[0]:.1f} to {proto_y_lim[1]:.1f}")
    print(f"  Overall change limits: {overall_change_lim[0]:.1f} to {overall_change_lim[1]:.1f}")
    print(f"  Protocol change limits: ±100% (fixed)")
    
    # 1. Overall box plot
    print("\n1. Creating overall box plot...")
    overall_boxplot_path = os.path.join(plots_dir, "overall_involvement_boxplot.png")
    create_simple_boxplot(subject_df, "", overall_boxplot_path)  # No suffix
    
    # 2. Overall change plot
    print("2. Creating overall change plot...")
    if len(all_changes) > 0:
        overall_change_path = os.path.join(plots_dir, "overall_percentage_changes.png")
        create_simple_change_plot(all_changes, "", overall_change_path, overall_change_lim)  # No suffix
    
    # 3. Protocol-specific plots
    print("3. Creating protocol-specific plots...")
    create_protocol_plots(wave_df, results_dir, proto_y_lim, proto_change_lim)
    
    # 4. Create detailed subject report
    report_path = create_subject_detailed_report(wave_df, subject_df, results_dir)
    
    # 5. Create subject line charts from the detailed report
    report_df = pd.read_csv(report_path)
    plot_subject_line_charts(report_df, results_dir)
    
    print(f"\nALL PLOTS COMPLETED")
    print(f"Results in: {plots_dir}")
    
    # Count files
    plot_files = list(Path(plots_dir).glob("**/*.png"))
    print(f"Total plot files: {len(plot_files)}")
    
    # Count CSV files including the new report
    csv_files = list(Path(results_dir).glob("*.csv"))
    print(f"Total CSV files: {len(csv_files)}")


if __name__ == "__main__":
    main()
