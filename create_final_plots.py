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
    """Create simple protocol percentage change plot with fixed ±100% bounds."""
    
    if len(changes_df) == 0:
        return None
    
    bar_colors = {'Active': 'lightblue', 'SHAM': 'lightgreen'}
    point_colors = {'Active': 'blue', 'SHAM': 'green'}
    
    # Fixed order: Stim-Pre left, Post-Pre right
    comparisons = ['Stim-Pre', 'Post-Pre']
    available_comps = [c for c in comparisons if c in changes_df['Comparison'].unique()]
    
    fig, axes = plt.subplots(1, len(available_comps), figsize=(5*len(available_comps), 6))
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
                std = np.std(values)
                
                # Bar
                ax.bar(j, mean, yerr=std, color=bar_colors[group], 
                      alpha=0.8, capsize=5, ecolor='black', edgecolor='black')
                
                # Points
                np.random.seed(42)
                x_jitter = np.random.normal(j, 0.05, len(values))
                ax.scatter(x_jitter, values, color=point_colors[group], 
                          s=50, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Always show mean value prominently
                label_y = mean + std + 8 if mean >= 0 else mean - std - 8
                ax.text(j, label_y, f'{mean:.1f}%\n(n={len(values)})', 
                       ha='center', va='bottom' if mean >= 0 else 'top',
                       fontsize=11, fontweight='bold',
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
    
    print(f"\nALL PLOTS COMPLETED")
    print(f"Results in: {plots_dir}")
    
    # Count files
    plot_files = list(Path(plots_dir).glob("**/*.png"))
    print(f"Total plot files: {len(plot_files)}")


if __name__ == "__main__":
    main()
