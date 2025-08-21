#!/usr/bin/env python3
"""
Standalone script to plot percentage changes in involvement data.
Creates plots for each protocol showing Stim-Pre and Post-Pre percentage changes
with mean and 1 std error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def calculate_percentage_changes(df):
    """
    Calculate percentage changes (Stim-Pre and Post-Pre) from involvement data.
    
    Args:
        df: DataFrame with Protocol, Treatment_Group, Stage, Mean_Involvement columns
        
    Returns:
        DataFrame with percentage change data
    """
    percentage_changes = []
    
    protocols = df['Protocol'].unique()
    treatment_groups = df['Treatment_Group'].unique()
    
    for protocol in protocols:
        protocol_data = df[df['Protocol'] == protocol]
        
        for group in treatment_groups:
            group_data = protocol_data[protocol_data['Treatment_Group'] == group]
            
            # Get values for each stage
            pre_data = group_data[group_data['Stage'] == 'pre']
            stim_data = group_data[group_data['Stage'] == 'stim']
            post_data = group_data[group_data['Stage'] == 'post']
            
            if not pre_data.empty:
                pre_mean = pre_data['Mean_Involvement'].iloc[0]
                pre_std = pre_data['Std_Involvement'].iloc[0]
                pre_count = pre_data['Count'].iloc[0]
                
                # Calculate Stim-Pre percentage change
                if not stim_data.empty and pre_mean > 0:
                    stim_mean = stim_data['Mean_Involvement'].iloc[0]
                    stim_std = stim_data['Std_Involvement'].iloc[0]
                    stim_count = stim_data['Count'].iloc[0]
                    
                    # Percentage change calculation: ((stim - pre) / pre) * 100
                    pct_change_mean = ((stim_mean - pre_mean) / pre_mean) * 100
                    
                    # Error propagation for percentage change
                    # For f = ((B-A)/A)*100, error = 100 * sqrt((sB/A)^2 + ((B-A)*sA/A^2)^2)
                    if pre_mean > 0:
                        pct_change_std = 100 * np.sqrt((stim_std/pre_mean)**2 + ((stim_mean - pre_mean)*pre_std/pre_mean**2)**2)
                    else:
                        pct_change_std = 0
                    
                    percentage_changes.append({
                        'Protocol': protocol,
                        'Treatment_Group': group,
                        'Comparison': 'Stim-Pre',
                        'Percentage_Change_Mean': pct_change_mean,
                        'Percentage_Change_Std': pct_change_std,
                        'Count': min(stim_count, pre_count)
                    })
                
                # Calculate Post-Pre percentage change
                if not post_data.empty and pre_mean > 0:
                    post_mean = post_data['Mean_Involvement'].iloc[0]
                    post_std = post_data['Std_Involvement'].iloc[0]
                    post_count = post_data['Count'].iloc[0]
                    
                    # Percentage change calculation: ((post - pre) / pre) * 100
                    pct_change_mean = ((post_mean - pre_mean) / pre_mean) * 100
                    
                    # Error propagation for percentage change
                    if pre_mean > 0:
                        pct_change_std = 100 * np.sqrt((post_std/pre_mean)**2 + ((post_mean - pre_mean)*pre_std/pre_mean**2)**2)
                    else:
                        pct_change_std = 0
                    
                    percentage_changes.append({
                        'Protocol': protocol,
                        'Treatment_Group': group,
                        'Comparison': 'Post-Pre',
                        'Percentage_Change_Mean': pct_change_mean,
                        'Percentage_Change_Std': pct_change_std,
                        'Count': min(post_count, pre_count)
                    })
    
    return pd.DataFrame(percentage_changes)

def plot_percentage_changes_by_protocol(pct_df, output_dir="percentage_change_plots"):
    """
    Create individual plots for each protocol showing percentage changes.
    
    Args:
        pct_df: DataFrame with percentage change data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    protocols = sorted(pct_df['Protocol'].unique())
    treatment_groups = sorted(pct_df['Treatment_Group'].unique())
    comparisons = ['Stim-Pre', 'Post-Pre']  # Ensure this order
    
    print(f"Found protocols: {protocols}")
    print(f"Treatment groups: {treatment_groups}")
    
    # Use hardcoded global bounds for consistent scaling
    global_min = -100
    global_max = 100
    
    print(f"Global scale: {global_min:.1f} to {global_max:.1f}")
    
    # Create plot for each protocol
    for protocol in protocols:
        protocol_data = pct_df[pct_df['Protocol'] == protocol]
        
        if protocol_data.empty:
            print(f"No percentage change data for {protocol}, skipping...")
            continue
        
        # Create figure with Stim-Pre on left, Post-Pre on right
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        
        for i, comparison in enumerate(comparisons):
            ax = axes[i]
            comp_data = protocol_data[protocol_data['Comparison'] == comparison]
            
            if comp_data.empty:
                ax.text(0.5, 0.5, f'No data for {comparison}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{comparison} Change', fontsize=14, fontweight='bold')
                continue
            
            # Set up bar positions
            x_pos = np.arange(len(treatment_groups))
            bar_width = 0.6
            
            means = []
            stds = []
            counts = []
            
            for group in treatment_groups:
                group_data = comp_data[comp_data['Treatment_Group'] == group]
                if not group_data.empty:
                    means.append(group_data['Percentage_Change_Mean'].iloc[0])
                    stds.append(group_data['Percentage_Change_Std'].iloc[0])
                    counts.append(group_data['Count'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
                    counts.append(0)
            
            # Choose colors
            colors = ['lightblue' if group.lower() == 'active' else 'lightgreen' for group in treatment_groups]
            
            # Create bars without error bars
            bars = ax.bar(x_pos, means, bar_width,
                         color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for j, (mean, std, count) in enumerate(zip(means, stds, counts)):
                if count > 0:
                    label_y = mean + (global_max - global_min) * 0.02 if mean >= 0 else mean - (global_max - global_min) * 0.02
                    ax.text(j, label_y, f'{mean:.1f}%\n(n={count})',
                           ha='center', va='bottom' if mean >= 0 else 'top', fontsize=10)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Customize subplot
            ax.set_title(f'{comparison} Change', fontsize=14, fontweight='bold')
            ax.set_xlabel('Treatment Group', fontsize=12)
            ax.set_ylabel('Percentage Change (%)', fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(treatment_groups)
            ax.set_ylim(global_min, global_max)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{protocol.upper()} - Involvement Percentage Changes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(output_dir, f"{protocol}_percentage_changes.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved percentage change plot for {protocol} as '{filename}'")
        
        plt.close()

def create_summary_percentage_plot(pct_df, output_dir="percentage_change_plots"):
    """Create a summary plot showing all protocols together."""
    protocols = sorted(pct_df['Protocol'].unique())
    treatment_groups = sorted(pct_df['Treatment_Group'].unique())
    comparisons = ['Stim-Pre', 'Post-Pre']
    
    # Use hardcoded global bounds for consistent scaling
    global_min = -100
    global_max = 100
    
    # Create subplot grid: protocols × comparisons
    n_protocols = len(protocols)
    n_comparisons = len(comparisons)
    
    fig, axes = plt.subplots(n_protocols, n_comparisons, 
                           figsize=(6*n_comparisons, 4*n_protocols))
    
    # Handle single protocol case
    if n_protocols == 1:
        axes = [axes] if n_comparisons == 1 else axes
    elif n_comparisons == 1:
        axes = [[ax] for ax in axes]
    
    for i, protocol in enumerate(protocols):
        protocol_data = pct_df[pct_df['Protocol'] == protocol]
        
        for j, comparison in enumerate(comparisons):
            if n_protocols == 1 and n_comparisons == 1:
                ax = axes
            elif n_protocols == 1:
                ax = axes[j]
            elif n_comparisons == 1:
                ax = axes[i][0]
            else:
                ax = axes[i][j]
            
            comp_data = protocol_data[protocol_data['Comparison'] == comparison]
            
            if comp_data.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                if i == 0:  # Top row
                    ax.set_title(f'{comparison}', fontsize=12, fontweight='bold')
                if j == 0:  # Left column
                    ax.set_ylabel(f'{protocol.upper()}\nPercentage Change (%)', fontsize=11)
                continue
            
            # Set up bar positions
            x_pos = np.arange(len(treatment_groups))
            bar_width = 0.6
            
            means = []
            stds = []
            
            for group in treatment_groups:
                group_data = comp_data[comp_data['Treatment_Group'] == group]
                if not group_data.empty:
                    means.append(group_data['Percentage_Change_Mean'].iloc[0])
                    stds.append(group_data['Percentage_Change_Std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            colors = ['lightblue' if group.lower() == 'active' else 'lightgreen' for group in treatment_groups]
            
            ax.bar(x_pos, means, bar_width,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Set titles and labels
            if i == 0:  # Top row
                ax.set_title(f'{comparison}', fontsize=12, fontweight='bold')
            if j == 0:  # Left column
                ax.set_ylabel(f'{protocol.upper()}\nPercentage Change (%)', fontsize=11)
            if i == n_protocols - 1:  # Bottom row
                ax.set_xlabel('Treatment Group', fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(treatment_groups, fontsize=10)
            else:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([])
            
            ax.set_ylim(global_min, global_max)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Protocol-Specific Involvement Percentage Changes\n(Consistent Scale)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    filename = os.path.join(output_dir, "all_protocols_percentage_changes_summary.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary percentage changes plot as '{filename}'")
    
    plt.close()

def main():
    """Main function to run the percentage change plotting script."""
    # Look for CSV file in current directory
    csv_files = list(Path('.').glob('*involvement*.csv'))
    if not csv_files:
        csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("No CSV files found in current directory.")
        print("Please ensure your data file is in the current directory.")
        return
    
    # Use the first CSV file found, or let user specify
    if len(csv_files) == 1:
        csv_file = csv_files[0]
        print(f"Using CSV file: {csv_file}")
    else:
        print("Multiple CSV files found:")
        for i, f in enumerate(csv_files):
            print(f"  {i+1}. {f}")
        
        try:
            choice = input("Enter file number to use (or press Enter for first file): ").strip()
            if choice:
                csv_file = csv_files[int(choice) - 1]
            else:
                csv_file = csv_files[0]
        except (ValueError, IndexError):
            csv_file = csv_files[0]
        
        print(f"Using CSV file: {csv_file}")
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check required columns
    required_columns = ['Protocol', 'Treatment_Group', 'Stage', 'Mean_Involvement', 'Std_Involvement', 'Count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Calculate percentage changes
    print("\nCalculating percentage changes...")
    pct_df = calculate_percentage_changes(df)
    
    if pct_df.empty:
        print("No percentage changes could be calculated. Check your data.")
        return
    
    print(f"Calculated {len(pct_df)} percentage change entries")
    
    # Create output directory
    output_dir = "percentage_change_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating individual protocol percentage change plots...")
    plot_percentage_changes_by_protocol(pct_df, output_dir)
    
    print("\nGenerating summary percentage change plot...")
    create_summary_percentage_plot(pct_df, output_dir)
    
    print(f"\nAll percentage change plots saved in '{output_dir}' directory!")
    print("Individual plots: [protocol]_percentage_changes.png")
    print("Summary plot: all_protocols_percentage_changes_summary.png")

if __name__ == "__main__":
    main()
