#!/usr/bin/env python3
"""
Standalone script to plot protocol-specific involvement data with error bars.
Creates individual plots for each protocol showing mean involvement with 1 std error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_data(csv_file):
    """Load the protocol-specific data from CSV file."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with {len(df)} rows and columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_protocol_specific_data(df, output_dir="protocol_plots"):
    """
    Create individual plots for each protocol with mean and error bars.
    
    Args:
        df: DataFrame with protocol-specific data
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique protocols
    protocols = sorted(df['Protocol'].unique())
    print(f"Found protocols: {protocols}")
    
    # Get unique stages and treatment groups
    stages = sorted(df['Stage'].unique())
    treatment_groups = sorted(df['Treatment_Group'].unique())
    
    print(f"Stages: {stages}")
    print(f"Treatment groups: {treatment_groups}")
    
    # Calculate global scale for consistent y-axis across all protocols
    global_min = -100
    global_max = 100
    
    print(f"Global scale: {global_min:.1f} to {global_max:.1f}")
    
    # Create plot for each protocol
    for protocol in protocols:
        protocol_data = df[df['Protocol'] == protocol]
        
        if protocol_data.empty:
            print(f"No data for {protocol}, skipping...")
            continue
            
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Set up positions for bars
        x_pos = np.arange(len(stages))
        bar_width = 0.35
        
        # Plot data for each treatment group
        for i, group in enumerate(treatment_groups):
            group_data = protocol_data[protocol_data['Treatment_Group'] == group]
            
            means = []
            stds = []
            counts = []
            
            # Get data for each stage
            for stage in stages:
                stage_data = group_data[group_data['Stage'] == stage]
                if not stage_data.empty:
                    means.append(stage_data['Mean_Involvement'].iloc[0])
                    stds.append(stage_data['Std_Involvement'].iloc[0])
                    counts.append(stage_data['Count'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
                    counts.append(0)
            
            # Choose colors
            color = 'lightblue' if group.lower() == 'active' else 'lightgreen'
            
            # Create bars with error bars
            bars = ax.bar(x_pos + i * bar_width, means, bar_width, 
                         yerr=stds, capsize=5, label=group,
                         color=color, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for j, (mean, std, count) in enumerate(zip(means, stds, counts)):
                if count > 0:  # Only show label if we have data
                    ax.text(j + i * bar_width, mean + std + (global_max - global_min) * 0.02,
                           f'{mean:.1f}%\n(n={count})',
                           ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Mean Involvement (%)', fontsize=12)
        ax.set_title(f'{protocol.upper()} - Mean Involvement by Stage and Treatment Group', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + bar_width/2)
        ax.set_xticklabels([s.capitalize() for s in stages])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(global_min, global_max)  # Consistent scale across all protocols
        
        # Add horizontal line at zero if needed
        if global_min < 0:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(output_dir, f"{protocol}_involvement_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {protocol} as '{filename}'")
        
        plt.close()

def create_summary_plot(df, output_dir="protocol_plots"):
    """Create a summary plot showing all protocols together."""
    protocols = sorted(df['Protocol'].unique())
    stages = sorted(df['Stage'].unique())
    treatment_groups = sorted(df['Treatment_Group'].unique())
    
    # Calculate global scale
    all_means = df['Mean_Involvement'].values
    all_stds = df['Std_Involvement'].values
    global_min = max(0, (all_means - all_stds).min() * 0.9)
    global_max = (all_means + all_stds).max() * 1.1
    
    # Create subplot grid
    n_protocols = len(protocols)
    n_cols = 3  # 3 protocols per row
    n_rows = (n_protocols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, protocol in enumerate(protocols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        protocol_data = df[df['Protocol'] == protocol]
        
        # Set up positions for bars
        x_pos = np.arange(len(stages))
        bar_width = 0.35
        
        # Plot data for each treatment group
        for i, group in enumerate(treatment_groups):
            group_data = protocol_data[protocol_data['Treatment_Group'] == group]
            
            means = []
            stds = []
            
            # Get data for each stage
            for stage in stages:
                stage_data = group_data[group_data['Stage'] == stage]
                if not stage_data.empty:
                    means.append(stage_data['Mean_Involvement'].iloc[0])
                    stds.append(stage_data['Std_Involvement'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Choose colors
            color = 'lightblue' if group.lower() == 'active' else 'lightgreen'
            
            # Create bars with error bars
            ax.bar(x_pos + i * bar_width, means, bar_width, 
                  yerr=stds, capsize=3, label=group if idx == 0 else "",
                  color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize subplot
        ax.set_title(f'{protocol.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos + bar_width/2)
        ax.set_xticklabels([s.capitalize() for s in stages], fontsize=10)
        ax.set_ylim(global_min, global_max)
        ax.grid(True, alpha=0.3)
        
        # Only add labels to leftmost and bottom subplots
        if col == 0:
            ax.set_ylabel('Mean Involvement (%)', fontsize=11)
        if row == n_rows - 1:
            ax.set_xlabel('Stage', fontsize=11)
    
    # Hide empty subplots
    for idx in range(n_protocols, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1:
            axes[row][col].set_visible(False)
        else:
            if n_cols > 1:
                axes[col].set_visible(False)
    
    # Add legend
    if n_protocols > 0:
        if n_rows > 1:
            axes[0][0].legend(loc='upper right')
        else:
            axes[0].legend(loc='upper right')
    
    plt.suptitle('Protocol-Specific Involvement Comparison (All Protocols)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    filename = os.path.join(output_dir, "all_protocols_summary.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot as '{filename}'")
    
    plt.close()

def main():
    """Main function to run the plotting script."""
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
    df = load_data(csv_file)
    if df is None:
        return
    
    # Check required columns
    required_columns = ['Protocol', 'Treatment_Group', 'Stage', 'Mean_Involvement', 'Std_Involvement', 'Count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create output directory
    output_dir = "protocol_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating individual protocol plots...")
    plot_protocol_specific_data(df, output_dir)
    
    print("\nGenerating summary plot...")
    create_summary_plot(df, output_dir)
    
    print(f"\nAll plots saved in '{output_dir}' directory!")
    print("Individual plots: [protocol]_involvement_comparison.png")
    print("Summary plot: all_protocols_summary.png")

if __name__ == "__main__":
    main()
