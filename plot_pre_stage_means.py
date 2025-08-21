#!/usr/bin/env python3
"""
Standalone script to plot pre stage mean involvement for proto1-8 comparing Active vs SHAM groups.
Uses sample_involvement_data.csv file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pre_stage_means():
    """
    Plot pre stage mean involvement for protocols 1-8 comparing Active vs SHAM groups.
    """
    # Load the data
    df = pd.read_csv('sample_involvement_data.csv')
    
    # Filter for pre stage only
    pre_data = df[df['Stage'] == 'pre'].copy()
    
    # Filter for proto1-8 (should already be all present)
    protocols = [f'proto{i}' for i in range(1, 9)]
    pre_data = pre_data[pre_data['Protocol'].isin(protocols)]
    
    # Sort by protocol to ensure correct order
    pre_data['Protocol_Num'] = pre_data['Protocol'].str.extract('(\d+)').astype(int)
    pre_data = pre_data.sort_values('Protocol_Num')
    
    # Separate Active and SHAM data
    active_data = pre_data[pre_data['Treatment_Group'] == 'Active']
    sham_data = pre_data[pre_data['Treatment_Group'] == 'SHAM']
    
    # Create the plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Set up x positions for bars
    x_pos = np.arange(len(protocols))
    bar_width = 0.6
    
    # Determine global y-axis limits for consistent scaling
    all_means = list(active_data['Mean_Involvement']) + list(sham_data['Mean_Involvement'])
    all_stds = list(active_data['Std_Involvement']) + list(sham_data['Std_Involvement'])
    max_val = max([m + s for m, s in zip(all_means, all_stds)]) * 1.1
    
    # Left subplot: Active group (lightblue)
    active_bars = ax1.bar(x_pos, active_data['Mean_Involvement'], 
                         bar_width, color='lightblue', alpha=0.8, 
                         edgecolor='black', linewidth=1,
                         yerr=active_data['Std_Involvement'], capsize=5)
    
    # Add value labels on Active bars
    for i, active_mean in enumerate(active_data['Mean_Involvement']):
        ax1.text(x_pos[i], active_mean + active_data.iloc[i]['Std_Involvement'] + 0.1, 
                f'{active_mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Customize left subplot (Active)
    ax1.set_xlabel('Protocol', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Involvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Active Group\nPre Stage Mean Involvement', 
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Proto{i}' for i in range(1, 9)], fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max_val)
    
    # Add sample size to Active subplot
    active_n = active_data['Count'].iloc[0] if not active_data.empty else 0
    ax1.text(0.02, 0.98, f'n={active_n}', transform=ax1.transAxes, 
             fontsize=11, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    # Right subplot: SHAM group (lightgreen)
    sham_bars = ax2.bar(x_pos, sham_data['Mean_Involvement'], 
                       bar_width, color='lightgreen', alpha=0.8, 
                       edgecolor='black', linewidth=1,
                       yerr=sham_data['Std_Involvement'], capsize=5)
    
    # Add value labels on SHAM bars
    for i, sham_mean in enumerate(sham_data['Mean_Involvement']):
        ax2.text(x_pos[i], sham_mean + sham_data.iloc[i]['Std_Involvement'] + 0.1, 
                f'{sham_mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Customize right subplot (SHAM)
    ax2.set_xlabel('Protocol', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Involvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('SHAM Group\nPre Stage Mean Involvement', 
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Proto{i}' for i in range(1, 9)], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max_val)
    
    # Add sample size to SHAM subplot
    sham_n = sham_data['Count'].iloc[0] if not sham_data.empty else 0
    ax2.text(0.02, 0.98, f'n={sham_n}', transform=ax2.transAxes, 
             fontsize=11, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    # Add overall title
    fig.suptitle('Pre Stage Mean Involvement by Protocol: Active vs SHAM Groups', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save the plot
    plt.savefig('pre_stage_means_proto1-8.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'pre_stage_means_proto1-8.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("Pre Stage Mean Involvement (%) by Protocol:")
    print("\nActive Group:")
    for _, row in active_data.iterrows():
        print(f"  {row['Protocol']}: {row['Mean_Involvement']:.2f} ± {row['Std_Involvement']:.2f} (n={row['Count']})")
    
    print("\nSHAM Group:")
    for _, row in sham_data.iterrows():
        print(f"  {row['Protocol']}: {row['Mean_Involvement']:.2f} ± {row['Std_Involvement']:.2f} (n={row['Count']})")
    
    # Calculate overall means
    active_overall = np.mean(active_data['Mean_Involvement'])
    sham_overall = np.mean(sham_data['Mean_Involvement'])
    
    print(f"\nOverall Mean Across All Protocols:")
    print(f"  Active: {active_overall:.2f}%")
    print(f"  SHAM: {sham_overall:.2f}%")
    print(f"  Difference (Active - SHAM): {active_overall - sham_overall:.2f}%")

if __name__ == "__main__":
    plot_pre_stage_means()
