import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import random
import logging # Added
from collections import Counter
from scipy.signal import find_peaks
from utils import extract_region_name

# Removed visualize_results function as requested

def visualize_treatment_comparison(treatment_comparison_results):
    """
    Create visualizations comparing treatment groups (Active vs. SHAM).
    Currently minimal since the detailed plots are in the new overall functions.
    """
    pass

def create_involvement_boxplot(data, labels, title, filename, colors=None):
    """
    Create a boxplot for involvement data.
    """
    if not data:
        return

    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data, labels=labels, patch_artist=True)

    if colors:
        for i, label in enumerate(labels):
            stage = label.split('\n')[1] if '\n' in label else label
            if stage in colors:
                bp['boxes'][i].set_facecolor(colors[stage])

    plt.title(title)
    plt.ylabel('Involvement (%)')
    plt.xlabel('Stage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Saved involvement boxplot as '{filename}'")
    plt.close()


def create_involvement_summary(all_results, protocols):
    """
    Create a summary bar chart of involvement across stages for each protocol.
    """
    protocols_list = []
    stages_list = []
    means = []
    errors = []
    
    # Get all stages across all protocols
    all_stages = set()
    for protocol in protocols:
        all_stages.update(all_results[protocol].keys())
    
    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order

    for protocol in protocols:
        for stage in stages:
            if stage in all_results[protocol] and all_results[protocol][stage]:
                protocols_list.append(protocol)
                stages_list.append(stage)
                data = [res['involvement_percentage'] for res in all_results[protocol][stage]]
                means.append(np.mean(data) if data else 0)
                errors.append(np.std(data) if len(data) > 1 else 0)

    summary_df = pd.DataFrame({
        'Protocol': protocols_list,
        'Stage': stages_list,
        'Mean': means,
        'Error': errors
    })

    if not summary_df.empty:
        plt.figure(figsize=(12, 6))
        stage_colors = {'pre': 'skyblue', 'early': 'lightgreen', 'late': 'salmon', 'post': 'gold', 'stim': 'purple'}

        for protocol in protocols:
            proto_data = summary_df[summary_df['Protocol'] == protocol]
            if not proto_data.empty:
                for _, row in proto_data.iterrows():
                    plt.bar(row['Protocol'] + '_' + row['Stage'],
                            row['Mean'],
                            yerr=row['Error'],
                            color=stage_colors[row['Stage']],
                            label=row['Stage']
                            if row['Stage'] not in plt.gca().get_legend_handles_labels()[1] else '')

        plt.title('Mean Involvement Percentage by Protocol and Stage')
        plt.ylabel('Involvement (%)')
        plt.xlabel('Protocol and Stage')
        plt.xticks(rotation=45)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Stage')

        plt.tight_layout()
        plt.savefig('involvement_summary.png')
        logging.info("Saved involvement summary as 'involvement_summary.png'")
        plt.close()


def create_origin_stage_comparison_barplot(origin_data,
                                           group,
                                           stages,
                                           master_region_list,
                                           min_count=2,
                                           output_dir=".", # Base directory for the specific comparison type
                                           use_csv_data=False,
                                           csv_file=None,
                                           limit_regions=False):
    """
    Create a horizontal bar chart showing origin distributions across stages
    for a single treatment group. Saves plot within the provided output_dir.
    """
    # output_dir is already specific (e.g., results/origin/within_group_comparison)
    os.makedirs(output_dir, exist_ok=True)

    # If using CSV data, read it from the file (optional, tries to unify with stats).
    # Note: csv_file path needs to be constructed correctly before calling this function if used.
    if use_csv_data and csv_file and os.path.exists(csv_file):
        try:
            csv_data = pd.read_csv(csv_file)
            group_data = csv_data[csv_data['Treatment_Group'] == group]
            plot_data = []
            for stage in stages:
                stage_data = group_data[group_data['Stage'] == stage]
                if not stage_data.empty:
                    total_waves = stage_data['Total_Waves'].iloc[0]
                    for _, row in stage_data.iterrows():
                        plot_data.append({
                            'Stage': stage,
                            'Region': row['Region'],
                            'Count': row['Count'],
                            'Total_Waves': total_waves,
                            'Percentage': row['Percentage']
                        })
            all_regions = set(group_data['Region'].unique())
            filtered_regions = [r for r in master_region_list if r in all_regions]

            # Sort regions by total count for better visualization
            region_total_counts = {}
            for region in filtered_regions:
                subdf = group_data[group_data['Region'] == region]
                region_total_counts[region] = subdf['Count'].sum()
            
            # Only limit regions if requested
            if limit_regions:
                max_regions = 15
                if len(filtered_regions) > max_regions:
                    filtered_regions = sorted(filtered_regions,
                                             key=lambda r: region_total_counts[r],
                                             reverse=True)[:max_regions]
            else:
                # Sort all regions by frequency
                filtered_regions = sorted(filtered_regions,
                                         key=lambda r: region_total_counts[r],
                                         reverse=True)

            df = pd.DataFrame(plot_data)

        except Exception as e:
            logging.error(f"Error reading CSV data: {e}. Falling back to direct calculation.")
            use_csv_data = False

    if not use_csv_data:
        # Collect relevant regions that meet min_count
        all_regions = set()
        for stage in stages:
            if stage in origin_data and 'region_counts' in origin_data[stage]:
                all_regions.update(origin_data[stage]['region_counts'].keys())

        # Filter by min_count across all stages
        filtered_regions = []
        for r in master_region_list:
            if r in all_regions:
                # Check if this region has >= min_count in any stage
                for stage in stages:
                    if stage in origin_data and 'region_counts' in origin_data[stage]:
                        if origin_data[stage]['region_counts'].get(r, 0) >= min_count:
                            filtered_regions.append(r)
                            break

        # Sort regions by frequency across stages
        region_total_counts = {}
        for region in filtered_regions:
            total_c = 0
            for stage in stages:
                if stage in origin_data and 'region_counts' in origin_data[stage]:
                    total_c += origin_data[stage]['region_counts'].get(region, 0)
            region_total_counts[region] = total_c
        
        # Only limit regions if requested
        if limit_regions:
            max_regions = 15
            if len(filtered_regions) > max_regions:
                filtered_regions = sorted(filtered_regions,
                                         key=lambda r: region_total_counts[r],
                                         reverse=True)[:max_regions]
        else:
            # Sort all regions by frequency
            filtered_regions = sorted(filtered_regions,
                                     key=lambda r: region_total_counts[r],
                                              reverse=True)

        if not filtered_regions:
            logging.warning(f"No regions with sufficient counts for origin comparison for {group} group")
            return (None, None)

        plot_data = []
        for stage in stages:
            if stage in origin_data and 'region_counts' in origin_data[stage]:
                rc = origin_data[stage]['region_counts']
                tw = origin_data[stage]['total_waves']
                for region in filtered_regions:
                    count = rc.get(region, 0)
                    percentage = (count / tw * 100) if tw > 0 else 0
                    plot_data.append({
                        'Stage': stage,
                        'Region': region,
                        'Count': count,
                        'Total_Waves': tw,
                        'Percentage': percentage
                    })

        df = pd.DataFrame(plot_data)

    if df.empty:
        logging.warning(f"No data available for origin comparison for {group} group")
        return (None, None)

    max_percentage = df['Percentage'].max() * 1.1 if not df.empty else 100 # for x-axis scaling

    # Create subplots for each stage with dynamic sizing based on number of regions
    region_count = len(df['Region'].unique())
    # Adjust height per region as count increases
    height_per_region = 0.4 if region_count <= 15 else 0.3 if region_count <= 30 else 0.25
    fig, axs = plt.subplots(1, len(stages),
                            figsize=(20, max(8, region_count * height_per_region)),
                            sharey=True)

    bar_width = 0.35
    # For consistent region ordering in the y-axis, gather them once
    unique_regions = df['Region'].unique()

    for i, stage in enumerate(stages):
        ax = axs[i] if len(stages) > 1 else axs
        stage_data = df[df['Stage'] == stage]
        if not stage_data.empty:
            # Adjust font size if there are many regions
            if region_count > 30:
                ax.tick_params(axis='y', labelsize=8)
            # We have 2 "groups" of bars if you want to subdivide; but here it's by stage alone.
            # Actually, we just plot a single bar because it's "one group." We'll do a single bar set.
            # Or we can just do one bar for each region. So it is a vertical/horizontal?
            # We'll do horizontal bars, each region is one bar.

            # Make sure to keep the same region order (filtered_regions)
            region_vals = []
            counts = []
            tws = []
            percentages = []
            for region in filtered_regions:
                row = stage_data[stage_data['Region'] == region]
                if not row.empty:
                    c = row['Count'].iloc[0]
                    p = row['Percentage'].iloc[0]
                    t = row['Total_Waves'].iloc[0]
                else:
                    c = 0
                    p = 0
                    t = 0
                region_vals.append(region)
                counts.append(c)
                percentages.append(p)
                tws.append(t)

            y_pos = np.arange(len(region_vals))
            bars = ax.barh(y_pos, percentages, height=0.7, color='skyblue')
            ax.set_title(f"{stage.capitalize()} Stage")
            ax.set_xlabel('Percentage of Waves (%)')
            ax.set_xlim(0, max_percentage)

            for j, bar in enumerate(bars):
                ax.text(bar.get_width() + 1,
                        bar.get_y() + bar.get_height()/2,
                        f"{counts[j]}/{tws[j]}",
                        va='center',
                        fontsize=8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(region_vals)
        else:
            ax.text(0.5, 0.5, f"No data for {stage} stage",
                    ha='center', va='center', transform=ax.transAxes)

    plt.suptitle(f"Origin Distribution Across Stages for {group} Group")
    axs[0].set_ylabel('Brain Region')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    filename = os.path.join(output_dir, f"origin_stages_{group.lower()}.png")
    plt.savefig(filename)
    logging.info(f"Saved origin stage comparison for {group} group as '{filename}'")
    plt.close()

    return (filtered_regions, max_percentage)


def create_combined_origin_comparison_barplot(origin_data,
                                              groups,
                                              stages,
                                              master_region_list,
                                              protocol_name=None, # Added for unique filenames
                                              filtered_regions=None,
                                              max_percentage=None,
                                              min_count=2,
                                              output_dir=".", # Base directory for the specific comparison type
                                              use_csv_data=False,
                                              csv_file=None,
                                              limit_regions=False):
    """
    Create a combined horizontal bar chart comparing origin distributions
    between treatment groups across all stages. Saves plot within the provided output_dir.
    Includes protocol_name in filename if provided.
    """
    # output_dir is already specific (e.g., results/origin/overall_comparison)
    os.makedirs(output_dir, exist_ok=True)

    # Attempt to read CSV data if requested
    # Note: csv_file path needs to be constructed correctly before calling this function if used.
    if use_csv_data and csv_file and os.path.exists(csv_file):
        try:
            csv_data = pd.read_csv(csv_file)
            plot_data = []
            for group in groups:
                group_data = csv_data[csv_data['Treatment_Group'] == group]
                for stage in stages:
                    stage_data = group_data[group_data['Stage'] == stage]
                    if not stage_data.empty:
                        tw = stage_data['Total_Waves'].iloc[0]
                        for _, row in stage_data.iterrows():
                            plot_data.append({
                                'Group': group,
                                'Stage': stage,
                                'Region': row['Region'],
                                'Count': row['Count'],
                                'Total_Waves': tw,
                                'Percentage': row['Percentage']
                            })

            df = pd.DataFrame(plot_data)

            # If filtered_regions was not provided, compute from CSV
            if filtered_regions is None:
                all_regions = set(df['Region'].unique())
                filtered_regions = [r for r in master_region_list if r in all_regions]

                # Sort regions by total count
                region_total_counts = {}
                for region in filtered_regions:
                    subdf = df[df['Region'] == region]
                    region_total_counts[region] = subdf['Count'].sum()
                
                # Only limit regions if requested
                if limit_regions:
                    max_regions = 15
                    if len(filtered_regions) > max_regions:
                        filtered_regions = sorted(filtered_regions,
                                                  key=lambda r: region_total_counts[r],
                                                  reverse=True)[:max_regions]
                else:
                    # Sort all regions by frequency
                    filtered_regions = sorted(filtered_regions,
                                              key=lambda r: region_total_counts[r],
                                              reverse=True)

        except Exception as e:
            logging.error(f"Error reading CSV data: {e}. Falling back to direct calculation.")
            use_csv_data = False

    if not use_csv_data:
        # If we have not built 'filtered_regions' from CSV, build them from the origin_data
        if filtered_regions is None:
            all_regions = set()
            for group in groups:
                if group in origin_data:
                    for stage in stages:
                        stage_dict = origin_data[group].get(stage, {})
                        if 'region_counts' in stage_dict:
                            all_regions.update(stage_dict['region_counts'].keys())

            # Filter by min_count
            preliminary = []
            for r in master_region_list:
                if r in all_regions:
                    # If it meets min_count in any group/stage
                    for group in groups:
                        for stage in stages:
                            stage_dict = origin_data[group].get(stage, {})
                            rc = stage_dict.get('region_counts', {})
                            if rc.get(r, 0) >= min_count:
                                preliminary.append(r)
                                break
            filtered_regions = preliminary

            # Sort regions by total occurrences
            region_total_counts = {}
            for region in filtered_regions:
                total_occurrences = 0
                for group in groups:
                    for stage in stages:
                        rc = origin_data[group].get(stage, {}).get('region_counts', {})
                        total_occurrences += rc.get(region, 0)
                region_total_counts[region] = total_occurrences
            
            # Only limit regions if requested
            if limit_regions:
                max_regions = 15
                if len(filtered_regions) > max_regions:
                    filtered_regions = sorted(filtered_regions,
                                              key=lambda r: region_total_counts[r],
                                              reverse=True)[:max_regions]
            else:
                # Sort all regions by frequency
                filtered_regions = sorted(filtered_regions,
                                          key=lambda r: region_total_counts[r],
                                          reverse=True)

        # Build a DataFrame for plotting
        plot_data = []
        for group in groups:
            if group in origin_data:
                for stage in stages:
                    stage_dict = origin_data[group].get(stage, {})
                    rc = stage_dict.get('region_counts', {})
                    tw = stage_dict.get('total_waves', 0)
                    for region in filtered_regions:
                        count = rc.get(region, 0)
                        percentage = (count / tw * 100) if tw > 0 else 0
                        plot_data.append({
                            'Group': group,
                            'Stage': stage,
                            'Region': region,
                            'Count': count,
                            'Total_Waves': tw,
                            'Percentage': percentage
                        })

        df = pd.DataFrame(plot_data)

    if df.empty or not filtered_regions:
        logging.warning(f"No regions with sufficient counts for combined origin comparison {f'for {protocol_name}' if protocol_name else ''}")
        return

    if max_percentage is None:
        max_percentage = df['Percentage'].max() * 1.1 if not df.empty else 100

    # Dynamic figure sizing based on number of regions
    region_count = len(filtered_regions)
    # Adjust height per region as count increases
    height_per_region = 0.4 if region_count <= 15 else 0.3 if region_count <= 30 else 0.25
    fig, axs = plt.subplots(1, len(stages),
                            figsize=(20, max(8, region_count * height_per_region)),
                            sharey=True)
    bar_width = 0.35
    y_pos = np.arange(len(filtered_regions))

    for i, stage in enumerate(stages):
        ax = axs[i] if len(stages) > 1 else axs
        stage_data = df[df['Stage'] == stage]
        if stage_data.empty:
            ax.text(0.5, 0.5, f"No data for {stage} stage",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        for j, group in enumerate(groups):
            group_stage_data = stage_data[stage_data['Group'] == group]
            if group_stage_data.empty:
                # Make a 0-value bar
                zeros = [0]*len(filtered_regions)
                bars = ax.barh(y_pos + j*bar_width, zeros,
                               bar_width, label=group if i == 0 else '')
            else:
                # Map region -> (percentage, count)
                region_pct = []
                region_cnt = []
                region_tw = []
                for region in filtered_regions:
                    row = group_stage_data[group_stage_data['Region'] == region]
                    if not row.empty:
                        p = row['Percentage'].iloc[0]
                        c = row['Count'].iloc[0]
                        t = row['Total_Waves'].iloc[0]
                    else:
                        p, c, t = 0, 0, 0
                    region_pct.append(p)
                    region_cnt.append(c)
                    region_tw.append(t)

                bars = ax.barh(y_pos + j*bar_width, region_pct,
                               bar_width,
                               label=group if i == 0 else '',
                               color=('lightblue' if group.lower() == 'active' else 'lightgreen'))

                for k, bar in enumerate(bars):
                    ax.text(bar.get_width() + 1,
                            bar.get_y() + bar.get_height()/2,
                            f"{region_cnt[k]}/{region_tw[k]}",
                            va='center', fontsize=8)

        ax.set_title(f"{stage.capitalize()}")
        ax.set_xlabel('Percentage of Waves (%)')
        ax.set_xlim(0, max_percentage)
        ax.set_yticks(y_pos + bar_width/2)
        # Adjust font size if there are many regions
        if len(filtered_regions) > 30:
            ax.tick_params(axis='y', labelsize=8)
        ax.set_yticklabels(filtered_regions)

    if len(groups) >= 2:
        axs[0].legend(loc='upper left')

    plt.suptitle('Combined Origin Distribution Comparison: Active vs. SHAM')
    axs[0].set_ylabel('Brain Region')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Construct filename, including protocol if provided
    base_filename = "combined_origin_comparison"
    if protocol_name:
        base_filename = f"{protocol_name}_{base_filename}"
    filename = os.path.join(output_dir, f"{base_filename}.png")

    plt.savefig(filename)
    logging.info(f"Saved combined origin comparison {f'for {protocol_name}' if protocol_name else ''} as '{filename}'")
    plt.close()


def visualize_overall_treatment_comparison(overall_comparison_results, source_dir=None): # Removed output_dir
    """
    Create visualizations for the overall treatment group comparison
    (collapsing subjects, nights, and protos).
    
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
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    # Get all stages from the involvement data
    involvement_stats = overall_comparison_results['overall_involvement_stats']
    all_stages = set()
    for group in involvement_stats:
        all_stages.update(involvement_stats[group].keys())
    
    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order

    involvement_stats = overall_comparison_results['overall_involvement_stats']
    origin_data = overall_comparison_results['overall_origin_data']
    master_region_list = overall_comparison_results['master_region_list']
    groups = list(involvement_stats.keys())

    # Construct paths to potential CSV files within the new structure
    origin_csv_file = os.path.join(origin_dir, "overall_origin_statistics.csv")
    use_origin_csv = os.path.exists(origin_csv_file)

    involvement_csv_file = os.path.join(involvement_dir, "overall_involvement_statistics.csv")
    use_involvement_csv = os.path.exists(involvement_csv_file)

    # 1) Barplot for involvement comparison
    if len(groups) >= 2:
        # Build data
        stage_data = []
        for stage in stages:
            stage_means = []
            stage_errors = []
            for group in groups:
                if stage in involvement_stats[group]:
                    s = involvement_stats[group][stage]
                    stage_means.append(s['mean'])
                    stage_errors.append(s['std'])
                else:
                    stage_means.append(0)
                    stage_errors.append(0)
            stage_data.append({'stage': stage, 'means': stage_means, 'errors': stage_errors})

        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        index = np.arange(len(stages))

        for i, group in enumerate(groups):
            means = [d['means'][i] for d in stage_data]
            errors = [d['errors'][i] for d in stage_data]
            plt.bar(index + i*bar_width, means, bar_width,
                    label=group,
                    color=('lightblue' if group.lower() == 'active' else 'lightgreen'),
                    yerr=errors, capsize=5)

        plt.xlabel('Stage')
        plt.ylabel('Mean Involvement (%)')
        plt.title('Overall Involvement Comparison: Active vs. SHAM (All Protos Combined)')
        plt.xticks(index + bar_width/2, [s.capitalize() for s in stages])
        plt.legend()

        # Add value labels
        for i, group in enumerate(groups):
            means = [d['means'][i] for d in stage_data]
            for j, mean in enumerate(means):
                if mean > 0:
                    plt.text(j + i*bar_width, mean + 1, f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        # Save to the new involvement directory
        filename = os.path.join(involvement_dir, "overall_involvement_comparison.png")
        plt.savefig(filename)
        logging.info(f"Saved overall involvement comparison barplot as '{filename}'")
        plt.close()

    # 2) Separate bar charts for each group - REMOVED as per Task 1

    # 3) Origin comparison visualizations - Only combined chart needed
    #   - Determine filtered regions and max percentage from combined data
    all_origin_plot_data = []
    for group in groups:
        if group in origin_data:
            for stage in stages:
                stage_dict = origin_data[group].get(stage, {})
                rc = stage_dict.get('region_counts', {})
                tw = stage_dict.get('total_waves', 0)
                for region, count in rc.items():
                    percentage = (count / tw * 100) if tw > 0 else 0
                    all_origin_plot_data.append({
                        'Group': group, 'Stage': stage, 'Region': region,
                        'Count': count, 'Total_Waves': tw, 'Percentage': percentage
                    })

    filtered_regions = None
    max_percentage = None
    if all_origin_plot_data:
        origin_df = pd.DataFrame(all_origin_plot_data)
        all_regions_set = set(origin_df['Region'].unique())
        # Filter by min_count across all groups/stages
        regions_meeting_min_count = []
        min_count_overall = 3 # Use the same threshold as before
        for r in master_region_list:
            if r in all_regions_set:
                # Check if count >= min_count_overall in any group/stage combo
                if origin_df[(origin_df['Region'] == r) & (origin_df['Count'] >= min_count_overall)].shape[0] > 0:
                     regions_meeting_min_count.append(r)

        # Sort by total count
        region_total_counts = origin_df.groupby('Region')['Count'].sum()
        filtered_regions = sorted(regions_meeting_min_count,
                                  key=lambda r: region_total_counts.get(r, 0),
                                  reverse=True)
        if filtered_regions:
             max_percentage = origin_df[origin_df['Region'].isin(filtered_regions)]['Percentage'].max() * 1.1

    #   - Create the combined chart
    if filtered_regions is not None and max_percentage is not None:
             create_combined_origin_comparison_barplot(
                 origin_data, # Pass the original structured data
                 groups,
                 stages,
                 master_region_list,
                 protocol_name=None, # No protocol name for overall comparison
                 filtered_regions=filtered_regions,
                 max_percentage=max_percentage,
                 min_count=3,
                 output_dir=origin_dir, # Save to the new origin directory
                 use_csv_data=use_origin_csv,
                 csv_file=origin_csv_file, # Use correctly constructed path
                 limit_regions=False  # Show all regions
             )


def plot_voxel_waveforms(csv_file, wave_name, num_voxels=3, source_dir=None): # Removed output_dir
    """
    Visualize the time series data for a few randomly selected voxels.

    Args:
        csv_file: Path to the original CSV file containing the time series data
        wave_name: Name of the wave file (for titling and saving)
        num_voxels: Number of random voxels to plot
        source_dir: Source directory where the data is read from, used to construct the output path

    This function:
    1. Loads the CSV data.
    2. Randomly selects `num_voxels` voxels.
    3. Plots the time series waveform for each selected voxel in a separate subplot.
    4. Uses scipy.signal.find_peaks to find local maxima.
    5. Marks local maxima with blue dots.
    6. Saves the figure to a structured directory.
    """
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")

    # Extract protocol and stage from the wave name for directory structure
    protocol = "unknown"
    stage = "unknown"
    protocol_match = re.search(r'(proto\d+)', wave_name, re.IGNORECASE)
    stage_match = re.search(r'(pre|early|late|post|stim)(?:-stim)?', wave_name, re.IGNORECASE)
    if protocol_match:
        protocol = protocol_match.group(1).lower()
    if stage_match:
        stage = stage_match.group(1).lower()

    # Create output directory if it doesn't exist
    plot_dir = os.path.join(base_output_dir, "voxel_plots", protocol, stage) # Use base_output_dir
    os.makedirs(plot_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_file)

        # Extract time points and convert to ms
        if 'Time' in df.columns:
            numeric_cols = []
            for col in df.columns[1:]:
                if not str(col).startswith('Unnamed'):
                    try:
                        float(col)  # test if col is numeric
                        numeric_cols.append(col)
                    except ValueError:
                        continue

            time_points = np.array([float(t) for t in numeric_cols]) * 1000  # Convert to ms
            data = np.abs(df.loc[:, numeric_cols].values)
            voxel_names = df.iloc[:, 0].values
        else:
            logging.error(f"CSV format doesn't match expected format for {wave_name}")
            return False

        # Select random voxels
        num_available_voxels = data.shape[0]
        if num_available_voxels < num_voxels:
            logging.warning(f"Only {num_available_voxels} voxels available in {wave_name}, plotting all.")
            selected_indices = list(range(num_available_voxels))
        else:
            selected_indices = random.sample(range(num_available_voxels), num_voxels)

        if not selected_indices:
            logging.error(f"No voxels to plot for {wave_name}")
            return False

        # Create subplots
        fig, axs = plt.subplots(len(selected_indices), 1, figsize=(10, 3 * len(selected_indices)), sharex=True)
        if len(selected_indices) == 1:
            axs = [axs] # Make it iterable if only one subplot

        for i, voxel_idx in enumerate(selected_indices):
            voxel_data = data[voxel_idx, :]
            voxel_name = voxel_names[voxel_idx]

            # Find peaks
            peaks_indices, _ = find_peaks(voxel_data)

            # Plot waveform
            axs[i].plot(time_points, voxel_data, label=f'Voxel {voxel_idx}')
            # Plot peaks
            if len(peaks_indices) > 0:
                axs[i].plot(time_points[peaks_indices], voxel_data[peaks_indices], 'bo', label='Local Maxima')

            axs[i].set_title(f'Voxel: {voxel_name}')
            axs[i].set_ylabel('Amplitude')
            axs[i].legend(loc='upper right')
            axs[i].grid(True, linestyle='--', alpha=0.6)

        axs[-1].set_xlabel('Time (ms)')
        fig.suptitle(f'Random Voxel Waveforms - {wave_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        # Save the figure
        filename = os.path.join(plot_dir, f"{wave_name}_voxel_waveforms.png")
        plt.savefig(filename)
        logging.info(f"Saved voxel waveform plot as '{filename}'")

        # Save a copy next to the original CSV file
        csv_dir = os.path.dirname(str(csv_file))
        csv_basename = os.path.basename(str(csv_file))
        local_filename = os.path.join(csv_dir, f"{os.path.splitext(csv_basename)[0]}_voxel_waveforms.png")
        plt.savefig(local_filename)
        logging.info(f"Saved copy of voxel waveform plot next to CSV as '{local_filename}'")

        plt.close(fig)
        return True

    except Exception as e:
        logging.error(f"Error visualizing voxel waveforms for {wave_name}: {str(e)}")
        return False


def visualize_proto_specific_comparison(proto_specific_results, source_dir=None): # Removed output_dir
    """
    Create visualizations for the proto-specific comparison.

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
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    # Get all stages from the protocol-specific results
    proto_results = proto_specific_results['proto_specific_results']
    all_stages = set()
    for protocol in proto_results:
        for group in proto_results[protocol]['involvement_stats']:
            all_stages.update(proto_results[protocol]['involvement_stats'][group].keys())
    
    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order
    master_region_list = proto_specific_results['master_region_list']
    proto_results = proto_specific_results['proto_specific_results']

    # Construct paths to potential CSV files within the new structure
    origin_csv_file = os.path.join(origin_dir, "proto_specific_origin_statistics.csv")
    use_origin_csv = os.path.exists(origin_csv_file)

    involvement_csv_file = os.path.join(involvement_dir, "proto_specific_involvement_statistics.csv")
    use_involvement_csv = os.path.exists(involvement_csv_file)


    for protocol, results in proto_results.items():
        involvement_stats = results['involvement_stats']
        origin_data = results['origin_data']
        groups = list(involvement_stats.keys())

        # No longer need protocol-specific subfolder here, comparison type folder handles it

        # 1) Combined barplot for involvement comparison between treatment groups
        if len(groups) >= 2:
            # Build data
            stage_data = []
            for stage in stages:
                stage_means = []
                stage_errors = []
                for group in groups:
                    if stage in involvement_stats[group]:
                        s = involvement_stats[group][stage]
                        stage_means.append(s['mean'])
                        stage_errors.append(s['std'])
                    else:
                        stage_means.append(0)
                        stage_errors.append(0)
                stage_data.append({'stage': stage, 'means': stage_means, 'errors': stage_errors})

            plt.figure(figsize=(12, 8))
            bar_width = 0.35
            index = np.arange(len(stages))

            for i, group in enumerate(groups):
                means = [d['means'][i] for d in stage_data]
                errors = [d['errors'][i] for d in stage_data]
                plt.bar(index + i*bar_width, means, bar_width,
                        label=group,
                        color=('lightblue' if group.lower() == 'active' else 'lightgreen'),
                        yerr=errors, capsize=5)

            plt.xlabel('Stage')
            plt.ylabel('Mean Involvement (%)')
            plt.title(f'Involvement Comparison: Active vs. SHAM for {protocol}')
            plt.xticks(index + bar_width/2, [s.capitalize() for s in stages])
            plt.legend()

            # Add value labels
            for i, group in enumerate(groups):
                means = [d['means'][i] for d in stage_data]
                for j, mean in enumerate(means):
                    if mean > 0:
                        plt.text(j + i*bar_width, mean + 1, f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            # Save to the new involvement directory, including protocol in filename
            filename = os.path.join(involvement_dir, f"{protocol}_involvement_comparison.png")
            plt.savefig(filename)
            logging.info(f"Saved involvement comparison barplot for {protocol} as '{filename}'")
            plt.close()

        # 2) Create individual bar charts for involvement for each group - REMOVED as per Task 1

        # 3) Create combined origin comparison barplot for this protocol
        #    (Individual group plots are not requested for this comparison type)

        # Determine filtered regions and max percentage from combined data for this protocol
        proto_origin_plot_data = []
        for group in groups:
            if group in origin_data:
                for stage in stages:
                    stage_dict = origin_data[group].get(stage, {})
                    rc = stage_dict.get('region_counts', {})
                    tw = stage_dict.get('total_waves', 0)
                    for region, count in rc.items():
                        percentage = (count / tw * 100) if tw > 0 else 0
                        proto_origin_plot_data.append({
                            'Group': group, 'Stage': stage, 'Region': region,
                            'Count': count, 'Total_Waves': tw, 'Percentage': percentage
                        })

        filtered_regions = None
        max_percentage = None
        if proto_origin_plot_data:
            origin_df = pd.DataFrame(proto_origin_plot_data)
            all_regions_set = set(origin_df['Region'].unique())
            regions_meeting_min_count = []
            min_count_proto = 2 # Use the same threshold as before
            for r in master_region_list:
                 if r in all_regions_set:
                     if origin_df[(origin_df['Region'] == r) & (origin_df['Count'] >= min_count_proto)].shape[0] > 0:
                          regions_meeting_min_count.append(r)

            region_total_counts = origin_df.groupby('Region')['Count'].sum()
            filtered_regions = sorted(regions_meeting_min_count,
                                      key=lambda r: region_total_counts.get(r, 0),
                                      reverse=True)
            if filtered_regions:
                 max_percentage = origin_df[origin_df['Region'].isin(filtered_regions)]['Percentage'].max() * 1.1


        # Create the combined origin comparison barplot if we have multiple groups and data
        if filtered_regions is not None and max_percentage is not None and len(groups) >= 2:
             create_combined_origin_comparison_barplot(
                 origin_data, # Pass original structured data
                 groups,
                 stages,
                 master_region_list,
                 protocol_name=protocol, # Pass protocol name for unique filename
                 filtered_regions=filtered_regions,
                 max_percentage=max_percentage,
                 min_count=2,
                 output_dir=origin_dir, # Save to the new origin directory
                 use_csv_data=use_origin_csv, # Use overall proto-specific CSV flag
                 csv_file=origin_csv_file, # Use overall proto-specific CSV path
                 limit_regions=False  # Show all regions
             )


def visualize_region_time_series(wave_data, csv_file, source_dir=None): # Removed output_dir
    """
    Visualize the time series data for each brain region.

    Args:
        wave_data: Dictionary containing the results from analyze_slow_wave
        csv_file: Path to the original CSV file containing the time series data
        source_dir: Source directory where the data is read from, used to construct the output path

    This function:
    1. Groups voxels by region
    2. Plots time series data for each region (-50ms to 50ms)
    3. Marks local maxima with blue dots
    4. Marks origins with red dots
    """
    # Define base output directory structure
    base_output_dir = "results"
    if source_dir:
        base_output_dir = os.path.join(source_dir, "Source_Ana")

    # Extract protocol and stage from wave_name
    protocol = "unknown"
    stage = "unknown"
    
    # Extract protocol and stage from the wave name
    wave_name = wave_data["wave_name"]
    protocol_match = re.search(r'(proto\d+)', wave_name, re.IGNORECASE)
    stage_match = re.search(r'(pre|early|late|post|stim)(?:-stim)?', wave_name, re.IGNORECASE)
    
    if protocol_match:
        protocol = protocol_match.group(1).lower()
    if stage_match:
        stage = stage_match.group(1).lower()

    # Create subdirectories
    plot_dir = os.path.join(base_output_dir, "region_plots", protocol, stage) # Use base_output_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Load the CSV data
    try:
        df = pd.read_csv(csv_file)
        
        # Extract time points and convert to ms
        if 'Time' in df.columns:
            numeric_cols = []
            for col in df.columns[1:]:
                if not str(col).startswith('Unnamed'):
                    try:
                        float(col)  # test if col is numeric
                        numeric_cols.append(col)
                    except ValueError:
                        continue

            time_points = np.array([float(t) for t in numeric_cols]) * 1000  # Convert to ms
            data = np.abs(df.loc[:, numeric_cols].values)
            voxel_names = df.iloc[:, 0].values

            # Filter time points to the window of -50ms to 50ms
            window_start = -50  # ms
            window_end = 50     # ms
            window_mask = (time_points >= window_start) & (time_points <= window_end)

            if sum(window_mask) == 0:
                logging.warning(f"No time points found in window [{window_start}, {window_end}] ms for {wave_name}")
                return False # Indicate failure

            window_times = time_points[window_mask]
            window_data = data[:, window_mask]

            # Group voxels by region
            region_voxels = {}
            for i, voxel_name in enumerate(voxel_names):
                region = extract_region_name(voxel_name)
                if region not in region_voxels:
                    region_voxels[region] = []
                region_voxels[region].append(i)
            
            # Calculate average time series for each region
            region_time_series = {}
            for region, voxel_indices in region_voxels.items():
                region_data = np.mean(window_data[voxel_indices, :], axis=0)
                region_time_series[region] = region_data
            
            # Create a figure for all regions
            plt.figure(figsize=(10, 8))
            
            # Store local maxima for each region with more details
            region_maxima = {}
            
            # Plot time series for each region
            for region, time_series in region_time_series.items():
                # Plot the time series
                plt.plot(window_times, time_series, label=region)
                
                # Find local maxima
                maxima_indices = []
                for i in range(1, len(time_series)-1):
                    if time_series[i] > time_series[i-1] and time_series[i] > time_series[i+1]:
                        maxima_indices.append(i)
                
                # Mark local maxima with blue dots
                if maxima_indices:
                    max_times = window_times[maxima_indices]
                    max_values = time_series[maxima_indices]
                    plt.scatter(max_times, max_values, color='blue', s=30, zorder=3)
                    
                    # Store local maxima info in a more structured way
                    region_maxima[region] = {
                        'indices': maxima_indices,
                        'times': max_times,
                        'values': max_values
                    }
            
            # Mark origins with red stars, ensuring they align with local maxima
            if 'origins' in wave_data and not wave_data['origins'].empty:
                for _, row in wave_data['origins'].iterrows():
                    region = row['region']
                    origin_time = row['peak_time']
                    
                    if region in region_maxima and len(region_maxima[region]['times']) > 0:
                        # Find the closest local maximum to the origin time
                        local_max_times = region_maxima[region]['times']
                        local_max_values = region_maxima[region]['values']
                        closest_idx = np.argmin(np.abs(local_max_times - origin_time))
                        closest_max_time = local_max_times[closest_idx]
                        closest_max_value = local_max_values[closest_idx]
                        
                        # Check if the closest local maximum is within a reasonable time window (e.g., 5ms)
                        if abs(closest_max_time - origin_time) <= 5:  # 5ms tolerance
                            # Plot the origin at the actual local maximum
                            plt.scatter(closest_max_time, closest_max_value,
                                       color='red', s=50, marker='*', zorder=4)
                        else:
                            logging.warning(f"No local maximum found near origin time {origin_time:.2f}ms "
                                  f"for region {region} in wave {wave_name} within ±5ms window")
                    else:
                        logging.warning(f"No local maxima detected for origin region {region} in wave {wave_name}")

            # Add labels and title
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.title(f'Region Time Series - {wave_name}')
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)  # Mark t=0
            
            # Add legend with smaller font
            plt.legend(fontsize='small', loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure to the organized directory structure
            filename = os.path.join(plot_dir, f"{wave_name}_region_time_series.png")
            plt.savefig(filename)
            logging.info(f"Saved region time series plot as '{filename}'")

            # Save a copy next to the original CSV file
            csv_dir = os.path.dirname(str(csv_file))
            csv_basename = os.path.basename(str(csv_file))
            local_filename = os.path.join(csv_dir, f"{os.path.splitext(csv_basename)[0]}_region_time_series.png")
            plt.savefig(local_filename)
            logging.info(f"Saved copy of region time series plot next to CSV as '{local_filename}'")

            plt.close()

            return True

        else:
            logging.error(f"CSV format doesn't match expected format for {wave_name}")
            return False

    except Exception as e:
        logging.error(f"Error visualizing region time series for {wave_name}: {str(e)}")
        return False


def visualize_within_group_stage_comparison(within_group_results, source_dir=None): # Removed output_dir
    """
    Create visualizations for the within-group stage comparison.

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
    os.makedirs(involvement_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    # Get all stages from the within-group results
    results_dict = within_group_results['within_group_results']
    all_stages = set()
    for group in results_dict:
        all_stages.update(results_dict[group]['involvement_stats'].keys())
    
    # Determine standard stage order if possible
    if all_stages == {'pre', 'early', 'late', 'post'}:
        stages = ['pre', 'early', 'late', 'post']  # 4-stage scheme
    elif all_stages == {'pre', 'stim', 'post'}:
        stages = ['pre', 'stim', 'post']  # 3-stage scheme
    else:
        stages = sorted(all_stages)  # fallback to alphabetical order
    master_region_list = within_group_results['master_region_list']
    results_dict = within_group_results['within_group_results']

    # Construct paths to potential CSV files within the new structure
    origin_csv_file = os.path.join(origin_dir, "within_group_origin_statistics.csv")
    use_origin_csv = os.path.exists(origin_csv_file)

    involvement_csv_file = os.path.join(involvement_dir, "within_group_involvement_statistics.csv")
    use_involvement_csv = os.path.exists(involvement_csv_file)


    for group, results in results_dict.items():
        involvement_stats = results['involvement_stats']
        origin_data = results['origin_data']

        # No longer need group-specific subfolder here

        # Single bar chart for involvement across the stages for this group
        plt.figure(figsize=(10, 6))
        means = []
        errors = []
        valid_stages = []
        for stage in stages:
            if stage in involvement_stats:
                s = involvement_stats[stage]
                if s['count'] > 0:
                    means.append(s['mean'])
                    errors.append(s['std'])
                    valid_stages.append(stage)

        if valid_stages:
            plt.bar(valid_stages, means, yerr=errors, capsize=5, color='skyblue')
            plt.xlabel('Stage')
            plt.ylabel('Mean Involvement (%)')
            plt.title(f'Involvement Across Stages for {group} Group')
            plt.xticks(rotation=0)

            for i, mean in enumerate(means):
                if mean > 0:
                    plt.text(i, mean + 1, f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            # Save to the new involvement directory, including group in filename
            filename = os.path.join(involvement_dir, f"within_group_involvement_{group.lower()}.png")
            plt.savefig(filename)
            logging.info(f"Saved within-group involvement barplot for {group} as '{filename}'")
            plt.close()

        # Create origin comparison barplot for this group
        create_origin_stage_comparison_barplot(
            origin_data,
            group,
            stages,
            master_region_list,
            min_count=2,
            output_dir=origin_dir, # Save to the new origin directory
            use_csv_data=use_origin_csv,
            csv_file=origin_csv_file, # Use correctly constructed path
            limit_regions=False  # Show all regions
        )
