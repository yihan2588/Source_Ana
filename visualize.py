import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import Counter

def visualize_results(all_results, master_region_list):
    """
    Create visualizations for individual protocol analysis (boxplots, etc.) if desired.
    Currently, this is mostly a placeholder or minimal usage.
    """
    pass


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
    print(f"\nSaved involvement boxplot as '{filename}'")
    plt.close()


def create_involvement_summary(all_results, protocols):
    """
    Create a summary bar chart of involvement across stages for each protocol.
    """
    protocols_list = []
    stages_list = []
    means = []
    errors = []

    for protocol in protocols:
        for stage in ['pre', 'early', 'late', 'post']:
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
        stage_colors = {'pre': 'skyblue', 'early': 'lightgreen', 'late': 'salmon', 'post': 'gold'}

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
        print("Saved involvement summary as 'involvement_summary.png'")
        plt.close()


def create_origin_stage_comparison_barplot(origin_data,
                                           group,
                                           stages,
                                           master_region_list,
                                           min_count=2,
                                           output_dir=".",
                                           use_csv_data=False,
                                           csv_file=None):
    """
    Create a horizontal bar chart showing origin distributions across stages
    for a single treatment group. This was merged from create_origin_stage_comparison_barplot.py
    """
    os.makedirs(output_dir, exist_ok=True)

    # If using CSV data, read it from the file (optional, tries to unify with stats).
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

            # Limit to top 15 by total count if needed
            max_regions = 15
            if len(filtered_regions) > max_regions:
                region_total_counts = {}
                for region in filtered_regions:
                    subdf = group_data[group_data['Region'] == region]
                    region_total_counts[region] = subdf['Count'].sum()
                filtered_regions = sorted(filtered_regions,
                                          key=lambda r: region_total_counts[r],
                                          reverse=True)[:max_regions]

            df = pd.DataFrame(plot_data)

        except Exception as e:
            print(f"Error reading CSV data: {e}. Falling back to direct calculation.")
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

        # Limit to top 15 by sum across stages
        max_regions = 15
        if len(filtered_regions) > max_regions:
            region_total_counts = {}
            for region in filtered_regions:
                total_c = 0
                for stage in stages:
                    if stage in origin_data and 'region_counts' in origin_data[stage]:
                        total_c += origin_data[stage]['region_counts'].get(region, 0)
                region_total_counts[region] = total_c
            filtered_regions = sorted(filtered_regions,
                                      key=lambda r: region_total_counts[r],
                                      reverse=True)[:max_regions]

        if not filtered_regions:
            print(f"No regions with sufficient counts for origin comparison for {group} group")
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
        print(f"No data available for origin comparison for {group} group")
        return (None, None)

    max_percentage = df['Percentage'].max() * 1.1  # for x-axis scaling

    # Create subplots for each stage
    fig, axs = plt.subplots(1, len(stages),
                            figsize=(20, max(8, len(df['Region'].unique()) * 0.4)),
                            sharey=True)

    bar_width = 0.35
    # For consistent region ordering in the y-axis, gather them once
    unique_regions = df['Region'].unique()

    for i, stage in enumerate(stages):
        ax = axs[i] if len(stages) > 1 else axs
        stage_data = df[df['Stage'] == stage]
        if not stage_data.empty:
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
    print(f"Saved origin stage comparison for {group} group as '{filename}'")
    plt.close()

    return (filtered_regions, max_percentage)


def create_combined_origin_comparison_barplot(origin_data,
                                              groups,
                                              stages,
                                              master_region_list,
                                              filtered_regions=None,
                                              max_percentage=None,
                                              min_count=2,
                                              output_dir=".",
                                              use_csv_data=False,
                                              csv_file=None):
    """
    Create a combined horizontal bar chart comparing origin distributions
    between treatment groups across all stages.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Attempt to read CSV data if requested
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

                # Limit to top 15
                max_regions = 15
                if len(filtered_regions) > max_regions:
                    region_total_counts = {}
                    for region in filtered_regions:
                        subdf = df[df['Region'] == region]
                        region_total_counts[region] = subdf['Count'].sum()
                    filtered_regions = sorted(filtered_regions,
                                              key=lambda r: region_total_counts[r],
                                              reverse=True)[:max_regions]

        except Exception as e:
            print(f"Error reading CSV data: {e}. Falling back to direct calculation.")
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

            # Limit to top 15
            max_regions = 15
            if len(filtered_regions) > max_regions:
                region_total_counts = {}
                for region in filtered_regions:
                    total_occurrences = 0
                    for group in groups:
                        for stage in stages:
                            rc = origin_data[group].get(stage, {}).get('region_counts', {})
                            total_occurrences += rc.get(region, 0)
                    region_total_counts[region] = total_occurrences
                filtered_regions = sorted(filtered_regions,
                                          key=lambda r: region_total_counts[r],
                                          reverse=True)[:max_regions]

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
        print("No regions with sufficient counts for combined origin comparison")
        return

    if max_percentage is None:
        max_percentage = df['Percentage'].max() * 1.1

    fig, axs = plt.subplots(1, len(stages),
                            figsize=(20, max(8, len(filtered_regions) * 0.4)),
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
        ax.set_yticklabels(filtered_regions)

    if len(groups) >= 2:
        axs[0].legend(loc='upper left')

    plt.suptitle('Combined Origin Distribution Comparison: Active vs. SHAM')
    axs[0].set_ylabel('Brain Region')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    filename = os.path.join(output_dir, "combined_origin_comparison.png")
    plt.savefig(filename)
    print(f"Saved combined origin comparison as '{filename}'")
    plt.close()


def visualize_overall_treatment_comparison(overall_comparison_results, output_dir="results"):
    """
    Create visualizations for the overall treatment group comparison
    (collapsing subjects, nights, and protos).
    """
    os.makedirs(output_dir, exist_ok=True)
    stages = ['pre', 'early', 'late', 'post']

    involvement_stats = overall_comparison_results['overall_involvement_stats']
    origin_data = overall_comparison_results['overall_origin_data']
    master_region_list = overall_comparison_results['master_region_list']
    groups = list(involvement_stats.keys())

    # Attempt to read CSV files for consistent stats usage (if they exist)
    csv_file = os.path.join(output_dir, "overall_origin_statistics.csv")
    use_csv_data = os.path.exists(csv_file)

    involvement_csv_file = os.path.join(output_dir, "overall_involvement_statistics.csv")
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
        filename = os.path.join(output_dir, "overall_involvement_comparison.png")
        plt.savefig(filename)
        print(f"\nSaved overall involvement comparison barplot as '{filename}'")
        plt.close()

    # 2) Separate bar charts for each group
    for group in groups:
        plt.figure(figsize=(10, 6))
        means = []
        errors = []
        valid_stages = []
        for stage in stages:
            if stage in involvement_stats[group]:
                s = involvement_stats[group][stage]
                if s['count'] > 0:
                    means.append(s['mean'])
                    errors.append(s['std'])
                    valid_stages.append(stage)

        if valid_stages:
            plt.bar(valid_stages, means, yerr=errors, capsize=5,
                    color='lightblue' if group.lower() == 'active' else 'lightgreen')
            plt.xlabel('Stage')
            plt.ylabel('Mean Involvement (%)')
            plt.title(f'Involvement Across Stages for {group} Group (All Protos Combined)')
            plt.xticks(rotation=0)

            for i, mean in enumerate(means):
                if mean > 0:
                    plt.text(i, mean + 1, f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            filename = os.path.join(output_dir, f"overall_{group.lower()}_involvement.png")
            plt.savefig(filename)
            print(f"Saved overall {group} involvement barplot as '{filename}'")
            plt.close()

    # 3) Origin comparison visualizations
    #   - First, individual charts for each group
    filtered_regions = None
    max_percentage = None
    for group in groups:
        if group in origin_data:
            fr, mp = create_origin_stage_comparison_barplot(
                origin_data[group],
                group,
                stages,
                master_region_list,
                min_count=3,
                output_dir=output_dir,
                use_csv_data=use_csv_data,
                csv_file=csv_file
            )
            # Keep last valid for combined usage
            if fr is not None:
                filtered_regions = fr
                max_percentage = mp

    #   - Then create a combined chart using the same filtered regions & max percentage
    if filtered_regions is not None:
        create_combined_origin_comparison_barplot(
            origin_data,
            groups,
            stages,
            master_region_list,
            filtered_regions=filtered_regions,
            max_percentage=max_percentage,
            min_count=3,
            output_dir=output_dir,
            use_csv_data=use_csv_data,
            csv_file=csv_file
        )


def visualize_proto_specific_comparison(proto_specific_results, output_dir="results"):
    """
    Create visualizations for the proto-specific comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    stages = ['pre', 'early', 'late', 'post']
    master_region_list = proto_specific_results['master_region_list']
    proto_results = proto_specific_results['proto_specific_results']

    # Attempt CSV usage
    csv_file = os.path.join(output_dir, "proto_specific_origin_statistics.csv")
    use_csv_data = os.path.exists(csv_file)

    for protocol, results in proto_results.items():
        involvement_stats = results['involvement_stats']
        origin_data = results['origin_data']
        groups = list(involvement_stats.keys())

        # Protocol-specific folder
        proto_output_dir = os.path.join(output_dir, protocol)
        os.makedirs(proto_output_dir, exist_ok=True)

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
            filename = os.path.join(proto_output_dir, f"{protocol}_involvement_comparison.png")
            plt.savefig(filename)
            print(f"\nSaved involvement comparison barplot for {protocol} as '{filename}'")
            plt.close()

        # 2) Create individual bar charts for involvement for each group
        for group in groups:
            plt.figure(figsize=(10, 6))
            means = []
            errors = []
            valid_stages = []
            for stage in stages:
                if stage in involvement_stats[group]:
                    s = involvement_stats[group][stage]
                    if s['count'] > 0:
                        means.append(s['mean'])
                        errors.append(s['std'])
                        valid_stages.append(stage)

            if valid_stages:
                plt.bar(valid_stages,
                        means,
                        yerr=errors,
                        capsize=5,
                        color='lightblue' if group.lower() == 'active' else 'lightgreen')
                plt.xlabel('Stage')
                plt.ylabel('Mean Involvement (%)')
                plt.title(f'{group} Involvement Across Stages for {protocol}')
                plt.xticks(rotation=0)

                for i, mean in enumerate(means):
                    if mean > 0:
                        plt.text(i, mean + 1, f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                filename = os.path.join(proto_output_dir, f"{protocol}_{group.lower()}_involvement.png")
                plt.savefig(filename)
                print(f"Saved {group} involvement barplot for {protocol} as '{filename}'")
                plt.close()

        # Create origin comparison for each group
        proto_csv_file = None
        if use_csv_data:
            # Attempt to filter CSV data for just this protocol
            try:
                df = pd.read_csv(csv_file)
                proto_data = df[df['Protocol'].str.lower() == protocol.lower()]
                if not proto_data.empty:
                    proto_csv_file = os.path.join(proto_output_dir, f"{protocol}_origin_statistics.csv")
                    proto_data.to_csv(proto_csv_file, index=False)
            except Exception as e:
                print(f"Error filtering CSV data for protocol {protocol}: {e}")
                proto_csv_file = None

        filtered_regions = None
        max_percentage = None
        for group in groups:
            if group in origin_data:
                fr, mp = create_origin_stage_comparison_barplot(
                    origin_data[group],
                    group,
                    stages,
                    master_region_list,
                    min_count=2,
                    output_dir=proto_output_dir,
                    use_csv_data=(proto_csv_file is not None),
                    csv_file=proto_csv_file
                )
                if fr is not None:
                    filtered_regions = fr
                    max_percentage = mp
        
        # Create a combined origin comparison barplot for this protocol if we have multiple groups
        if filtered_regions is not None and len(groups) >= 2:
            create_combined_origin_comparison_barplot(
                origin_data,
                groups,
                stages,
                master_region_list,
                filtered_regions=filtered_regions,
                max_percentage=max_percentage,
                min_count=2,
                output_dir=proto_output_dir,
                use_csv_data=(proto_csv_file is not None),
                csv_file=proto_csv_file
            )


def visualize_within_group_stage_comparison(within_group_results, output_dir="results"):
    """
    Create visualizations for the within-group stage comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    stages = ['pre', 'early', 'late', 'post']
    master_region_list = within_group_results['master_region_list']
    results_dict = within_group_results['within_group_results']

    csv_file = os.path.join(output_dir, "within_group_origin_statistics.csv")
    use_csv_data = os.path.exists(csv_file)

    for group, results in results_dict.items():
        involvement_stats = results['involvement_stats']
        origin_data = results['origin_data']

        group_output_dir = os.path.join(output_dir, f"{group.lower()}_stages")
        os.makedirs(group_output_dir, exist_ok=True)

        # Single bar chart for involvement across the 4 stages
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
            filename = os.path.join(group_output_dir, f"within_group_involvement_{group.lower()}.png")
            plt.savefig(filename)
            print(f"Saved within-group involvement barplot for {group} as '{filename}'")
            plt.close()

        # Create origin comparison barplot
        create_origin_stage_comparison_barplot(
            origin_data,
            group,
            stages,
            master_region_list,
            min_count=2,
            output_dir=group_output_dir,
            use_csv_data=use_csv_data,
            csv_file=csv_file
        )
