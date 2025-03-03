import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

# Helper functions for visualization
def create_involvement_boxplot(data, labels, title, filename, colors=None):
    """
    Create a boxplot for involvement data.
    
    Args:
        data: List of data arrays to plot
        labels: Labels for each box
        title: Plot title
        filename: Output filename
        colors: Optional dictionary mapping stages to colors
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

def create_horizontal_bar_chart(axs, results, stages, master_region_list, protocol_name):
    """
    Create horizontal bar charts for origin regions.
    
    Args:
        axs: Array of matplotlib axes
        results: Dictionary with results by stage
        stages: List of stages to plot
        master_region_list: Ordered list of regions
        protocol_name: Name of the protocol
    """
    # Collect all regions for this protocol
    protocol_regions = set()
    for stage in stages:
        if stage in results:
            for result in results[stage]:
                if 'origins' in result and not result['origins'].empty:
                    regions = set(result['origins']['region'].tolist())
                    protocol_regions.update(regions)
    
    protocol_ordered_regions = [r for r in master_region_list if r in protocol_regions]
    
    for i, stage in enumerate(stages):
        if stage in results:
            waves_with_origin_regions = {}
            for j, result in enumerate(results[stage]):
                if 'origins' in result and not result['origins'].empty:
                    wave_id = f"wave_{j}"
                    wave_regions = set(result['origins']['region'].tolist())
                    waves_with_origin_regions[wave_id] = wave_regions
            
            region_wave_counts = Counter()
            for wave_regions in waves_with_origin_regions.values():
                region_wave_counts.update(wave_regions)
            
            total_waves = len(results[stage])
            regions_to_plot = []
            percentages = []
            counts = []
            for region in protocol_ordered_regions:
                if region in region_wave_counts:
                    regions_to_plot.append(region)
                    count = region_wave_counts[region]
                    counts.append(count)
                    percentages.append((count/total_waves*100) if total_waves > 0 else 0)
            
            if regions_to_plot:
                bars = axs[i].barh(regions_to_plot, percentages, color='skyblue')
                axs[i].set_title(f"{stage.capitalize()} (n={total_waves})")
                axs[i].set_xlabel('Percentage of Waves (%)')
                for bar, cnt in zip(bars, counts):
                    axs[i].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                                f"{cnt}/{total_waves}", va='center')
            else:
                axs[i].text(0.5, 0.5, "No data", ha='center', va='center', transform=axs[i].transAxes)
        else:
            axs[i].text(0.5, 0.5, "No data", ha='center', va='center', transform=axs[i].transAxes)
    
    plt.suptitle(f'Origin Regions for {protocol_name}')
    axs[0].set_ylabel('Brain Region')
    if protocol_ordered_regions:
        for ax in axs:
            ax.set_ylim(-0.5, len(protocol_ordered_regions) - 0.5)

def create_involvement_summary(all_results, protocols):
    """
    Create a summary bar chart of involvement across stages for each protocol.
    
    Args:
        all_results: Dictionary with results by protocol and stage
        protocols: List of protocols to include
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
                            label=row['Stage'] if row['Stage'] not in plt.gca().get_legend_handles_labels()[1] else '')
        
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

def visualize_results(all_results, master_region_list):
    """Create visualizations for individual protocol analysis with consistent region ordering"""
    protocols = list(all_results.keys())
    stages = ['pre', 'early', 'late', 'post']
    
    # Plot 1: Involvement percentages by protocol and stage
    all_data = []
    labels = []
    for protocol in protocols:
        for stage in stages:
            if stage in all_results[protocol]:
                data = [res['involvement_percentage'] for res in all_results[protocol][stage]]
                if data:
                    all_data.append(data)
                    labels.append(f"{protocol}\n{stage}")
    
    colors = {'pre': 'lightblue', 'early': 'lightgreen', 'late': 'salmon', 'post': 'lightyellow'}
    create_involvement_boxplot(all_data, labels, 'Involvement Percentage by Protocol and Stage', 
                              'involvement_by_protocol_and_stage.png', colors)
    
    # Plot 2: Origins by protocol (horizontal bar charts)
    for protocol in protocols:
        fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)
        create_horizontal_bar_chart(axs, all_results[protocol], stages, master_region_list, protocol)
        plt.tight_layout()
        plt.savefig(f'origins_{protocol}.png')
        print(f"Saved origins chart for {protocol} as 'origins_{protocol}.png'")
        plt.close()
    
    # Plot 4: Summary bar chart of involvement across stages for each protocol
    create_involvement_summary(all_results, protocols)

def visualize_meta_results(meta_results):
    """Create visualizations for the meta (protocol-blind) analysis"""
    stages = ['pre', 'early', 'late', 'post']
    
    # 1) Meta involvement boxplot
    meta_involvement_data = meta_results['meta_involvement_data']
    data = [meta_involvement_data[st] for st in stages if meta_involvement_data[st]]
    labels = [st for st in stages if meta_involvement_data[st]]
    
    colors = {'pre': 'lightblue', 'early': 'lightgreen', 'late': 'salmon', 'post': 'lightyellow'}
    create_involvement_boxplot(data, labels, 'Meta Involvement Percentage by Stage', 
                              'meta_involvement_boxplot.png', colors)
    
    # 2) Horizontal bar chart for meta origins
    meta_protocol = {'meta': meta_results['meta_protocol_waves']}
    fig, axs = plt.subplots(1, 4, figsize=(20, 10), sharey=True)
    create_horizontal_bar_chart(axs, meta_protocol['meta'], stages, 
                               meta_results['master_region_list'], 'meta')
    plt.tight_layout()
    plt.savefig('origins_meta.png')
    print("Saved origins chart for meta as 'origins_meta.png'")
    plt.close()
