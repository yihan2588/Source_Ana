import os
import pandas as pd
import numpy as np
from collections import Counter
from utils import collect_wave_level_data

def save_statistical_results(analysis_results, output_dir="."):
    """Save statistical results for individual protocols to CSV files"""
    print("\n=== Saving Statistical Results for Individual Protocols ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save involvement statistics
    involvement_stats_data = []
    for protocol, results in analysis_results.items():
        for stage, stats_data in results['involvement_stats'].items():
            involvement_stats_data.append({
                'Protocol': protocol,
                'Stage': stage,
                'Mean_Involvement': stats_data['mean'],
                'Median_Involvement': stats_data['median'],
                'Std_Involvement': stats_data['std'],
                'Count': stats_data['count']
            })
    
    involvement_stats_df = pd.DataFrame(involvement_stats_data)
    involvement_stats_file = os.path.join(output_dir, "involvement_statistics.csv")
    involvement_stats_df.to_csv(involvement_stats_file, index=False)
    print(f"Saved involvement statistics to {involvement_stats_file}")
    
    # Save origin statistics for each protocol
    origin_files = []
    for protocol, results in analysis_results.items():
        origin_data = []
        for stage, counts in results['origin_data'].items():
            total_waves = len(results['involvement_data'][stage])
            for region, count in counts.items():
                percentage = (count / total_waves * 100) if total_waves > 0 else 0
                origin_data.append({
                    'Protocol': protocol,
                    'Stage': stage,
                    'Region': region,
                    'Count': count,
                    'Total_Waves': total_waves,
                    'Percentage': percentage
                })
        if origin_data:
            origin_df = pd.DataFrame(origin_data)
            origin_file = os.path.join(output_dir, f"{protocol}_origin_statistics.csv")
            origin_df.to_csv(origin_file, index=False)
            origin_files.append(origin_file)
            print(f"Saved origin statistics for {protocol} to {origin_file}")
    
    # Save statistical test results
    all_test_results = []
    for protocol, results in analysis_results.items():
        # Add protocol information to each test result
        for result in results.get('involvement_test_results', []):
            result_copy = result.copy()
            result_copy['Protocol'] = protocol
            all_test_results.append(result_copy)
        
        for result in results.get('origin_test_results', []):
            result_copy = result.copy()
            result_copy['Protocol'] = protocol
            all_test_results.append(result_copy)
    
    stats_file = None
    if all_test_results:
        stats_df = pd.DataFrame(all_test_results)
        stats_file = os.path.join(output_dir, "statistical_test_results.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"Saved statistical test results to {stats_file}")
    
    return {
        'involvement_stats': involvement_stats_file,
        'origin_stats': origin_files,
        'test_results': stats_file
    }

def save_meta_results(meta_results, output_dir="."):
    """Save meta protocol statistical results to CSV files"""
    print("\n=== Saving Meta Protocol Statistical Results ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save meta involvement statistics
    meta_involvement_stats = meta_results['meta_involvement_stats']
    meta_involvement_data = []
    for stage, stats_data in meta_involvement_stats.items():
        meta_involvement_data.append({
            'Stage': stage,
            'Mean_Involvement': stats_data['mean'],
            'Median_Involvement': stats_data['median'],
            'Std_Involvement': stats_data['std'],
            'Count': stats_data['count']
        })
    meta_involvement_df = pd.DataFrame(meta_involvement_data)
    meta_involvement_file = os.path.join(output_dir, "meta_involvement_statistics.csv")
    meta_involvement_df.to_csv(meta_involvement_file, index=False)
    print(f"Saved meta involvement statistics to {meta_involvement_file}")
    
    # Save meta origin statistics
    meta_origin_data = meta_results['meta_origin_data']
    total_waves_by_stage = {st: meta_results['meta_involvement_stats'][st]['count'] for st in ['pre', 'early', 'late', 'post']}
    meta_origin_list = []
    for stage, counts in meta_origin_data.items():
        for region, count in counts.items():
            percentage = (count / total_waves_by_stage[stage] * 100) if total_waves_by_stage[stage] > 0 else 0
            meta_origin_list.append({
                'Stage': stage,
                'Region': region,
                'Count': count,
                'Total_Waves': total_waves_by_stage[stage],
                'Percentage': percentage
            })
    if meta_origin_list:
        meta_origin_df = pd.DataFrame(meta_origin_list)
        meta_origin_file = os.path.join(output_dir, "meta_origin_statistics.csv")
        meta_origin_df.to_csv(meta_origin_file, index=False)
        print(f"Saved meta origin statistics to {meta_origin_file}")
    else:
        meta_origin_file = None
    
    # Save meta-level statistical test results
    meta_stats_results = meta_results.get('meta_stats_results', [])
    meta_stats_file = None
    if meta_stats_results:
        meta_stats_df = pd.DataFrame(meta_stats_results)
        meta_stats_file = os.path.join(output_dir, "meta_statistical_test_results.csv")
        meta_stats_df.to_csv(meta_stats_file, index=False)
        print(f"Saved meta statistical test results to {meta_stats_file}")
    
    return {
        'meta_involvement_stats': meta_involvement_file,
        'meta_origin_stats': meta_origin_file,
        'meta_test_results': meta_stats_file
    }
