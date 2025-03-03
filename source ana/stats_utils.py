import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

def perform_involvement_tests(involvement_data):
    """
    Perform statistical tests on involvement data across stages.
    
    Args:
        involvement_data: Dictionary with stages as keys and lists of involvement percentages as values
        
    Returns:
        List of dictionaries containing test results
    """
    stats_results = []
    valid_stages = [stage for stage in involvement_data.keys() if involvement_data[stage]]
    
    if len(valid_stages) >= 2:
        groups = [involvement_data[st] for st in valid_stages]
        try:
            statistic, p_value = scipy_stats.kruskal(*groups)
            stats_results.append({
                'Test': 'Kruskal-Wallis',
                'Metric': 'Involvement',
                'Statistic': statistic,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
            
            if p_value < 0.05:
                pairwise_results = []
                for i, stage1 in enumerate(valid_stages):
                    for stage2 in valid_stages[i+1:]:
                        if involvement_data[stage1] and involvement_data[stage2]:
                            u_stat, p_val = scipy_stats.mannwhitneyu(
                                involvement_data[stage1],
                                involvement_data[stage2],
                                alternative='two-sided'
                            )
                            pairwise_results.append((stage1, stage2, u_stat, p_val))
                
                if pairwise_results:
                    pvals = [r[3] for r in pairwise_results]
                    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
                    
                    for i, (st1, st2, u_stat, _) in enumerate(pairwise_results):
                        stats_results.append({
                            'Test': 'Mann-Whitney U',
                            'Metric': f'Involvement: {st1} vs {st2}',
                            'Statistic': u_stat,
                            'P_Value': pvals_corrected[i],
                            'Significant': reject[i]
                        })
        except Exception as e:
            print(f"Error performing statistical test for involvement: {str(e)}")
    
    return stats_results

def perform_origin_distribution_tests(origin_data, master_region_list, top_n=10):
    """
    Perform statistical tests on origin distribution across stages.
    
    Args:
        origin_data: Dictionary with stages as keys and Counter objects of region counts as values
        master_region_list: List of regions in order of frequency
        top_n: Number of top regions to include in the analysis
        
    Returns:
        List of dictionaries containing test results
    """
    stats_results = []
    valid_stages = [stage for stage in origin_data.keys() if origin_data[stage]]
    
    if len(master_region_list) > 0 and len(valid_stages) >= 2:
        top_regions = master_region_list[:top_n]
        contingency_table = []
        
        for stage in valid_stages:
            row = [origin_data[stage].get(region, 0) for region in top_regions]
            contingency_table.append(row)
        
        try:
            contingency_table = np.array(contingency_table)
            
            if contingency_table.shape[1] <= 2 and contingency_table.shape[0] <= 2:
                oddsratio, p_value = scipy_stats.fisher_exact(contingency_table)
                stats_results.append({
                    'Test': "Fisher's Exact Test",
                    'Metric': 'Origin Distribution',
                    'Statistic': oddsratio,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
            else:
                expected = scipy_stats.chi2_contingency(contingency_table)[3]
                valid_for_chi2 = np.all(expected >= 1) and np.sum(expected < 5) <= expected.size * 0.2
                
                if valid_for_chi2:
                    chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency_table)
                    test_name = 'Chi-Square'
                else:
                    chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency_table, correction=True)
                    test_name = 'G-test with Williams\' correction'
                
                stats_results.append({
                    'Test': test_name,
                    'Metric': 'Origin Distribution',
                    'Statistic': chi2,
                    'DF': dof,
                    'P_Value': p,
                    'Significant': p < 0.05
                })
        except Exception as e:
            print(f"Error performing statistical test for origin distribution: {str(e)}")
    
    return stats_results
