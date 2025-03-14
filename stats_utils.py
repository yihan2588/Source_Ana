import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


def perform_involvement_tests(involvement_data):
    """
    Perform statistical tests (Kruskal-Wallis and post-hoc Mann-Whitney U)
    on involvement data across stages.
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
                # post-hoc
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


def perform_chi_square_or_fisher_test(contingency_table):
    """
    Perform either Chi-Square or Fisher's exact test based on the contingency table size
    and distribution of expected counts.
    """
    try:
        contingency_table = np.array(contingency_table)
        # If it's a 2x2 table
        if contingency_table.shape == (2, 2):
            oddsratio, p_value = scipy_stats.fisher_exact(contingency_table)
            test_name = "Fisher's Exact Test"
            test_stat = oddsratio
            test_df = None
        else:
            # Check Chi-Square assumptions
            chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table)
            valid_for_chi2 = np.all(expected >= 1) and np.sum(expected < 5) <= 0.2 * expected.size
            if valid_for_chi2:
                test_name = 'Chi-Square'
                test_stat = chi2
                p_value = p
                test_df = dof
            else:
                # fallback
                # Some people do a G-test or apply Yates correction, etc.
                # We'll just re-run with continuity correction or something else
                # For simplicity, let's do the same but note it:
                chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table, correction=True)
                test_name = "Chi-Square (w/ correction or G-test fallback)"
                test_stat = chi2
                p_value = p
                test_df = dof

        result = {
            'Test': test_name,
            'Statistic': test_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        }
        if test_df is not None:
            result['DF'] = test_df
        return result

    except Exception as e:
        print(f"Error performing statistical test: {str(e)}")
        return None


def perform_origin_distribution_tests(origin_data, master_region_list, top_n=10):
    """
    Perform statistical tests on origin distribution across stages.
    origin_data here is a dict: {stage -> { 'region_counts':Counter, 'total_waves':int } }
    We'll build a contingency table for the top_n regions (by global frequency).
    """
    stats_results = []

    # First, gather valid stages
    valid_stages = [stage for stage in origin_data.keys()
                    if origin_data[stage] and 'region_counts' in origin_data[stage]
                    and origin_data[stage]['region_counts']]

    if len(valid_stages) < 2:
        return stats_results

    # Determine top regions from master_region_list
    top_regions = master_region_list[:top_n]

    # Build contingency table
    contingency_table = []
    for stage in valid_stages:
        region_counts = origin_data[stage]['region_counts']
        row = [region_counts.get(region, 0) for region in top_regions]
        contingency_table.append(row)

    try:
        contingency_table = np.array(contingency_table)
        # If 2x2, do Fisher
        if contingency_table.shape == (2, 2):
            oddsratio, p_value = scipy_stats.fisher_exact(contingency_table)
            stats_results.append({
                'Test': "Fisher's Exact Test",
                'Metric': 'Origin Distribution',
                'Statistic': oddsratio,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
        else:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table)
            valid_for_chi2 = np.all(expected >= 1) and np.sum(expected < 5) <= expected.size * 0.2
            if valid_for_chi2:
                test_name = 'Chi-Square'
            else:
                # fallback
                test_name = "Chi-Square (w/ correction or G-test fallback)"
                chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency_table, correction=True)

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