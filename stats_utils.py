import numpy as np
import logging # Added
from scipy import stats as scipy_stats
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import itertools


def perform_friedman_test(*args):
    """
    Perform the Friedman test for repeated measures.

    Args:
        *args: Multiple lists or arrays, each representing measurements for a
               different condition/stage, with subjects matched across lists.
               Example: perform_friedman_test(stage1_scores, stage2_scores, stage3_scores)
               where stage1_scores[i] corresponds to the same subject as stage2_scores[i].

    Returns:
        A dictionary containing the test results or None if the test cannot be performed.
    """
    # Ensure all groups have the same number of subjects
    if not args:
        logging.error("No data provided for Friedman test.")
        return None
    n_subjects = len(args[0])
    if not all(len(arg) == n_subjects for arg in args):
        logging.error("All groups must have the same number of subjects for Friedman test.")
        # Consider handling missing data more gracefully if needed (e.g., imputation or subject removal)
        return None
    if n_subjects < 2 or len(args) < 2:
        logging.error("Friedman test requires at least 2 subjects and 2 conditions.")
        return None

    try:
        statistic, p_value = friedmanchisquare(*args)
        return {
            'Test': 'Friedman Test',
            'Metric': 'Involvement',
            'Statistic': statistic,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        }
    except Exception as e:
        logging.error(f"Error performing Friedman test for involvement: {str(e)}")
        return None


def perform_wilcoxon_posthoc(paired_data, stage_names):
    """
    Perform post-hoc Wilcoxon signed-rank tests with FDR correction.

    Args:
        paired_data: A list of lists/arrays, where each inner list contains the
                     measurements for one stage, ordered by subject.
                     Example: [[subj1_pre, subj2_pre, ...], [subj1_post, subj2_post, ...]]
        stage_names: A list of names corresponding to the stages in paired_data.
                      Example: ['pre', 'post']

    Returns:
        A list of dictionaries, each containing results for a pairwise comparison.
    """
    if len(paired_data) != len(stage_names) or len(paired_data) < 2:
        logging.error("Need at least two stages with corresponding names for Wilcoxon post-hoc.")
        return []

    pairwise_results_raw = []
    comparisons = list(itertools.combinations(range(len(stage_names)), 2))

    for i, j in comparisons:
        stage1_name = stage_names[i]
        stage2_name = stage_names[j]
        data1 = paired_data[i]
        data2 = paired_data[j]

        # Ensure equal length and handle potential all-zero differences
        if len(data1) != len(data2):
            logging.warning(f"Skipping Wilcoxon for {stage1_name} vs {stage2_name} due to unequal lengths.")
            continue
        diff = np.array(data1) - np.array(data2)
        if np.all(diff == 0):
             # Wilcoxon raises error if all differences are zero
             logging.info(f"Skipping Wilcoxon for {stage1_name} vs {stage2_name}: all differences are zero.")
             # Assign non-significant p-value or handle as needed
             stat = np.nan
             p_val = 1.0
        else:
            try:
                stat, p_val = wilcoxon(data1, data2, alternative='two-sided', zero_method='pratt') # 'pratt' handles zeros
            except ValueError as e:
                 # Handle cases like fewer than required data points after removing zeros/ties
                 logging.warning(f"Wilcoxon failed for {stage1_name} vs {stage2_name}: {e}. Assigning p=1.0")
                 stat = np.nan
                 p_val = 1.0
            except Exception as e:
                logging.error(f"Error performing Wilcoxon for {stage1_name} vs {stage2_name}: {str(e)}")
                continue # Skip this comparison

        pairwise_results_raw.append({
            'stage1': stage1_name,
            'stage2': stage2_name,
            'statistic': stat,
            'p_value_raw': p_val
        })

    if not pairwise_results_raw:
        return []

    # Apply FDR correction
    pvals_raw = [res['p_value_raw'] for res in pairwise_results_raw]
    reject, pvals_corrected, _, _ = multipletests(pvals_raw, method='fdr_bh', alpha=0.05)

    # Format final results
    final_results = []
    for i, raw_res in enumerate(pairwise_results_raw):
        final_results.append({
            'Test': 'Wilcoxon Signed-Rank',
            'Metric': f'Involvement: {raw_res["stage1"]} vs {raw_res["stage2"]}',
            'Statistic': raw_res['statistic'],
            'P_Value': pvals_corrected[i],
            'Significant': reject[i]
        })

    return final_results


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
        logging.error(f"Error performing Chi2/Fisher test: {str(e)}")
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
        logging.error(f"Error performing statistical test for origin distribution: {str(e)}")

    return stats_results
