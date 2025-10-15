"""Statistical utilities for EEG involvement analysis."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


STAGE_ORDER_3 = ["pre", "stim", "post"]
STAGE_ORDER_4 = ["pre", "early", "late", "post"]


def _resolve_stage_order(stages: Iterable[str]) -> List[str]:
    stages_lower = {stage.lower() for stage in stages}
    if stages_lower == set(STAGE_ORDER_4):
        return STAGE_ORDER_4.copy()
    if stages_lower == set(STAGE_ORDER_3):
        return STAGE_ORDER_3.copy()
    return sorted(stages_lower)


def create_wave_dataframe(
    results_by_treatment_group: Dict[str, Dict[str, Dict[str, Dict[str, List[dict]]]]],
    subject_thresholds: Dict[str, float],
) -> pd.DataFrame:
    """Flatten nested wave results into a DataFrame."""
    rows: List[dict] = []
    for group, subjects in results_by_treatment_group.items():
        for subject_id, protocols in subjects.items():
            threshold = subject_thresholds.get(subject_id, np.nan)
            for protocol, stages in protocols.items():
                for stage, waves in stages.items():
                    for wave in waves:
                        rows.append(
                            {
                                "Subject_ID": subject_id,
                                "Treatment_Group": group,
                                "Protocol": protocol,
                                "Stage": stage,
                                "Wave_Name": wave.get("wave_name", ""),
                                "Involvement_Percentage": wave.get("involvement_percentage", np.nan),
                                "Involvement_Count": wave.get("involvement_count", np.nan),
                                "Threshold": threshold,
                                "Global_Max_Value": wave.get("global_max_value", np.nan),
                                "Global_Max_Time": wave.get("global_max_time", np.nan),
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Protocol_Number"] = (
        df["Protocol"].str.extract(r"proto(\d+)", expand=False).astype(float)
    )
    return df


def create_subject_averaged_data(wave_df: pd.DataFrame, *, by_protocol: bool = False) -> pd.DataFrame:
    """Compute subject-level mean involvement per stage (optionally per protocol)."""
    if wave_df.empty:
        return pd.DataFrame(
            columns=[
                "Subject_ID",
                "Treatment_Group",
                "Stage",
                "Mean_Involvement",
                "Std_Involvement",
                "N_Waves",
            ]
        )

    grouping = ["Subject_ID", "Treatment_Group", "Stage"]
    if by_protocol:
        grouping.insert(2, "Protocol")

    subject_means = (
        wave_df.groupby(grouping)
        .agg({"Involvement_Percentage": ["mean", "std", "count"], "Threshold": "first"})
        .reset_index()
    )
    if by_protocol:
        subject_means.columns = [
            "Subject_ID",
            "Treatment_Group",
            "Protocol",
            "Stage",
            "Mean_Involvement",
            "Std_Involvement",
            "N_Waves",
            "Threshold",
        ]
    else:
        subject_means.columns = [
            "Subject_ID",
            "Treatment_Group",
            "Stage",
            "Mean_Involvement",
            "Std_Involvement",
            "N_Waves",
            "Threshold",
        ]
    return subject_means


def _summarize_stage_statistics(subject_means_df: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    summary: Dict[str, Dict[str, dict]] = {}
    for group, group_df in subject_means_df.groupby("Treatment_Group"):
        group_summary: Dict[str, dict] = {}
        for stage, stage_df in group_df.groupby("Stage"):
            values = stage_df["Mean_Involvement"].to_numpy(dtype=float)
            if values.size:
                group_summary[stage] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "count": int(values.size),
                }
            else:
                group_summary[stage] = {"mean": 0.0, "std": 0.0, "count": 0}
        summary[group] = group_summary
    return summary


def _compute_within_group_tests(
    subject_means_df: pd.DataFrame,
    *,
    stage_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, dict]], List[dict]]:
    comparisons: Dict[str, Dict[str, dict]] = {}
    rows: List[dict] = []
    groups = sorted(subject_means_df["Treatment_Group"].unique())
    stages = stage_order or _resolve_stage_order(subject_means_df["Stage"].unique())

    for group in groups:
        group_df = subject_means_df[subject_means_df["Treatment_Group"] == group]
        comparisons[group] = {}
        subject_stage = {
            subject: stage_df.set_index("Stage")["Mean_Involvement"].to_dict()
            for subject, stage_df in group_df.groupby("Subject_ID")
        }
        for i, stage1 in enumerate(stages):
            for stage2 in stages[i + 1 :]:
                paired_stage1: List[float] = []
                paired_stage2: List[float] = []
                paired_ids: List[str] = []
                for subject_id, values in subject_stage.items():
                    if stage1 in values and stage2 in values:
                        paired_stage1.append(values[stage1])
                        paired_stage2.append(values[stage2])
                        paired_ids.append(subject_id)
                if len(paired_stage1) < 3:
                    continue
                try:
                    stat, p_value = scipy_stats.wilcoxon(paired_stage1, paired_stage2)
                except ValueError as exc:
                    logging.warning(
                        "Wilcoxon test failed for %s %s vs %s: %s", group, stage1, stage2, exc
                    )
                    continue
                paired_stage1_arr = np.array(paired_stage1, dtype=float)
                differences = np.array(paired_stage2, dtype=float) - paired_stage1_arr
                effect_size = 0.0
                if np.std(differences, ddof=1) > 0:
                    effect_size = float(np.mean(differences) / np.std(differences, ddof=1))
                comparison_key = f"{stage1}_vs_{stage2}"
                summary = {
                    "test": "Wilcoxon Signed-Rank",
                    "n_subjects": len(paired_stage1),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),
                    "effect_size": effect_size,
                    "stage1_mean": float(np.mean(paired_stage1_arr)),
                    "stage2_mean": float(np.mean(paired_stage2)),
                    "stage1_std": float(np.std(paired_stage1_arr, ddof=0)),
                    "stage2_std": float(np.std(paired_stage2, ddof=0)),
                    "mean_difference": float(np.mean(differences)),
                    "subjects_analyzed": paired_ids,
                }
                comparisons[group][comparison_key] = summary
                rows.append(
                    {
                        "Analysis_Level": "Overall",
                        "Protocol": "All",
                        "Test_Type": "Within_Group",
                        "Group": group,
                        "Stage1": stage1,
                        "Stage2": stage2,
                        "Comparison": f"{stage2} vs {stage1}",
                        "Test_Method": "Wilcoxon Signed-Rank",
                        "N_Subjects": len(paired_stage1),
                        "Mean1": summary["stage1_mean"],
                        "Std1": summary["stage1_std"],
                        "Mean2": summary["stage2_mean"],
                        "Std2": summary["stage2_std"],
                        "Mean_Difference": summary["mean_difference"],
                        "Statistic": summary["statistic"],
                        "P_Value": summary["p_value"],
                        "Significant": summary["significant"],
                        "Effect_Size": summary["effect_size"],
                    }
                )
    return comparisons, rows


def _compute_between_group_tests(
    subject_means_df: pd.DataFrame,
    *,
    stage_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, dict], List[dict]]:
    comparisons: Dict[str, dict] = {}
    rows: List[dict] = []
    if subject_means_df.empty:
        return comparisons, rows

    stages = stage_order or _resolve_stage_order(subject_means_df["Stage"].unique())
    groups = sorted(subject_means_df["Treatment_Group"].unique())
    if "Active" not in groups or "SHAM" not in groups:
        return comparisons, rows

    for stage in stages:
        stage_df = subject_means_df[subject_means_df["Stage"] == stage]
        active_values = stage_df[stage_df["Treatment_Group"] == "Active"]["Mean_Involvement"].to_numpy(dtype=float)
        sham_values = stage_df[stage_df["Treatment_Group"] == "SHAM"]["Mean_Involvement"].to_numpy(dtype=float)
        if len(active_values) < 3 or len(sham_values) < 3:
            continue
        stat, p_value = scipy_stats.mannwhitneyu(active_values, sham_values, alternative="two-sided")
        pooled_std = 0.0
        if active_values.size + sham_values.size > 2:
            pooled_std = np.sqrt(
                (
                    (active_values.size - 1) * np.var(active_values, ddof=1)
                    + (sham_values.size - 1) * np.var(sham_values, ddof=1)
                )
                / (active_values.size + sham_values.size - 2)
            )
        effect_size = 0.0
        if pooled_std > 0:
            effect_size = float((np.mean(active_values) - np.mean(sham_values)) / pooled_std)
        summary = {
            "test": "Mann-Whitney U",
            "n_active": int(active_values.size),
            "n_sham": int(sham_values.size),
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "effect_size": effect_size,
            "active_mean": float(np.mean(active_values)),
            "active_std": float(np.std(active_values, ddof=0)),
            "sham_mean": float(np.mean(sham_values)),
            "sham_std": float(np.std(sham_values, ddof=0)),
        }
        comparisons[stage] = summary
        rows.append(
            {
                "Analysis_Level": "Overall",
                "Protocol": "All",
                "Test_Type": "Between_Group",
                "Group": "Active_vs_SHAM",
                "Stage1": stage,
                "Stage2": None,
                "Comparison": f"Active vs SHAM ({stage})",
                "Test_Method": "Mann-Whitney U",
                "N_Subjects": f"{summary['n_active']},{summary['n_sham']}",
                "Mean1": summary["active_mean"],
                "Std1": summary["active_std"],
                "Mean2": summary["sham_mean"],
                "Std2": summary["sham_std"],
                "Mean_Difference": summary["active_mean"] - summary["sham_mean"],
                "Statistic": summary["statistic"],
                "P_Value": summary["p_value"],
                "Significant": summary["significant"],
                "Effect_Size": summary["effect_size"],
            }
        )
    return comparisons, rows


def generate_optimal_statistics(subject_means_df: pd.DataFrame) -> Dict[str, object]:
    """Produce dictionaries describing within-group and between-group tests."""
    stage_order = _resolve_stage_order(subject_means_df["Stage"].unique()) if not subject_means_df.empty else []
    stage_stats = _summarize_stage_statistics(subject_means_df)
    within_dict, _ = _compute_within_group_tests(subject_means_df, stage_order=stage_order)
    between_dict, _ = _compute_between_group_tests(subject_means_df, stage_order=stage_order)
    return {
        "stage_statistics": stage_stats,
        "within_group": within_dict,
        "between_group": between_dict,
    }


def compute_percentage_changes(
    subject_means_df: pd.DataFrame,
    *,
    stage_order: Optional[List[str]] = None,
) -> Dict[str, Dict[str, dict]]:
    """Calculate percentage changes (Stim-Pre and Post-Pre where available)."""
    changes: Dict[str, Dict[str, dict]] = {}
    if subject_means_df.empty:
        return changes
    stages = stage_order or _resolve_stage_order(subject_means_df["Stage"].unique())
    if "pre" not in stages:
        return changes
    comparison_map = []
    if "stim" in stages:
        comparison_map.append(("Stim-Pre", "stim"))
    if "post" in stages:
        comparison_map.append(("Post-Pre", "post"))
    if not comparison_map:
        return changes

    for group, group_df in subject_means_df.groupby("Treatment_Group"):
        group_changes: Dict[str, dict] = {}
        pre_df = group_df[group_df["Stage"] == "pre"]
        if pre_df.empty:
            continue
        pre_mean = float(pre_df["Mean_Involvement"].mean())
        pre_std = float(pre_df["Mean_Involvement"].std(ddof=0))
        pre_count = int(pre_df.shape[0])
        if pre_mean <= 0:
            continue
        for label, target_stage in comparison_map:
            target_df = group_df[group_df["Stage"] == target_stage]
            if target_df.empty:
                continue
            target_mean = float(target_df["Mean_Involvement"].mean())
            target_std = float(target_df["Mean_Involvement"].std(ddof=0))
            target_count = int(target_df.shape[0])
            pct_change = ((target_mean - pre_mean) / pre_mean) * 100.0
            pct_std = 0.0
            if pre_mean:
                pct_std = 100.0 * np.sqrt(
                    (target_std / pre_mean) ** 2
                    + ((target_mean - pre_mean) * pre_std / (pre_mean ** 2)) ** 2
                )
            group_changes[label] = {
                "percentage_change_mean": pct_change,
                "percentage_change_std": pct_std,
                "count": min(pre_count, target_count),
                "pre_mean": pre_mean,
                "target_mean": target_mean,
            }
        if group_changes:
            changes[group] = group_changes
    return changes


def run_comprehensive_tests(wave_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the standalone statistical analysis on the supplied wave data."""
    all_rows: List[dict] = []
    overall_subject_means = create_subject_averaged_data(wave_df, by_protocol=False)
    _, within_rows = _compute_within_group_tests(overall_subject_means)
    _, between_rows = _compute_between_group_tests(overall_subject_means)
    for row in within_rows:
        row = row.copy()
        row["Analysis_Level"] = "Overall"
        row["Protocol"] = "All"
        all_rows.append(row)
    for row in between_rows:
        row = row.copy()
        row["Analysis_Level"] = "Overall"
        row["Protocol"] = "All"
        all_rows.append(row)

    protocols = sorted(wave_df["Protocol"].dropna().unique())
    for protocol in protocols:
        protocol_df = wave_df[wave_df["Protocol"] == protocol]
        subject_means = create_subject_averaged_data(protocol_df, by_protocol=True)
        if subject_means.empty:
            continue
        within_rows_proto = _compute_within_group_tests(subject_means)[1]
        between_rows_proto = _compute_between_group_tests(subject_means)[1]
        # The helper functions label everything as Overall; adjust metadata here.
        for row in within_rows_proto:
            new_row = row.copy()
            new_row["Analysis_Level"] = "Proto_Specific"
            new_row["Protocol"] = protocol
            all_rows.append(new_row)
        for row in between_rows_proto:
            new_row = row.copy()
            new_row["Analysis_Level"] = "Proto_Specific"
            new_row["Protocol"] = protocol
            all_rows.append(new_row)
    if not all_rows:
        return pd.DataFrame()
    results_df = pd.DataFrame(all_rows)
    numeric_cols = [
        "Mean1",
        "Std1",
        "Mean2",
        "Std2",
        "Mean_Difference",
        "Statistic",
        "P_Value",
        "Effect_Size",
    ]
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(float).round(6)
    return results_df.sort_values(
        ["Analysis_Level", "Protocol", "Test_Type", "Group", "Stage1", "Stage2"],
        na_position="last",
    ).reset_index(drop=True)


def optimal_results_to_dataframe(stat_summary: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """Convert optimal statistics summary into a flat table."""
    rows: List[dict] = []
    within = stat_summary.get("within_group", {})
    for group, comparisons in within.items():
        for comparison, payload in comparisons.items():
            rows.append(
                {
                    "Analysis_Type": "Within_Group",
                    "Group": group,
                    "Comparison": comparison,
                    "Test": payload.get("test"),
                    "N_Subjects": payload.get("n_subjects"),
                    "Statistic": payload.get("statistic"),
                    "P_Value": payload.get("p_value"),
                    "Significant": payload.get("significant"),
                    "Effect_Size": payload.get("effect_size"),
                    "Stage1_Mean": payload.get("stage1_mean"),
                    "Stage2_Mean": payload.get("stage2_mean"),
                    "Stage1_Std": payload.get("stage1_std"),
                    "Stage2_Std": payload.get("stage2_std"),
                    "Mean_Difference": payload.get("mean_difference"),
                }
            )

    between = stat_summary.get("between_group", {})
    for stage, payload in between.items():
        rows.append(
            {
                "Analysis_Type": "Between_Group",
                "Group": stage,
                "Comparison": f"Active vs SHAM ({stage})",
                "Test": payload.get("test"),
                "N_Subjects": f"{payload.get('n_active', 0)},{payload.get('n_sham', 0)}",
                "Statistic": payload.get("statistic"),
                "P_Value": payload.get("p_value"),
                "Significant": payload.get("significant"),
                "Effect_Size": payload.get("effect_size"),
                "Active_Mean": payload.get("active_mean"),
                "Active_Std": payload.get("active_std"),
                "SHAM_Mean": payload.get("sham_mean"),
                "SHAM_Std": payload.get("sham_std"),
            }
        )

    return pd.DataFrame(rows)


def percentage_changes_to_dataframe(changes: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    """Convert percentage change dictionary to a DataFrame."""
    rows: List[dict] = []
    for group, comparisons in changes.items():
        for label, payload in comparisons.items():
            rows.append(
                {
                    "Treatment_Group": group,
                    "Comparison": label,
                    "Percentage_Change_Mean": payload.get("percentage_change_mean"),
                    "Percentage_Change_Std": payload.get("percentage_change_std"),
                    "Count": payload.get("count"),
                    "Pre_Mean": payload.get("pre_mean"),
                    "Target_Mean": payload.get("target_mean"),
                }
            )
    return pd.DataFrame(rows)


def stage_statistics_to_dataframe(stage_stats: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    """Flatten stage statistics into a table."""
    rows: List[dict] = []
    for group, stages in stage_stats.items():
        for stage, payload in stages.items():
            rows.append(
                {
                    "Treatment_Group": group,
                    "Stage": stage,
                    "Mean": payload.get("mean"),
                    "Std": payload.get("std"),
                    "Count": payload.get("count"),
                }
            )
    return pd.DataFrame(rows)
