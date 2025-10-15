"""Visualization helpers for EEG involvement analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks

from involvement import extract_region_name


def _parse_protocol_and_stage(wave_name: str) -> tuple[str, str]:
    protocol = "unknown"
    stage = "unknown"
    if not wave_name:
        return protocol, stage
    wave_name_lower = wave_name.lower()
    for token in ["proto1", "proto2", "proto3", "proto4", "proto5", "proto6", "proto7", "proto8"]:
        if token in wave_name_lower:
            protocol = token
            break
    for token in ["pre", "early", "late", "post", "stim"]:
        if token in wave_name_lower:
            stage = token
            break
    return protocol, stage


def visualize_region_time_series(
    wave_data: Dict[str, object],
    csv_file: Path,
    *,
    source_dir: Optional[Path] = None,
) -> bool:
    """Plot averaged region time series for a single wave."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Unable to read %s for plotting: %s", csv_file, exc)
        return False

    if "Time" not in df.columns:
        logging.warning("Skipping visualization for %s (missing Time column)", csv_file)
        return False

    numeric_cols = [col for col in df.columns[1:] if not str(col).startswith("Unnamed")]
    if not numeric_cols:
        logging.warning("Skipping visualization for %s (no numeric columns)", csv_file)
        return False

    time_points = np.array([float(col) for col in numeric_cols]) * 1000.0
    data = np.abs(df[numeric_cols].to_numpy(dtype=float))
    voxel_names = df.iloc[:, 0].astype(str).to_numpy()

    window_mask = (time_points >= wave_data.get("window", (-50, 50))[0]) & (
        time_points <= wave_data.get("window", (-50, 50))[1]
    )
    if not np.any(window_mask):
        window_mask = np.ones_like(time_points, dtype=bool)

    window_times = time_points[window_mask]
    window_data = data[:, window_mask]

    region_map: Dict[str, list[int]] = {}
    for idx, voxel_name in enumerate(voxel_names):
        region = extract_region_name(voxel_name)
        region_map.setdefault(region, []).append(idx)

    if not region_map:
        logging.warning("No regions detected for %s", csv_file)
        return False

    protocol, stage = _parse_protocol_and_stage(str(wave_data.get("wave_name", csv_file.stem)))
    threshold = float(wave_data.get("threshold", 0.0))

    base_output_dir = Path("results")
    if source_dir:
        base_output_dir = Path(source_dir) / "results"
    output_dir = base_output_dir / "region_plots" / protocol / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    region_curves: Dict[str, np.ndarray] = {}
    for region, indices in region_map.items():
        region_series = window_data[indices].mean(axis=0)
        region_curves[region] = region_series
        plt.plot(window_times, region_series, label=region)

        for voxel_idx in indices:
            voxel_series = window_data[voxel_idx]
            peaks, _ = find_peaks(voxel_series, height=threshold)
            if peaks.size == 0:
                continue
            peak_times = window_times[peaks]
            peak_vals = np.interp(peak_times, window_times, region_series)
            plt.scatter(peak_times, peak_vals, color="steelblue", s=12, alpha=0.5, label="_nolegend_")

    origins = wave_data.get("origins")
    if isinstance(origins, pd.DataFrame) and not origins.empty:
        for _, row in origins.iterrows():
            region = row.get("region")
            peak_time = float(row.get("peak_time", 0.0))
            y_value = threshold
            if region in region_curves:
                y_value = float(np.interp(peak_time, window_times, region_curves[region]))
            plt.scatter(peak_time, y_value, color="red", marker="*", s=80, label="_nolegend_")

    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.title(f"Region Time Series – {wave_data.get('wave_name', csv_file.stem)}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    output_path = output_dir / f"{wave_data.get('wave_name', csv_file.stem)}_region_time_series.png"
    plt.savefig(output_path, dpi=300)

    csv_output = csv_file.with_name(f"{csv_file.stem}_region_time_series.png")
    plt.savefig(csv_output, dpi=300)
    plt.close()
    logging.info("Region time series saved to %s", output_path)
    return True


def create_summary_visualizations(
    subject_means_df: pd.DataFrame,
    statistical_results: Dict[str, Dict[str, object]],
    output_dir: Path,
) -> Optional[Path]:
    """Generate summary plots for overall involvement statistics."""
    if subject_means_df.empty:
        logging.warning("No subject means available for visualization")
        return None

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ordered_stages = ["pre", "stim", "post"]
    subject_means_df = subject_means_df.copy()
    subject_means_df["Stage"] = pd.Categorical(
        subject_means_df["Stage"], categories=ordered_stages, ordered=True
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=subject_means_df,
        x="Stage",
        y="Mean_Involvement",
        hue="Treatment_Group",
        palette={"Active": "salmon", "SHAM": "skyblue"},
        showmeans=True,
    )
    sns.stripplot(
        data=subject_means_df,
        x="Stage",
        y="Mean_Involvement",
        hue="Treatment_Group",
        dodge=True,
        palette={"Active": "darkred", "SHAM": "darkblue"},
        alpha=0.6,
        linewidth=0,
    )
    plt.title("Involvement by Stage and Treatment Group")
    plt.xlabel("Stage")
    plt.ylabel("Mean Involvement (%)")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Treatment Group")
    plt.tight_layout()
    boxplot_path = plots_dir / "involvement_by_group_stage.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()

    active_data = subject_means_df[subject_means_df["Treatment_Group"] == "Active"]
    pivot = active_data.pivot(index="Subject_ID", columns="Stage", values="Mean_Involvement")
    if {"pre", "stim"}.issubset(pivot.columns):
        plt.figure(figsize=(8, 6))
        for _, row in pivot.dropna(subset=["pre", "stim"]).iterrows():
            plt.plot([0, 1], [row["pre"], row["stim"]], marker="o", color="salmon", alpha=0.7)
        mean_pre = pivot["pre"].mean()
        mean_stim = pivot["stim"].mean()
        plt.plot([0, 1], [mean_pre, mean_stim], marker="o", color="darkred", linewidth=3, label="Group mean")
        plt.xticks([0, 1], ["Pre", "Stim"])
        plt.ylabel("Mean Involvement (%)")
        plt.title("Active Group: Pre vs Stim")
        plt.grid(True, alpha=0.3)
        plt.legend()
        paired_path = plots_dir / "active_pre_vs_stim_paired.png"
        plt.tight_layout()
        plt.savefig(paired_path, dpi=300)
        plt.close()

    logging.info("Summary plots saved to %s", plots_dir)
    return plots_dir
