"""Utilities for loading EEG source data and computing wave involvement metrics."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PROTO_PATTERN = re.compile(r"proto(\d+)", re.IGNORECASE)
STAGE_PATTERN = re.compile(r"(pre|early|late|post|stim)(?:-stim)?", re.IGNORECASE)
VALID_PROTOCOL_RANGE = range(1, 9)
WINDOW_START_MS = -50
WINDOW_END_MS = 50


def extract_region_name(full_name: str) -> str:
    """Return the anatomical region prefix from a voxel label."""
    if not isinstance(full_name, str):
        return str(full_name)
    if "." in full_name:
        return full_name.split(".")[0]
    if "['" in full_name and "." in full_name:
        try:
            return full_name.split("['")[1].split(".")[0]
        except IndexError:
            return full_name
    return full_name


def read_subject_condition_mapping(json_path: Path) -> Dict[str, str]:
    """Load the subject-to-condition mapping."""
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logging.error("Unable to read subject condition file '%s': %s", json_path, exc)
        return {}
    return {str(key): str(value) for key, value in data.items()}


def scan_available_subjects_and_nights(directory_path: Path) -> Tuple[List[str], List[str]]:
    """List available subjects and nights in the EEG data directory."""
    subjects: List[str] = []
    nights: set[str] = set()
    if not directory_path.exists():
        return subjects, sorted(nights)
    for subject_dir in directory_path.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("Subject_"):
            subjects.append(subject_dir.name)
            for night_dir in subject_dir.iterdir():
                if night_dir.is_dir() and night_dir.name.startswith("Night"):
                    nights.add(night_dir.name)
    return sorted(subjects), sorted(nights)


def analyze_slow_wave(
    df: pd.DataFrame,
    wave_name: str,
    *,
    threshold_percent: float = 50.0,
    fixed_threshold: Optional[float] = None,
    process_origins: bool = True,
) -> Dict[str, object]:
    """Analyse a single SourceRecon CSV for involvement and origin metrics."""
    if "Time" not in df.columns:
        raise ValueError("CSV format requires a 'Time' column")

    numeric_cols: List[str] = []
    for column in df.columns[1:]:
        if str(column).startswith("Unnamed"):
            continue
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        numeric_cols.append(column)

    if not numeric_cols:
        raise ValueError("No numeric time columns detected")

    time_points = np.array([float(t) for t in numeric_cols]) * 1000.0
    data = np.abs(df.loc[:, numeric_cols].to_numpy(dtype=float))
    voxel_names = df.iloc[:, 0].astype(str).to_numpy()

    window_mask = (time_points >= WINDOW_START_MS) & (time_points <= WINDOW_END_MS)
    if not np.any(window_mask):
        window_mask = np.ones_like(time_points, dtype=bool)
        logging.warning(
            "Wave %s has no data in %d to %d ms; using full window",
            wave_name,
            WINDOW_START_MS,
            WINDOW_END_MS,
        )

    window_data = data[:, window_mask]
    window_times = time_points[window_mask]

    origins_df = pd.DataFrame(columns=["full_name", "region", "peak_time"])
    global_max_value = 0.0
    global_max_time = float("nan")
    threshold = float(fixed_threshold) if fixed_threshold is not None else 0.0

    if window_data.size:
        peak_value = float(np.max(window_data))
        if fixed_threshold is None:
            threshold = peak_value * (threshold_percent / 100.0)
        flat_idx = int(np.argmax(window_data))
        voxel_idx_max, time_idx_max = np.unravel_index(flat_idx, window_data.shape)
        global_max_value = peak_value
        global_max_time = float(window_times[time_idx_max])

    voxel_peak_rows: List[dict] = []
    involved_voxels: List[str] = []

    for voxel_idx in range(window_data.shape[0]):
        voxel_series = window_data[voxel_idx]
        peaks, _ = find_peaks(voxel_series, height=threshold)
        if peaks.size == 0:
            continue
        peak_times = window_times[peaks]
        peak_values = voxel_series[peaks]
        peaks_above = list(zip(peak_times, peak_values))
        if not peaks_above:
            continue
        closest_peak_time, _ = min(peaks_above, key=lambda item: abs(item[0]))
        region = extract_region_name(voxel_names[voxel_idx])
        voxel_peak_rows.append(
            {
                "full_name": voxel_names[voxel_idx],
                "region": region,
                "peak_time": float(closest_peak_time),
            }
        )
        involved_voxels.append(voxel_names[voxel_idx])

    if process_origins and voxel_peak_rows:
        peaks_df = pd.DataFrame(voxel_peak_rows).sort_values("peak_time")
        n_origins = max(1, int(len(peaks_df) * 0.1))
        origins_df = peaks_df.head(n_origins).reset_index(drop=True)

    total_voxels = int(data.shape[0]) if data is not None else 0
    num_involved = len(involved_voxels)
    involvement_percentage = (num_involved / total_voxels) * 100.0 if total_voxels else 0.0

    return {
        "wave_name": wave_name,
        "origins": origins_df,
        "involvement_count": num_involved,
        "involvement_percentage": involvement_percentage,
        "involved_voxels": involved_voxels,
        "window": (WINDOW_START_MS, WINDOW_END_MS),
        "threshold": threshold,
        "global_max_value": global_max_value,
        "global_max_time": global_max_time,
    }


def _log_wave_result(result: Dict[str, object], csv_file: Path, *, subject_id: Optional[str] = None) -> None:
    """Log summary details for a processed wave and save an optional sidecar log."""
    wave_name = str(result.get("wave_name", csv_file.stem))
    lines = [f"Wave: {wave_name}"]
    if subject_id:
        threshold = result.get("threshold")
        lines.append(
            f"Subject: {subject_id} | Threshold: {threshold:.6e}" if isinstance(threshold, (int, float)) else f"Subject: {subject_id}"
        )
    involvement = result.get("involvement_percentage", 0.0)
    count = result.get("involvement_count", 0)
    lines.append(f"Involvement: {involvement:.2f}% ({count} voxels)")

    origins = result.get("origins")
    if isinstance(origins, pd.DataFrame) and not origins.empty:
        lines.append("Origin regions:")
        for _, row in origins.iterrows():
            lines.append(f"  - {row['region']} at {row['peak_time']:.2f} ms")
    else:
        lines.append("Origin regions: none")

    window = result.get("window", (WINDOW_START_MS, WINDOW_END_MS))
    lines.append(f"Window: {window[0]} ms to {window[1]} ms")

    for line in lines:
        logging.info(line)

    log_path = csv_file.with_suffix(".log")
    if subject_id:
        log_path = csv_file.with_suffix("")
        log_path = log_path.with_name(f"{log_path.name}_{subject_id}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:
        logging.warning("Unable to write wave log for %s: %s", csv_file, exc)


def _collect_subject_threshold(
    subject_dir: Path, *, night_filter: Optional[set[str]] = None
) -> Optional[float]:
    """Compute the 95th percentile pre-stim threshold for a subject."""
    values: List[float] = []
    for night_dir in subject_dir.iterdir():
        if not night_dir.is_dir() or not night_dir.name.startswith("Night"):
            continue
        if night_filter and night_dir.name not in night_filter:
            continue
        source_recon_dir = night_dir / "Output" / "SourceRecon"
        if not source_recon_dir.exists():
            continue
        for csv_file in source_recon_dir.glob("*proto*pre*.csv"):
            match = PROTO_PATTERN.search(csv_file.name)
            if not match:
                continue
            proto_num = int(match.group(1))
            if proto_num not in VALID_PROTOCOL_RANGE:
                continue
            try:
                df = pd.read_csv(csv_file)
                if "Time" not in df.columns:
                    continue
                numeric_cols = [col for col in df.columns[1:] if not str(col).startswith("Unnamed")]
                if not numeric_cols:
                    continue
                data = np.abs(df.loc[:, numeric_cols].to_numpy(dtype=float))
                values.extend(data.ravel().tolist())
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning("Unable to read %s for threshold estimation: %s", csv_file, exc)
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), 95))


def process_subject_directory(
    subject_dir: Path,
    *,
    subject_threshold: float,
    subject_id: str,
    visualize_regions: bool,
    process_origins: bool,
    source_dir: Optional[Path],
    night_filter: Optional[set[str]],
) -> Tuple[Dict[str, Dict[str, List[dict]]], int]:
    """Process all waves for a subject using the supplied threshold."""
    subject_results: Dict[str, Dict[str, List[dict]]] = {}
    processed = 0

    night_dirs = [
        d
        for d in subject_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("Night")
        and (not night_filter or d.name in night_filter)
    ]
    for night_dir in sorted(night_dirs):
        source_recon_dir = night_dir / "Output" / "SourceRecon"
        if not source_recon_dir.exists():
            logging.warning("Skipping %s (missing SourceRecon)", night_dir)
            continue

        night_results: Dict[str, Dict[str, List[dict]]] = {}
        csv_files = sorted(source_recon_dir.glob("*.csv"))
        if not csv_files:
            continue
        for csv_file in csv_files:
            proto_match = PROTO_PATTERN.search(csv_file.name)
            stage_match = STAGE_PATTERN.search(csv_file.name)
            if not proto_match or not stage_match:
                continue
            proto_num = int(proto_match.group(1))
            if proto_num not in VALID_PROTOCOL_RANGE:
                continue
            protocol = f"proto{proto_num}"
            stage = stage_match.group(1).lower()

            try:
                df = pd.read_csv(csv_file)
                wave_name = csv_file.stem
                result = analyze_slow_wave(
                    df,
                    wave_name,
                    fixed_threshold=subject_threshold,
                    process_origins=process_origins,
                )
                result["subject_id"] = subject_id
                result["subject_threshold"] = subject_threshold
                night_results.setdefault(protocol, {}).setdefault(stage, []).append(result)
                _log_wave_result(result, csv_file, subject_id=subject_id)
                processed += 1

                if visualize_regions:
                    try:
                        from visual import visualize_region_time_series  # local import to avoid cycle

                        visualize_region_time_series(result, csv_file, source_dir=source_dir)
                    except Exception as exc:  # pylint: disable=broad-except
                        logging.warning("Region visualization failed for %s: %s", csv_file, exc)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Error processing %s: %s", csv_file, exc)

        for protocol, stage_dict in night_results.items():
            subject_results.setdefault(protocol, {})
            for stage, waves in stage_dict.items():
                subject_results[protocol].setdefault(stage, []).extend(waves)

    return subject_results, processed


def process_subject_data(
    directory_path: Path,
    subject_condition_mapping: Dict[str, str],
    *,
    selected_subjects: Optional[Iterable[str]] = None,
    selected_nights: Optional[Iterable[str]] = None,
    visualize_regions: bool = False,
    process_origins: bool = False,
    source_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, float], int]:
    """Process all subjects with subject-specific thresholds."""
    selected_subjects_set = set(selected_subjects or [])
    selected_nights_set = set(selected_nights or [])

    treatment_groups = {"Active", "SHAM"}
    results_by_group: Dict[str, Dict[str, dict]] = {group: {} for group in treatment_groups}
    subject_thresholds: Dict[str, float] = {}
    total_processed = 0

    subject_dirs = sorted(
        d for d in directory_path.iterdir() if d.is_dir() and d.name.startswith("Subject_")
    )
    if selected_subjects_set:
        subject_dirs = [d for d in subject_dirs if d.name in selected_subjects_set]

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        group = subject_condition_mapping.get(subject_id)
        if group not in treatment_groups:
            logging.warning("Subject %s missing or invalid treatment group", subject_id)
            continue

        threshold = _collect_subject_threshold(subject_dir, night_filter=selected_nights_set or None)
        if threshold is None:
            logging.error("Subject %s has no proto1-8 pre-stim data for thresholding", subject_id)
            continue

        logging.info("Subject %s threshold set to %.6e", subject_id, threshold)
        subject_thresholds[subject_id] = threshold

        subject_results, processed = process_subject_directory(
            subject_dir,
            subject_threshold=threshold,
            subject_id=subject_id,
            visualize_regions=visualize_regions,
            process_origins=process_origins,
            source_dir=source_dir,
            night_filter=selected_nights_set or None,
        )
        total_processed += processed
        if not subject_results:
            logging.warning("Subject %s produced no results", subject_id)
            continue
        results_by_group.setdefault(group, {})[subject_id] = subject_results

    return results_by_group, subject_thresholds, total_processed
