"""Command-line entry point for the EEG involvement analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import involvement
import stat
from visual import create_summary_visualizations


def setup_logging(output_dir: Path) -> Path:
    """Configure logging to file and stdout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "involvement_analysis.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove existing handlers to avoid duplicates when re-running in the same session.
    while root_logger.handlers:
        root_logger.handlers.pop()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return log_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG involvement analysis.")
    parser.add_argument(
        "data_directory",
        type=Path,
        help="Path to the directory containing EEG_data and Subject_Condition.json",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Optional list of subject IDs to include (e.g., Subject_001)",
    )
    parser.add_argument(
        "--nights",
        nargs="+",
        help="Optional list of night labels to include (e.g., Night1)",
    )
    parser.add_argument(
        "--visualize-regions",
        action="store_true",
        help="Generate region-level plots for each processed wave.",
    )
    parser.add_argument(
        "--process-origins",
        action="store_true",
        help="Compute origin statistics (earliest 10%% of involved voxels).",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir: Path = args.data_directory.resolve()
    eeg_dir = data_dir / "EEG_data"
    mapping_path = data_dir / "Subject_Condition.json"
    output_dir = data_dir / "Involvement_Analysis"

    log_path = setup_logging(output_dir)
    logging.info("Involvement analysis started")
    logging.info("Data directory: %s", data_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Log file: %s", log_path)

    if not eeg_dir.is_dir():
        raise FileNotFoundError(f"Missing EEG_data directory at {eeg_dir}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing Subject_Condition.json at {mapping_path}")

    subject_condition_mapping = involvement.read_subject_condition_mapping(mapping_path)
    if not subject_condition_mapping:
        raise RuntimeError("Subject condition mapping could not be loaded")

    available_subjects, available_nights = involvement.scan_available_subjects_and_nights(eeg_dir)
    logging.info("Available subjects: %s", ", ".join(available_subjects) or "none")
    logging.info("Available nights: %s", ", ".join(available_nights) or "none")

    selected_subjects = args.subjects
    selected_nights = args.nights
    if selected_subjects:
        logging.info("Filtering to subjects: %s", ", ".join(selected_subjects))
    if selected_nights:
        logging.info("Filtering to nights: %s", ", ".join(selected_nights))

    results_by_group, subject_thresholds, processed_count = involvement.process_subject_data(
        eeg_dir,
        subject_condition_mapping,
        selected_subjects=selected_subjects,
        selected_nights=selected_nights,
        visualize_regions=args.visualize_regions,
        process_origins=args.process_origins,
        source_dir=data_dir,
    )

    if processed_count == 0:
        logging.error("No waves were processed. Check filters and input data.")
        return

    logging.info("Total waves processed: %d", processed_count)
    threshold_path = output_dir / "subject_thresholds.csv"
    pd.DataFrame(
        [
            {"Subject_ID": subject, "Threshold": value}
            for subject, value in sorted(subject_thresholds.items())
        ]
    ).to_csv(threshold_path, index=False)
    logging.info("Subject thresholds saved to %s", threshold_path)

    wave_df = stat.create_wave_dataframe(results_by_group, subject_thresholds)
    if wave_df.empty:
        logging.error("Wave DataFrame is empty; aborting downstream analysis")
        return
    wave_path = output_dir / "wave_involvement_data.csv"
    wave_df.to_csv(wave_path, index=False)
    logging.info("Wave-level data saved to %s", wave_path)

    subject_means_df = stat.create_subject_averaged_data(wave_df)
    subject_means_path = output_dir / "subject_averaged_involvement.csv"
    subject_means_df.to_csv(subject_means_path, index=False)
    logging.info("Subject-averaged data saved to %s", subject_means_path)

    statistical_summary = stat.generate_optimal_statistics(subject_means_df)
    stage_stats_df = stat.stage_statistics_to_dataframe(statistical_summary.get("stage_statistics", {}))
    stage_stats_path = output_dir / "stage_statistics.csv"
    stage_stats_df.to_csv(stage_stats_path, index=False)

    optimal_df = stat.optimal_results_to_dataframe(statistical_summary)
    optimal_path = output_dir / "statistical_results.csv"
    optimal_df.to_csv(optimal_path, index=False)
    logging.info("Statistical summary saved to %s", optimal_path)

    percentage_changes = stat.compute_percentage_changes(subject_means_df)
    changes_df = stat.percentage_changes_to_dataframe(percentage_changes)
    if not changes_df.empty:
        changes_path = output_dir / "percentage_changes.csv"
        changes_df.to_csv(changes_path, index=False)
        logging.info("Percentage changes saved to %s", changes_path)

    comprehensive_df = stat.run_comprehensive_tests(wave_df)
    if not comprehensive_df.empty:
        comprehensive_path = output_dir / "comprehensive_statistical_results.csv"
        comprehensive_df.to_csv(comprehensive_path, index=False)
        logging.info("Comprehensive statistical results saved to %s", comprehensive_path)

    create_summary_visualizations(subject_means_df, statistical_summary, output_dir)
    logging.info("Involvement analysis complete")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
