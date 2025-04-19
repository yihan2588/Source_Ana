import os
import logging
import sys
from pathlib import Path

from analysis import (
    process_eeg_data_directory,
    analyze_overall_treatment_comparison,
    analyze_proto_specific_comparison,
    analyze_within_group_stage_comparison
)
from visualize import (
    visualize_overall_treatment_comparison,
    visualize_proto_specific_comparison,
    visualize_within_group_stage_comparison
)
from save_results import (
    save_overall_treatment_comparison_results,
    save_proto_specific_comparison_results,
    save_within_group_stage_comparison_results
)
from utils import (
    compute_master_region_list,
    read_subject_condition_mapping,
    scan_available_subjects_and_nights
)


def main():
    # --- User Input ---
    data_directory = input("Enter the path to your data directory: ").strip()
    if not os.path.isdir(data_directory):
        print(f"Error: '{data_directory}' is not a valid directory.") # Keep print for early errors before logging setup
        return

    # Find EEG_data subdirectory
    eeg_data_path = os.path.join(data_directory, "EEG_data")
    if not os.path.isdir(eeg_data_path):
        print(f"Error: EEG_data directory not found in '{data_directory}'.")
        return

    # Find Subject_Condition.json file
    json_path = os.path.join(data_directory, "Subject_Condition.json")
    if not os.path.isfile(json_path):
        print(f"Error: Subject_Condition.json file not found in '{data_directory}'.")
        return

    # Read subject-condition mapping
    subject_condition_mapping = read_subject_condition_mapping(json_path)
    if not subject_condition_mapping:
        return # Error printed in function

    # Scan for available subjects and nights
    subjects, nights = scan_available_subjects_and_nights(eeg_data_path)

    # Display available subjects
    print("\nAvailable subjects:") # Keep print for interactive part
    for i, subject in enumerate(subjects, 1):
        print(f"{i}. {subject}")

    # Get user selection for subjects
    subject_input = input("\nEnter the numbers of subjects to process (comma-separated, or 'all'): ").strip()
    if subject_input.lower() == 'all':
        selected_subjects = subjects
    else:
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in subject_input.split(',')]
            selected_subjects = [subjects[idx] for idx in selected_indices if 0 <= idx < len(subjects)]
        except (ValueError, IndexError):
            print("Invalid input. Please enter comma-separated numbers.")
            return

    # Display available nights
    print("\nAvailable nights:") # Keep print for interactive part
    for i, night in enumerate(nights, 1):
        print(f"{i}. {night}")

    # Get user selection for nights
    night_input = input("\nEnter the numbers of nights to process (comma-separated, or 'all'): ").strip()
    if night_input.lower() == 'all':
        selected_nights = nights
    else:
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in night_input.split(',')]
            selected_nights = [nights[idx] for idx in selected_indices if 0 <= idx < len(nights)]
        except (ValueError, IndexError):
            print("Invalid input. Please enter comma-separated numbers.")
            return

    # --- Logging Setup ---
    # The source directory is where we'll create the "Source_Ana" directory for results
    source_dir = data_directory
    results_base_dir = os.path.join(source_dir, "Source_Ana")
    os.makedirs(results_base_dir, exist_ok=True)
    log_file_path = os.path.join(results_base_dir, "pipeline_run.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Write mode to overwrite each run
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    logging.info(f"Pipeline started. Log file: {log_file_path}")
    logging.info(f"Processing {len(selected_subjects)} subjects and {len(selected_nights)} nights...")

    # --- Main Processing ---
    results_by_treatment_group = process_eeg_data_directory(
        eeg_data_path,
        subject_condition_mapping,
        selected_subjects,
        selected_nights,
        visualize_regions=True,
        source_dir=source_dir # Pass source_dir for output path construction
    )

    if results_by_treatment_group is None:
        logging.error("Processing failed. Exiting.")
        return

    # --- Data Aggregation & Summary ---
    # Flatten results for master region list computation (aggregating across groups and subjects)
    # This structure is NOT used for analysis anymore, only for master list
    results_for_master_list = {}
    all_protocols = set()
    all_stages = set()

    # Identify all protocols and stages
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            subject_protocols = results_by_treatment_group[group][subject].keys()
            all_protocols.update(subject_protocols)
            for protocol in subject_protocols:
                protocol_stages = results_by_treatment_group[group][subject][protocol].keys()
                all_stages.update(protocol_stages)

    # Initialize and populate the flattened structure for master list
    for protocol in all_protocols:
        results_for_master_list[protocol] = {stage: [] for stage in all_stages}
    for group in results_by_treatment_group:
        for subject in results_by_treatment_group[group]:
            for protocol in results_by_treatment_group[group][subject]:
                for stage in results_by_treatment_group[group][subject][protocol]:
                    if protocol in results_for_master_list and stage in results_for_master_list[protocol]:
                        results_for_master_list[protocol][stage].extend(results_by_treatment_group[group][subject][protocol][stage])

    # Log Overall Data Summary
    logging.info("\n=== Overall Data Summary (Waves per Subject/Proto/Stage) ===")
    for group in results_by_treatment_group:
        logging.info(f"\nTreatment Group: {group}")
        for subject, subject_data in results_by_treatment_group[group].items():
             logging.info(f"  Subject: {subject}")
             for protocol, proto_data in subject_data.items():
                 logging.info(f"    {protocol}:")
                 for stage, results in proto_data.items():
                     logging.info(f"      {stage}: {len(results)} waves")

    # Compute the master region list once
    master_region_list = compute_master_region_list(results_for_master_list) # Use the flattened data

    # --- Analysis & Comparison ---
    # Individual protocol analysis is removed as requested.

    # Perform comparisons if multiple treatment groups exist
    if len(results_by_treatment_group) > 1:
        logging.info("\n=== Analyzing Treatment Groups ===")

        # 1. Overall Treatment Group Comparison
        overall_comparison_results = analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list)
        # 2. Proto-Specific Comparison
        proto_specific_results = analyze_proto_specific_comparison(results_by_treatment_group, master_region_list)
        # 3. Within-Group Stage Comparison
        within_group_results = analyze_within_group_stage_comparison(results_by_treatment_group, master_region_list)

        # --- Visualization ---
        logging.info("\n=== Creating Visualizations ===")

        # 1. Overall Treatment Group Comparison Visualization
        logging.info("\n--- Overall Treatment Group Comparison (Active vs. SHAM) ---")
        visualize_overall_treatment_comparison(overall_comparison_results, source_dir=source_dir) # output_dir handled internally

        # 2. Protocol-Specific Treatment Comparison Visualization
        logging.info("\n--- Protocol-Specific Treatment Comparison ---")
        visualize_proto_specific_comparison(proto_specific_results, source_dir=source_dir) # output_dir handled internally

        # 3. Within-Treatment-Group Stage Comparison Visualization
        logging.info("\n--- Within-Group Stage Comparison ---")
        visualize_within_group_stage_comparison(within_group_results, source_dir=source_dir) # output_dir handled internally

        # --- Saving Results ---
        # Save comparison results
        overall_comparison_files = save_overall_treatment_comparison_results(overall_comparison_results, source_dir=source_dir) # output_dir handled internally
        proto_specific_files = save_proto_specific_comparison_results(proto_specific_results, source_dir=source_dir) # output_dir handled internally
        within_group_files = save_within_group_stage_comparison_results(within_group_results, source_dir=source_dir) # output_dir handled internally

    else:
        logging.warning("Only one treatment group found. Skipping comparison analyses, visualizations, and saving.")
        # Initialize file dictionaries as empty if no comparison is done
        overall_comparison_files = {}
        proto_specific_files = {}
        within_group_files = {}
        overall_comparison_results = None
        proto_specific_results = None
        within_group_results = None

    # Individual protocol visualizations and saving are removed as requested.

    logging.info("\nAnalysis complete. Results have been logged, saved, and visualizations have been created.")

    # Log paths for the comparison results if they were generated
    if overall_comparison_results: # Check if comparisons were run
        logging.info("\nOverall treatment comparison results saved to:")
        for file_type, file_path in overall_comparison_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        logging.info(f" - {path}")
            elif file_path is not None:
                logging.info(f" - {file_path}")

        logging.info("\nProto-specific comparison results saved to:")
        for file_type, file_path in proto_specific_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        logging.info(f" - {path}")
            elif file_path is not None:
                logging.info(f" - {file_path}")

        logging.info("\nWithin-group stage comparison results saved to:")
        for file_type, file_path in within_group_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        logging.info(f" - {path}")
            elif file_path is not None:
                logging.info(f" - {file_path}")


if __name__ == "__main__":
    main()
