import os
from pathlib import Path

from analysis import (
    process_directory,
    process_eeg_data_directory,
    analyze_protocol_results,
    analyze_treatment_groups,
    analyze_overall_treatment_comparison,
    analyze_proto_specific_comparison,
    analyze_within_group_stage_comparison
)
from visualize import (
    visualize_results,
    visualize_overall_treatment_comparison,
    visualize_proto_specific_comparison,
    visualize_within_group_stage_comparison
)
from save_results import (
    save_statistical_results,
    save_treatment_comparison_results,
    save_overall_treatment_comparison_results,
    save_proto_specific_comparison_results,
    save_within_group_stage_comparison_results
)
from utils import compute_master_region_list


def main():
    directory_path = input("Enter the path to your EEG data directory: ").strip()
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    # Check if the directory is the new EEG_data structure or the old SourceRecon structure
    if any(Path(directory_path).glob("Active")) or any(Path(directory_path).glob("SHAM")):
        print("Detected new EEG data structure with treatment groups...")

        # Process the new directory structure with treatment groups
        results_by_treatment_group = process_eeg_data_directory(directory_path)
        if results_by_treatment_group is None:
            return

        # Flatten the results for protocol-level analysis
        results_by_protocol = {}
        for group in results_by_treatment_group:
            for protocol in results_by_treatment_group[group]:
                if protocol not in results_by_protocol:
                    results_by_protocol[protocol] = {'pre': [], 'early': [], 'late': [], 'post': []}
                for stage in ['pre', 'early', 'late', 'post']:
                    results_by_protocol[protocol][stage].extend(results_by_treatment_group[group][protocol][stage])
    else:
        print("Detected old SourceRecon structure...")
        # Process individual CSV files by protocol (old method)
        results_by_protocol = process_directory(directory_path)
        if results_by_protocol is None:
            return

        # Create a dummy treatment group structure for compatibility
        results_by_treatment_group = {'SingleGroup': results_by_protocol}

    print("\n=== Overall Data Summary ===")
    for group in results_by_treatment_group:
        print(f"\nTreatment Group: {group}")
        for protocol, stages in results_by_treatment_group[group].items():
            print(f"  {protocol}:")
            for stage, results in stages.items():
                print(f"    {stage}: {len(results)} waves")

    # Compute the master region list once, from the protocol-level data
    master_region_list = compute_master_region_list(results_by_protocol)

    # Analyze each protocol separately
    analysis_results = {}
    for protocol, protocol_results in results_by_protocol.items():
        analysis_results[protocol] = analyze_protocol_results(protocol, protocol_results, master_region_list)

    # If we have multiple treatment groups, analyze and compare them
    if len(results_by_treatment_group) > 1:
        print("\n=== Analyzing Treatment Groups ===")
        treatment_comparison_results = analyze_treatment_groups(results_by_treatment_group, master_region_list)

        # New analyses
        # 1. Overall Treatment Group Comparison (Collapsing Subjects, Nights, and Protos)
        overall_comparison_results = analyze_overall_treatment_comparison(results_by_treatment_group, master_region_list)
        # 2. Proto-Specific Comparison
        proto_specific_results = analyze_proto_specific_comparison(results_by_treatment_group, master_region_list)
        # 3. Within-Group Stage Comparison
        within_group_results = analyze_within_group_stage_comparison(results_by_treatment_group, master_region_list)

        # Create visualizations in an organized manner
        print("\n=== Creating Visualizations ===")
        
        # 1. Treatment group comparison (Active vs. SHAM)
        print("\n--- Overall Treatment Group Comparison (Active vs. SHAM) ---")
        visualize_overall_treatment_comparison(overall_comparison_results, output_dir="results")
        
        # 2. Protocol-specific treatment comparison
        print("\n--- Protocol-Specific Treatment Comparison ---")
        visualize_proto_specific_comparison(proto_specific_results, output_dir="results")
        
        # 3. Within-treatment-group stage comparison
        print("\n--- Within-Group Stage Comparison ---")
        visualize_within_group_stage_comparison(within_group_results, output_dir="results")

        # Save all CSV results from the new analyses
        treatment_comparison_files = save_treatment_comparison_results(treatment_comparison_results, output_dir="results")
        overall_comparison_files = save_overall_treatment_comparison_results(overall_comparison_results, output_dir="results")
        proto_specific_files = save_proto_specific_comparison_results(proto_specific_results, output_dir="results")
        within_group_files = save_within_group_stage_comparison_results(within_group_results, output_dir="results")
    else:
        treatment_comparison_results = None
        treatment_comparison_files = {}
        overall_comparison_results = None
        proto_specific_results = None
        within_group_results = None

    # Create visualizations for individual protocols
    print("\n=== Creating Visualizations for Individual Protocols ===")
    visualize_results(results_by_protocol, master_region_list)

    # Save CSV files for individual protocols
    stats_files = save_statistical_results(analysis_results, output_dir="results")

    print("\nAnalysis complete. Results have been printed, saved, and visualizations have been created.")

    print("\nStatistical results saved to:")
    for file_type, file_path in stats_files.items():
        if isinstance(file_path, list):
            for path in file_path:
                if path is not None:
                    print(f" - {path}")
        elif file_path is not None:
            print(f" - {file_path}")

    if treatment_comparison_results:
        print("\nTreatment comparison results saved to:")
        for file_type, file_path in treatment_comparison_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        print(f" - {path}")
            elif file_path is not None:
                print(f" - {file_path}")

        print("\nOverall treatment comparison results saved to:")
        for file_type, file_path in overall_comparison_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        print(f" - {path}")
            elif file_path is not None:
                print(f" - {file_path}")

        print("\nProto-specific comparison results saved to:")
        for file_type, file_path in proto_specific_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        print(f" - {path}")
            elif file_path is not None:
                print(f" - {file_path}")

        print("\nWithin-group stage comparison results saved to:")
        for file_type, file_path in within_group_files.items():
            if isinstance(file_path, list):
                for path in file_path:
                    if path is not None:
                        print(f" - {path}")
            elif file_path is not None:
                print(f" - {file_path}")


if __name__ == "__main__":
    main()
