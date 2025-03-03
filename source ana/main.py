import os
from analysis import process_directory, analyze_protocol_results, analyze_meta_protocol
from visualize import visualize_results, visualize_meta_results
from save_results import save_statistical_results, save_meta_results
from utils import compute_master_region_list, collect_wave_level_data

def main():
    directory_path = input("Enter the path to your source reconstruction directory: ").strip()
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    # Process individual CSV files by protocol
    results_by_protocol = process_directory(directory_path)
    if results_by_protocol is None:
        return

    print("\n=== Overall Data Summary ===")
    for protocol, stages in results_by_protocol.items():
        print(f"\n{protocol}:")
        for stage, results in stages.items():
            print(f"  {stage}: {len(results)} waves")

    # Compute the master region list once
    master_region_list = compute_master_region_list(results_by_protocol)

    # Analyze each protocol separately
    analysis_results = {}
    for protocol, protocol_results in results_by_protocol.items():
        analysis_results[protocol] = analyze_protocol_results(protocol, protocol_results, master_region_list)

    # Perform meta protocol analysis (protocol-blind)
    meta_results = analyze_meta_protocol(results_by_protocol, master_region_list)

    # Create visualizations for individual protocols
    print("\n=== Creating Visualizations for Individual Protocols ===")
    visualize_results(results_by_protocol, master_region_list)

    # Create visualizations for the meta protocol analysis
    print("\n=== Creating Visualizations for Meta Protocol ===")
    visualize_meta_results(meta_results)

    # Save CSV files for individual protocols
    stats_files = save_statistical_results(analysis_results, output_dir="results")
    # Save CSV files for meta protocol results
    meta_stats_files = save_meta_results(meta_results, output_dir="results")

    print("\nAnalysis complete. Results have been printed, saved, and visualizations have been created.")
    print("\nStatistical results saved to:")
    for file_type, file_path in stats_files.items():
        if isinstance(file_path, list):
            for path in file_path:
                if path is not None:
                    print(f" - {path}")
        elif file_path is not None:
            print(f" - {file_path}")
    print("\nMeta statistical results saved to:")
    for file_type, file_path in meta_stats_files.items():
        if isinstance(file_path, list):
            for path in file_path:
                if path is not None:
                    print(f" - {path}")
        elif file_path is not None:
            print(f" - {file_path}")

if __name__ == "__main__":
    main()
