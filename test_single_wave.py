import pandas as pd
import logging
from pathlib import Path
import sys

# Add the current directory to the path to find analysis module
sys.path.append(str(Path(__file__).parent))

try:
    from analysis import analyze_slow_wave, validate_wave_result
except ImportError as e:
    print(f"Error importing analysis functions: {e}")
    print("Ensure analysis.py is in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Specify the full path to the single CSV file you want to test
csv_file_path_str = "/Users/wyh/STRENGTHEN/EEG_data/Subject_001/Night1/Output/SourceRecon/proto1_post-stim_sw103_E3_scouts.csv"
csv_file_path = Path(csv_file_path_str)
# --- End Configuration ---

def main():
    """Loads and analyzes a single wave CSV file."""
    logging.info(f"Attempting to load CSV file: {csv_file_path}")

    if not csv_file_path.exists():
        logging.error(f"CSV file not found: {csv_file_path}")
        return

    try:
        # Load the CSV data
        df = pd.read_csv(csv_file_path)
        wave_name = csv_file_path.stem
        logging.info(f"Successfully loaded {wave_name}. Starting analysis...")

        # Analyze the slow wave
        # Pass the threshold percentage if it's different from the default (25)
        result = analyze_slow_wave(df, wave_name, threshold_percent=25)

        logging.info(f"Analysis complete for {wave_name}. Validating results...")

        # Validate and print the results (including threshold)
        # Pass the original csv_file_path to enable saving the .log file
        validate_wave_result(result, csv_file_path=csv_file_path)

        logging.info("Test script finished.")

    except FileNotFoundError:
        logging.error(f"Error: The file {csv_file_path} was not found.")
    except pd.errors.EmptyDataError:
        logging.error(f"Error: The file {csv_file_path} is empty.")
    except Exception as e:
        logging.error(f"An error occurred during processing {csv_file_path}: {str(e)}")
        # Optionally re-raise for more detailed traceback
        # raise

if __name__ == "__main__":
    main()
