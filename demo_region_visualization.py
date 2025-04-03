#!/usr/bin/env python
"""
Demo script for region time series visualization.

This script demonstrates how to use the new region time series visualization
functionality to generate plots showing time series data for brain regions,
with highlighted local maxima and origins.

Usage:
    python demo_region_visualization.py [csv_file_path]
"""

import os
import sys
import pandas as pd
from pathlib import Path

from analysis import analyze_slow_wave
from visualize import visualize_region_time_series

def main():
    # Check if a specific CSV file was provided
    if len(sys.argv) > 1:
        csv_file_path = sys.argv[1]
        if not os.path.exists(csv_file_path):
            print(f"Error: File not found: {csv_file_path}")
            return
    else:
        # If no file provided, attempt to find one automatically
        print("No file specified. Trying to find a sample CSV file...")
        # Look for CSV files in the current directory or parent directories
        parent_dir = Path.cwd()
        csv_files = list(parent_dir.glob("*.csv"))
        
        if not csv_files:
            # Try one level up
            parent_dir = parent_dir.parent
            csv_files = list(parent_dir.glob("**/*.csv"))
        
        if not csv_files:
            print("Error: No CSV files found. Please provide a path to a CSV file.")
            print("Usage: python demo_region_visualization.py [csv_file_path]")
            return
            
        # Use the first CSV file found
        csv_file_path = csv_files[0]
    
    # Print the file we're using
    print(f"Using CSV file: {csv_file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        wave_name = Path(csv_file_path).stem
        
        # Analyze the slow wave
        print(f"Analyzing {wave_name}...")
        result = analyze_slow_wave(df, wave_name)
        
        # Create the region time series visualization
        print("Generating region time series visualization...")
        output_dir = "results"
        visualize_region_time_series(result, csv_file_path, output_dir)
        
        print(f"\nVisualization completed! Check the {output_dir}/region_plots directory for the plots.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()
