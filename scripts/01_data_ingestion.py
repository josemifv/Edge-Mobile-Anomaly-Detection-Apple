#!/usr/bin/env python3
"""
01_data_ingestion.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 1: Data Ingestion and Initial Processing

Loads raw telecommunication data files (.txt format), performs initial cleaning,
and combines them into a single Parquet file.
"""

import pandas as pd
import os
import argparse
import time
import multiprocessing
from pathlib import Path

# Column definitions for the Milano Telecom Dataset
COLUMN_NAMES = [
    'cell_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_traffic'
]

def process_single_file(file_path: Path) -> pd.DataFrame | None:
    """
    Loads, cleans, and processes a single raw data file.

    Args:
        file_path: The path to the input .txt file.

    Returns:
        A pandas DataFrame with the processed data, or None if an error occurs.
    """
    print(f"Processing: {file_path.name}")
    try:
        # Read file using a regex separator for variable whitespace
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            names=COLUMN_NAMES,
            dtype={'cell_id': 'Int64', 'timestamp_ms': 'Int64'},
            low_memory=False
        )

        # Convert timestamp and drop rows with conversion errors
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)

        # Drop unnecessary columns
        df.drop(columns=['timestamp_ms', 'country_code'], inplace=True)

        # Reorder for consistency
        final_cols = ['timestamp', 'cell_id', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']
        
        return df[final_cols]

    except Exception as e:
        print(f"  ERROR processing {file_path.name}: {e}")
        return None

def main():
    """Main function to orchestrate the data ingestion stage."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 1 - Data Ingestion")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .txt data files")
    parser.add_argument("--output_path", type=Path, default="outputs/01_ingested_data.parquet", help="Output file path")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes (default: auto)")
    args = parser.parse_args()

    print("="*60)
    print("Stage 1: Data Ingestion")
    print("="*60)

    start_time = time.perf_counter()

    # Find input files
    txt_files = sorted(list(args.input_dir.glob("*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {args.input_dir}")
    print(f"Found {len(txt_files)} files to process.")

    # Determine worker count and start parallel processing
    num_workers = args.max_workers or min(len(txt_files), os.cpu_count())
    print(f"Using {num_workers} parallel processes...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        dataframes = pool.map(process_single_file, txt_files)

    # Consolidate results
    valid_dfs = [df for df in dataframes if df is not None]
    print(f"\nSuccessfully processed {len(valid_dfs)}/{len(txt_files)} files.")

    if not valid_dfs:
        raise ValueError("No data files were processed successfully.")

    print("Combining processed files into a single DataFrame...")
    combined_df = pd.concat(valid_dfs, ignore_index=True)

    # Save output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving combined data to {args.output_path}...")
    combined_df.to_parquet(args.output_path, index=False)

    # Final summary
    total_time = time.perf_counter() - start_time
    print("\n--- STAGE 1 SUMMARY ---")
    print(f"Total rows ingested: {len(combined_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processing rate: {len(combined_df) / total_time:,.0f} rows/second")
    print("✅ Stage 1 completed successfully.")

if __name__ == "__main__":
    main()