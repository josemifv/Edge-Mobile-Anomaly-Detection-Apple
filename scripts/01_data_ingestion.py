#!/usr/bin/env python3
"""
01_data_ingestion.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 1: Data Ingestion and Initial Processing

Loads raw telecommunication data files (.txt format) and performs initial preprocessing.
Optimized for Apple Silicon with parallel processing capabilities.

Usage:
    python scripts/01_data_ingestion.py <input_path> [--output_path <path>]

Example:
    python scripts/01_data_ingestion.py data/raw/ --output_path data/processed/ingested_data.parquet
"""

import pandas as pd
import os
import argparse
import time
import multiprocessing
from datetime import datetime
from pathlib import Path

# Column definitions for Milano Telecom Dataset
COLUMN_NAMES = [
    'cell_id',          # Previously 'square_id' 
    'timestamp_ms',     # Unix timestamp in milliseconds
    'country_code',     # Country code (will be dropped in preprocessing)
    'sms_in',          # Incoming SMS activity
    'sms_out',         # Outgoing SMS activity  
    'call_in',         # Incoming call activity
    'call_out',        # Outgoing call activity
    'internet_traffic' # Internet activity
]


def load_single_file(file_path: str) -> pd.DataFrame:
    """Load and preprocess a single telecommunication data file."""
    print(f"Processing: {os.path.basename(file_path)}")
    start_time = time.perf_counter()
    
    try:
        # Read file with space-separated values 
        # Note: Using regex sep forces Python engine, but needed for variable whitespace
        # Performance optimization: Specify dtypes upfront for faster parsing
        df = pd.read_csv(
            file_path,
            sep=r'\s+',  # Use raw string to avoid escape sequence warning
            header=None,
            names=COLUMN_NAMES,
            dtype={
                'cell_id': 'Int64',
                'timestamp_ms': 'Int64', 
                'country_code': 'Int64',
                'sms_in': 'float64',
                'sms_out': 'float64',
                'call_in': 'float64',
                'call_out': 'float64',
                'internet_traffic': 'float64'
            },
            # Performance optimizations
            low_memory=False,
            na_values=[''],  # Handle empty strings as NaN
            keep_default_na=True  # Keep default NaN handling for robustness
        )
        
        # Convert timestamp from milliseconds to datetime
        initial_rows = len(df)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
        
        # Check for and handle NaT (Not-a-Time) values from timestamp conversion
        nat_mask = df['timestamp'].isna()
        nat_count = nat_mask.sum()
        
        if nat_count > 0:
            print(f"  WARNING: {nat_count} rows with invalid timestamps (NaT) will be dropped")
            df = df[~nat_mask].copy()
            print(f"  Kept {len(df):,} valid rows out of {initial_rows:,} ({(len(df)/initial_rows)*100:.1f}%)")
        
        # Drop unnecessary columns
        df = df.drop(columns=['timestamp_ms', 'country_code'])
        
        # Reorder columns
        df = df[['timestamp', 'cell_id', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']]
        
        duration = time.perf_counter() - start_time
        print(f"  Processed {len(df):,} rows in {duration:.2f}s")
        
        return df
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 1 - Data Ingestion")
    parser.add_argument("input_path", help="Directory containing .txt data files")
    parser.add_argument("--output_path", default="data/processed/ingested_data.parquet", 
                       help="Output file path")
    parser.add_argument("--max_workers", type=int, 
                       help="Max parallel processes (default: auto)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 1: Data Ingestion")
    print("="*60)
    
    start_time = time.perf_counter()
    
    # Find all .txt files
    input_path = Path(args.input_path)
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {args.input_path}")
    
    print(f"Found {len(txt_files)} files to process")
    
    # Determine number of workers (Apple Silicon optimization)
    max_workers = args.max_workers or min(len(txt_files), multiprocessing.cpu_count())
    print(f"Using {max_workers} parallel processes")
    
    # Process files in parallel
    with multiprocessing.Pool(processes=max_workers) as pool:
        dataframes = pool.map(load_single_file, [str(f) for f in txt_files])
    
    # Filter successful results
    valid_dfs = [df for df in dataframes if df is not None]
    failed_count = len(dataframes) - len(valid_dfs)
    
    if failed_count > 0:
        print(f"WARNING: {failed_count} files failed to process")
    
    if not valid_dfs:
        raise ValueError("No files were successfully processed")
    
    # Combine all DataFrames
    print("Combining all processed files...")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    print(f"Saving to {args.output_path}...")
    if output_path.suffix.lower() == '.parquet':
        combined_df.to_parquet(args.output_path, index=False)
    else:
        combined_df.to_csv(args.output_path, index=False)
    
    # Performance summary
    total_time = time.perf_counter() - start_time
    total_rows = len(combined_df)
    
    print("="*60)
    print("STAGE 1 SUMMARY")
    print("="*60)
    print(f"Files processed: {len(valid_dfs)}/{len(txt_files)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Processing rate: {total_rows/total_time:,.0f} rows/second")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Unique cells: {combined_df['cell_id'].nunique():,}")
    print(f"Output file: {args.output_path}")
    print("Stage 1 completed successfully!")


if __name__ == "__main__":
    main()
