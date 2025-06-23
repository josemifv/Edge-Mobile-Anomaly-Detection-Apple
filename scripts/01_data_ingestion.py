#!/usr/bin/env python3
"""
01_data_ingestion.py

CMMSE 2025: Stage 1 - Data Ingestion and Initial Processing
Performs high-performance, parallel data ingestion from raw text files
into a consolidated Parquet file.
"""

import polars as pl
import argparse
import time
from pathlib import Path

# Column definitions for the Milano Telecom Dataset
COLUMN_NAMES = [
    'cell_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_traffic'
]

# Data types definition for Polars
COLUMN_DTYPES = {
    'cell_id': pl.Int64,
    'timestamp_ms': pl.Int64,
    'country_code': pl.Int32,
    'sms_in': pl.Float64,
    'sms_out': pl.Float64,
    'call_in': pl.Float64,
    'call_out': pl.Float64,
    'internet_traffic': pl.Float64
}

def run_ingestion_stage(input_dir: Path, output_path: Path) -> pl.DataFrame:
    """
    Ingests all .txt files from a directory using a lazy engine for high performance.

    Args:
        input_dir: Directory containing the raw .txt files.
        output_path: Path to save the resulting Parquet file.

    Returns:
        A Polars DataFrame containing the combined and processed data.
    """
    print(f"Starting parallel ingestion from: {input_dir}")
    
    # Verify input files exist
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    print(f"Found {len(txt_files)} files to process")
    
    # Use scan_csv for lazy reading of all files.
    # The engine will automatically parallelize the reading process.
    # Milano dataset uses tab separation with some empty fields
    lazy_df = pl.scan_csv(
        source=str(input_dir / "*.txt"),
        separator='\t',  # Tab-separated values as shown in Milano dataset
        has_header=False,
        new_columns=COLUMN_NAMES,
        schema_overrides=COLUMN_DTYPES,
        ignore_errors=True,  # Skips rows with parsing errors
        try_parse_dates=False,  # We'll handle datetime conversion manually
        rechunk=True,  # Optimize memory layout
        null_values=['', 'NULL', 'null'],  # Handle empty values as null
        missing_utf8_is_empty_string=True  # Treat missing strings as empty
    )

    # Define all transformations in a lazy query plan for optimization.
    processed_lazy_df = (
        lazy_df
        # Convert timestamp from milliseconds to datetime.
        .with_columns(
            pl.from_epoch('timestamp_ms', time_unit="ms").alias('timestamp')
        )
        # Drop rows where timestamp conversion failed
        .drop_nulls('timestamp')
        # Drop unnecessary columns
        .drop(['timestamp_ms', 'country_code'])
        # Reorder columns for consistency
        .select(['timestamp', 'cell_id', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic'])
    )

    # Execute the optimized query plan and materialize the DataFrame.
    print("Executing the optimized query plan...")
    try:
        final_df = processed_lazy_df.collect(engine='streaming')
        
        if len(final_df) == 0:
            raise ValueError("No valid data rows were processed from input files")
            
        print(f"Successfully processed {len(final_df):,} rows")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process data: {str(e)}")
    
    # Save the result to a Parquet file with compression.
    print(f"Saving combined data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        final_df.write_parquet(
            output_path,
            compression='zstd',  # Better compression than default
            use_pyarrow=True     # Ensure compatibility
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save parquet file: {str(e)}")
    
    return final_df

def main():
    """Main function to run the data ingestion stage."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 1 - Data Ingestion")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .txt data files")
    parser.add_argument("--output_path", type=Path, default="outputs/01_ingested_data.parquet", help="Output file path")
    args = parser.parse_args()
    
    print("="*60)
    print("Stage 1: Data Ingestion")
    print("="*60)

    start_time = time.perf_counter()
    
    final_df = run_ingestion_stage(args.input_dir, args.output_path)
    
    total_time = time.perf_counter() - start_time
    
    print("\n--- STAGE 1 PERFORMANCE SUMMARY ---")
    print(f"Total rows ingested: {len(final_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processing rate: {len(final_df) / total_time:,.0f} rows/second")
    print("✅ Stage 1 completed successfully.")

if __name__ == "__main__":
    main()