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

COLUMN_NAMES = [
    'cell_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_traffic'
]
COLUMN_DTYPES = {
    'cell_id': pl.Int64, 'timestamp_ms': pl.Int64, 'country_code': pl.Int32,
    'sms_in': pl.Float64, 'sms_out': pl.Float64, 'call_in': pl.Float64,
    'call_out': pl.Float64, 'internet_traffic': pl.Float64
}

def run_ingestion_stage(input_dir: Path, output_path: Path) -> pl.DataFrame:
    """Ingests all .txt files from a directory using a lazy engine for high performance."""
    print(f"Starting parallel ingestion from: {input_dir}")
    
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    print(f"Found {len(txt_files)} files to process")
    
    lazy_df = pl.scan_csv(
        source=str(input_dir / "*.txt"),
        separator='\t',  # Tab-separated values
        has_header=False,
        new_columns=COLUMN_NAMES,
        schema_overrides=COLUMN_DTYPES,
        ignore_errors=True,
        try_parse_dates=False,
        rechunk=True,
        null_values=['', 'NULL', 'null'],
        missing_utf8_is_empty_string=True
    )
    processed_lazy_df = (
        lazy_df
        .with_columns(pl.from_epoch('timestamp_ms', time_unit="ms").alias('timestamp'))
        .drop_nulls('timestamp')
        .drop(['timestamp_ms', 'country_code'])
        .select(['timestamp', 'cell_id', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic'])
    )
    print("Executing the optimized query plan...")
    try:
        final_df = processed_lazy_df.collect(engine='streaming')
        
        if len(final_df) == 0:
            raise ValueError("No valid data rows were processed from input files")
            
        print(f"Successfully processed {len(final_df):,} rows")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process data: {str(e)}")
    
    print(f"Saving combined data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        final_df.write_parquet(
            output_path,
            compression='zstd',
            use_pyarrow=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save parquet file: {str(e)}")
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 1 - Data Ingestion")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .txt data files")
    parser.add_argument("--output_path", type=Path, default="outputs/01_ingested_data.parquet", help="Output file path")
    parser.add_argument("--preview", action="store_true", help="Show a preview of the output DataFrame.")
    args = parser.parse_args()
    
    print("="*60); print("Stage 1: Data Ingestion"); print("="*60)
    start_time = time.perf_counter()
    final_df = run_ingestion_stage(args.input_dir, args.output_path)
    total_time = time.perf_counter() - start_time
    
    if args.preview:
        print("\n--- DATA PREVIEW ---")
        print(f"Shape: {final_df.shape}"); print("Head(5):"); print(final_df.head(5))
        print("Schema:"); print(final_df.schema)

    print("\n--- STAGE 1 PERFORMANCE SUMMARY ---")
    print(f"Total rows ingested: {len(final_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    if total_time > 0: print(f"Processing rate: {len(final_df) / total_time:,.0f} rows/second")
    print("✅ Stage 1 completed successfully.")

if __name__ == "__main__":
    main()