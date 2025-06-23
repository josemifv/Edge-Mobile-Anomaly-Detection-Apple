#!/usr/bin/env python3
"""
02_data_preprocessing.py

CMMSE 2025: Stage 2 - Data Preprocessing and Aggregation
Performs high-performance data aggregation and validation on ingested data.
"""

import polars as pl
import argparse
import time
from pathlib import Path

def run_preprocessing_stage(input_path: Path, output_path: Path) -> pl.DataFrame:
    """Preprocesses the data by aggregating, merging columns, and validating."""
    print(f"Starting lazy preprocessing from: {input_path}")
    lazy_df = pl.scan_parquet(input_path)
    preprocessed_lazy_df = (
        lazy_df
        .group_by(['cell_id', 'timestamp'])
        .agg([pl.sum('sms_in'), pl.sum('sms_out'), pl.sum('call_in'), pl.sum('call_out'), pl.sum('internet_traffic')])
        .with_columns([(pl.col('sms_in') + pl.col('sms_out')).alias('sms_total'), (pl.col('call_in') + pl.col('call_out')).alias('calls_total')])
        .select(['cell_id', 'timestamp', 'sms_total', 'calls_total', 'internet_traffic'])
    )
    print("Executing the optimized query plan...")
    final_df = preprocessed_lazy_df.collect(streaming=True)

    print("Validating data quality...")
    duplicates = final_df.select(pl.struct(['cell_id', 'timestamp']).is_duplicated()).sum().item()
    if duplicates > 0: raise ValueError(f"Found {duplicates} duplicate rows")
    print("  ✓ No duplicates found")
    if final_df.null_count().sum_horizontal().item() > 0: raise ValueError("Found null values")
    print("  ✓ No null values found")

    print(f"Saving preprocessed data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(output_path)
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 2 - Data Preprocessing")
    parser.add_argument("input_path", type=Path, help="Path to ingested data file from Stage 1")
    parser.add_argument("--output_path", type=Path, default="outputs/02_preprocessed_data.parquet", help="Output file path")
    parser.add_argument("--preview", action="store_true", help="Show a preview of the output DataFrame.")
    args = parser.parse_args()

    print("="*60); print("Stage 2: Data Preprocessing"); print("="*60)
    start_time = time.perf_counter()
    final_df = run_preprocessing_stage(args.input_path, args.output_path)
    total_time = time.perf_counter() - start_time
    
    if args.preview:
        print("\n--- DATA PREVIEW ---")
        print(f"Shape: {final_df.shape}"); print("Head(5):"); print(final_df.head(5))

    print("\n--- STAGE 2 PERFORMANCE SUMMARY ---")
    print(f"Output rows: {len(final_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    if total_time > 0: print(f"Processing rate: {len(final_df) / total_time:,.0f} rows/second")
    print("✅ Stage 2 completed successfully.")

if __name__ == "__main__":
    main()