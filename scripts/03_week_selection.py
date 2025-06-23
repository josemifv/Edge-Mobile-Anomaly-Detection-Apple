#!/usr/bin/env python3
"""
03_week_selection.py

CMMSE 2025: Stage 3 - Reference Week Selection
Performs reference week selection using Median Absolute Deviation (MAD) analysis.
"""

import polars as pl
import argparse
import time
from pathlib import Path

def run_week_selection_stage(input_path: Path, output_path: Path, num_weeks: int, mad_threshold: float):
    """Performs the complete week selection stage using a lazy-evaluation engine."""
    print(f"Starting Stage 3 from: {input_path}")
    lazy_df = pl.scan_parquet(input_path)
    temporal_lazy_df = lazy_df.with_columns([
        pl.col("timestamp").dt.iso_year().alias("year"),
        pl.col("timestamp").dt.week().alias("week")
    ]).with_columns(
        (pl.col("year").cast(pl.Utf8) + "_W" + pl.col("week").cast(pl.Utf8).str.zfill(2)).alias("year_week")
    )
    weekly_agg_lazy_df = temporal_lazy_df.group_by(['cell_id', 'year_week']).agg([
        pl.sum('sms_total'), pl.sum('calls_total'), pl.sum('internet_traffic')
    ])
    
    weekly_df = weekly_agg_lazy_df.collect()
    cols = ['sms_total', 'calls_total', 'internet_traffic']
    mad_df = weekly_df.with_columns([pl.median(col).over('cell_id').alias(f"{col}_median") for col in cols])
    mad_df = mad_df.with_columns([(pl.col(col) - pl.col(f"{col}_median")).abs().alias(f"{col}_abs_dev") for col in cols])
    mad_df = mad_df.with_columns([pl.median(f"{col}_abs_dev").over('cell_id').alias(f"{col}_mad") for col in cols])
    mad_df = mad_df.with_columns([(pl.col(f"{col}_abs_dev") / (pl.col(f"{col}_mad") + 1e-8)).alias(f"{col}_norm_dev") for col in cols])

    normal_weeks_df = mad_df.with_columns(
        pl.when((pl.col('sms_total_norm_dev') <= mad_threshold) & (pl.col('calls_total_norm_dev') <= mad_threshold) & (pl.col('internet_traffic_norm_dev') <= mad_threshold))
        .then(True).otherwise(False).alias('is_normal')
    ).filter(pl.col('is_normal'))

    final_df = normal_weeks_df.with_columns(
        (pl.col('sms_total_norm_dev') + pl.col('calls_total_norm_dev') + pl.col('internet_traffic_norm_dev')).alias('stability_score')
    ).sort('stability_score').group_by('cell_id', maintain_order=True).head(num_weeks).select(['cell_id', 'year_week', 'stability_score'])
    
    final_df = final_df.rename({"year_week": "reference_week"})
    
    print(f"Saving reference weeks to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(output_path)
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 3 - Reference Week Selection")
    parser.add_argument("input_path", type=Path, help="Path to preprocessed data file")
    parser.add_argument("--output_path", type=Path, default="outputs/03_reference_weeks.parquet", help="Output file path")
    parser.add_argument("--num_weeks", type=int, default=4, help="Number of reference weeks per cell")
    parser.add_argument("--mad_threshold", type=float, default=1.5, help="MAD threshold for normal weeks")
    parser.add_argument("--preview", action="store_true", help="Show a preview of the output DataFrame.")
    args = parser.parse_args()

    print("="*60); print("Stage 3: Reference Week Selection"); print("="*60)
    start_time = time.perf_counter()
    final_df = run_week_selection_stage(args.input_path, args.output_path, args.num_weeks, args.mad_threshold)
    total_time = time.perf_counter() - start_time

    if args.preview:
        print("\n--- DATA PREVIEW ---")
        print(f"Shape: {final_df.shape}"); print("Head(5):"); print(final_df.head(5))
        print("\nReference weeks per cell distribution:"); print(final_df['cell_id'].value_counts().describe())

    print("\n--- STAGE 3 PERFORMANCE SUMMARY ---")
    print(f"Reference weeks selected: {len(final_df):,}")
    print(f"Cells with reference weeks: {final_df['cell_id'].n_unique():,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("✅ Stage 3 completed successfully.")

if __name__ == "__main__":
    main()