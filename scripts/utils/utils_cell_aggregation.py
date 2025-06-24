#!/usr/bin/env python3
"""
utils_cell_aggregation.py

CMMSE 2025: Step 2 - Cell-Level Aggregation Logic
Transform individual anomaly records into cell statistics using efficient Polars operations.
"""

import polars as pl
import pandas as pd
import argparse
import time
from pathlib import Path
import numpy as np


def aggregate_cell_statistics(anomalies_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform individual anomaly records into cell statistics.
    
    Args:
        anomalies_df: DataFrame with individual anomaly records
        
    Returns:
        DataFrame with cell-level aggregated statistics
    """
    
    # First, flatten the cell_id from List(Int64) to Int64
    processed_df = anomalies_df.with_columns([
        pl.col("cell_id").list.get(0).alias("cell_id_flat")
    ]).drop("cell_id").rename({"cell_id_flat": "cell_id"})
    
    # Group by cell_id and calculate required metrics
    cell_stats = processed_df.group_by("cell_id").agg([
        # Anomaly count: count of records per cell
        pl.len().alias("anomaly_count"),
        
        # Severity statistics
        pl.col("severity_score").mean().alias("avg_severity"),
        pl.col("severity_score").max().alias("max_severity"),
        pl.col("severity_score").std().alias("severity_std"),
        
        # Traffic means
        pl.col("sms_total").mean().alias("avg_sms_total"),
        pl.col("calls_total").mean().alias("avg_calls_total"), 
        pl.col("internet_traffic").mean().alias("avg_internet_traffic"),
        
        # Additional useful statistics
        pl.col("anomaly_score").mean().alias("avg_anomaly_score"),
        pl.col("anomaly_score").max().alias("max_anomaly_score"),
        pl.col("timestamp").min().alias("first_anomaly_timestamp"),
        pl.col("timestamp").max().alias("last_anomaly_timestamp")
    ]).sort("cell_id")
    
    # Handle potential null values in severity_std (when only one anomaly per cell)
    cell_stats = cell_stats.with_columns([
        pl.col("severity_std").fill_null(0.0)
    ])
    
    return cell_stats


def aggregate_cell_statistics_pandas(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative implementation using Pandas groupby operations.
    
    Args:
        anomalies_df: DataFrame with individual anomaly records
        
    Returns:
        DataFrame with cell-level aggregated statistics
    """
    
    # Flatten cell_id if it's stored as arrays
    if anomalies_df['cell_id'].dtype == 'object':
        anomalies_df['cell_id'] = anomalies_df['cell_id'].apply(
            lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x
        )
    
    # Group by cell_id and calculate metrics
    cell_stats = anomalies_df.groupby('cell_id').agg({
        # Anomaly count
        'severity_score': ['count', 'mean', 'max', 'std'],
        
        # Traffic means
        'sms_total': 'mean',
        'calls_total': 'mean', 
        'internet_traffic': 'mean',
        
        # Additional statistics
        'anomaly_score': ['mean', 'max'],
        'timestamp': ['min', 'max']
    }).round(6)
    
    # Flatten column names
    cell_stats.columns = [
        'anomaly_count', 'avg_severity', 'max_severity', 'severity_std',
        'avg_sms_total', 'avg_calls_total', 'avg_internet_traffic',
        'avg_anomaly_score', 'max_anomaly_score',
        'first_anomaly_timestamp', 'last_anomaly_timestamp'
    ]
    
    # Fill NaN values in severity_std with 0 (single anomaly cases)
    cell_stats['severity_std'] = cell_stats['severity_std'].fillna(0.0)
    
    # Reset index to make cell_id a column
    cell_stats = cell_stats.reset_index()
    
    return cell_stats


def main():
    parser = argparse.ArgumentParser(
        description="CMMSE 2025: Step 2 - Cell-Level Aggregation Logic"
    )
    parser.add_argument(
        "anomalies_path", 
        type=Path, 
        help="Path to individual anomalies file from Stage 4"
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        default="outputs/06_cell_statistics.parquet",
        help="Output file path for cell statistics"
    )
    parser.add_argument(
        "--use_pandas", 
        action="store_true",
        help="Use Pandas instead of Polars for aggregation"
    )
    parser.add_argument(
        "--preview", 
        action="store_true", 
        help="Show a preview of the output DataFrame"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Cell-Level Aggregation Logic")
    print("=" * 60)
    start_time = time.perf_counter()

    # Load individual anomaly records
    print(f"Loading individual anomaly records from {args.anomalies_path}...")
    
    if args.use_pandas:
        print("Using Pandas for aggregation...")
        anomalies_df = pd.read_parquet(args.anomalies_path)
        print(f"Loaded {len(anomalies_df):,} individual anomaly records.")
        
        # Perform aggregation
        cell_stats = aggregate_cell_statistics_pandas(anomalies_df)
        
    else:
        print("Using Polars for aggregation...")
        anomalies_df = pl.read_parquet(args.anomalies_path)
        print(f"Loaded {anomalies_df.height:,} individual anomaly records.")
        
        # Perform aggregation
        cell_stats = aggregate_cell_statistics(anomalies_df)
    
    # Create output directory if it doesn't exist
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    if args.use_pandas:
        cell_stats.to_parquet(args.output_path, index=False)
        num_cells = len(cell_stats)
    else:
        cell_stats.write_parquet(args.output_path)
        num_cells = cell_stats.height
    
    total_time = time.perf_counter() - start_time
    
    # Show preview if requested
    if args.preview:
        print("\n--- CELL STATISTICS PREVIEW ---")
        if args.use_pandas:
            print(f"Shape: {cell_stats.shape}")
            print("Head(10):")
            print(cell_stats.head(10))
            print("\nSummary Statistics:")
            print(cell_stats.describe())
        else:
            print(f"Shape: {cell_stats.shape}")
            print("Head(10):")
            print(cell_stats.head(10))
            print("\nSummary Statistics:")
            print(cell_stats.describe())
    
    # Performance summary
    print("\n--- AGGREGATION PERFORMANCE SUMMARY ---")
    print(f"Individual anomaly records processed: {len(anomalies_df) if args.use_pandas else anomalies_df.height:,}")
    print(f"Unique cells with statistics: {num_cells:,}")
    print(f"Average anomalies per cell: {(len(anomalies_df) if args.use_pandas else anomalies_df.height) / num_cells:.1f}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processing rate: {(len(anomalies_df) if args.use_pandas else anomalies_df.height) / total_time:,.0f} records/second")
    print(f"Output saved to: {args.output_path}")
    print("✅ Cell-level aggregation completed successfully.")


if __name__ == "__main__":
    main()
