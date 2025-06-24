#!/usr/bin/env python3
"""
demo_cell_aggregation.py

CMMSE 2025: Step 2 - Cell-Level Aggregation Demonstration
Demonstrates the efficiency of Polars vs Pandas for cell-level aggregation.
"""

import polars as pl
import pandas as pd
import time
from pathlib import Path


def main():
    # Load the individual anomalies data
    anomalies_path = Path("./results/04_individual_anomalies.parquet")
    
    print("=" * 70)
    print("Cell-Level Aggregation: Polars vs Pandas Performance Comparison")
    print("=" * 70)
    
    # Test with Polars
    print("\n🚀 POLARS IMPLEMENTATION")
    print("-" * 30)
    
    start_time = time.perf_counter()
    
    # Load data with Polars
    df_polars = pl.read_parquet(anomalies_path)
    print(f"Data loaded: {df_polars.height:,} records")
    
    # Perform aggregation with Polars
    cell_stats_polars = df_polars.with_columns([
        pl.col("cell_id").list.get(0).alias("cell_id_flat")
    ]).drop("cell_id").rename({"cell_id_flat": "cell_id"}).group_by("cell_id").agg([
        # Required metrics from the task
        pl.len().alias("anomaly_count"),
        pl.col("severity_score").mean().alias("avg_severity"),
        pl.col("severity_score").max().alias("max_severity"),
        pl.col("severity_score").std().alias("severity_std"),
        pl.col("sms_total").mean().alias("avg_sms_total"),
        pl.col("calls_total").mean().alias("avg_calls_total"),
        pl.col("internet_traffic").mean().alias("avg_internet_traffic")
    ]).sort("cell_id")
    
    polars_time = time.perf_counter() - start_time
    print(f"Polars processing time: {polars_time:.3f} seconds")
    print(f"Processing rate: {df_polars.height / polars_time:,.0f} records/second")
    print(f"Output shape: {cell_stats_polars.shape}")
    
    # Test with Pandas
    print("\n🐼 PANDAS IMPLEMENTATION")
    print("-" * 30)
    
    start_time = time.perf_counter()
    
    # Load data with Pandas
    df_pandas = pd.read_parquet(anomalies_path)
    print(f"Data loaded: {len(df_pandas):,} records")
    
    # Flatten cell_id column
    df_pandas['cell_id'] = df_pandas['cell_id'].apply(
        lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x
    )
    
    # Perform aggregation with Pandas
    cell_stats_pandas = df_pandas.groupby('cell_id').agg({
        'severity_score': ['count', 'mean', 'max', 'std'],
        'sms_total': 'mean',
        'calls_total': 'mean',
        'internet_traffic': 'mean'
    })
    
    # Flatten column names
    cell_stats_pandas.columns = [
        'anomaly_count', 'avg_severity', 'max_severity', 'severity_std',
        'avg_sms_total', 'avg_calls_total', 'avg_internet_traffic'
    ]
    cell_stats_pandas = cell_stats_pandas.reset_index()
    
    pandas_time = time.perf_counter() - start_time
    print(f"Pandas processing time: {pandas_time:.3f} seconds")
    print(f"Processing rate: {len(df_pandas) / pandas_time:,.0f} records/second")
    print(f"Output shape: {cell_stats_pandas.shape}")
    
    # Performance comparison
    print("\n📊 PERFORMANCE COMPARISON")
    print("-" * 30)
    speedup = pandas_time / polars_time
    print(f"Polars is {speedup:.1f}x faster than Pandas")
    print(f"Time difference: {pandas_time - polars_time:.3f} seconds")
    
    # Show sample results
    print("\n📋 SAMPLE RESULTS (Top 5 cells)")
    print("-" * 40)
    print("\nPolars Results:")
    print(cell_stats_polars.head(5))
    
    print("\nPandas Results:")
    print(cell_stats_pandas.head(5))
    
    # Key metrics summary
    print("\n📈 AGGREGATION SUMMARY")
    print("-" * 25)
    print(f"• Total anomaly records processed: {df_polars.height:,}")
    print(f"• Unique cells with anomalies: {cell_stats_polars.height:,}")
    print(f"• Average anomalies per cell: {df_polars.height / cell_stats_polars.height:.1f}")
    print(f"• Severity range: {cell_stats_polars['avg_severity'].min():.2f}σ to {cell_stats_polars['max_severity'].max():.2f}σ")
    
    print("\n✅ Cell-level aggregation demonstration completed!")


if __name__ == "__main__":
    main()
