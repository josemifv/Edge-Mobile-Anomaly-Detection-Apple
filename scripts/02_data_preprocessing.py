#!/usr/bin/env python3
"""
02_data_preprocessing.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline  
Stage 2: Data Preprocessing and Aggregation

Performs data aggregation, consolidation, and validation on ingested data.
Groups data by cell_id and timestamp, merges directional columns, and validates data quality.

Usage:
    python scripts/02_data_preprocessing.py <input_path> [--output_path <path>]

Example:
    python scripts/02_data_preprocessing.py data/processed/ingested_data.parquet --output_path data/processed/preprocessed_data.parquet
"""

import pandas as pd
import argparse
import time
from pathlib import Path

def load_ingested_data(input_path: str) -> pd.DataFrame:
    """Load the ingested data from Stage 1."""
    print(f"Loading data from: {input_path}")
    start_time = time.perf_counter()
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    load_time = time.perf_counter() - start_time
    print(f"  Loaded {len(df):,} rows in {load_time:.2f}s")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by aggregating and merging columns.
    
    Steps:
    1. Group by cell_id and timestamp 
    2. Aggregate numeric columns using sum
    3. Merge directional columns (SMS in/out -> SMS total, calls in/out -> calls total)
    4. Remove individual directional columns
    """
    print("Preprocessing data...")
    start_time = time.perf_counter()
    
    print("  Aggregating by cell_id and timestamp...")
    # Group and aggregate
    df_agg = df.groupby(['cell_id', 'timestamp']).agg({
        'sms_in': 'sum',
        'sms_out': 'sum', 
        'call_in': 'sum',
        'call_out': 'sum',
        'internet_traffic': 'sum'
    }).reset_index()
    
    print("  Merging directional columns...")
    # Create total columns
    df_agg['sms_total'] = df_agg['sms_in'] + df_agg['sms_out']
    df_agg['calls_total'] = df_agg['call_in'] + df_agg['call_out']
    
    # Drop individual directional columns
    df_final = df_agg.drop(['sms_in', 'sms_out', 'call_in', 'call_out'], axis=1)
    
    # Reorder columns
    df_final = df_final[['cell_id', 'timestamp', 'sms_total', 'calls_total', 'internet_traffic']]
    
    process_time = time.perf_counter() - start_time
    print(f"  Preprocessing completed in {process_time:.2f}s")
    print(f"  Rows after aggregation: {len(df_final):,}")
    
    return df_final


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the preprocessed data for quality issues.
    
    Checks:
    1. No duplicate rows (cell_id + timestamp combinations)
    2. No null values anywhere in the dataset
    """
    print("Validating data quality...")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['cell_id', 'timestamp']).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate rows based on cell_id + timestamp")
    print("  ✓ No duplicates found")
    
    # Check for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print("  Null value counts by column:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"    {col}: {count}")
        raise ValueError(f"Found {total_nulls} null values in dataset")
    print("  ✓ No null values found")
    
    print("  ✓ Data validation completed successfully")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 2 - Data Preprocessing")
    parser.add_argument("input_path", help="Path to ingested data file")
    parser.add_argument("--output_path", default="data/processed/preprocessed_data.parquet",
                       help="Output file path")
    parser.add_argument("--preview", action="store_true", help="Show data preview")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 2: Data Preprocessing")
    print("="*60)
    
    start_time = time.perf_counter()
    
    try:
        # Load data from Stage 1
        df = load_ingested_data(args.input_path)
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Validate data quality
        validate_data(df_processed)
        
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        print(f"Saving to {args.output_path}...")
        if output_path.suffix.lower() == '.parquet':
            df_processed.to_parquet(args.output_path, index=False)
        else:
            df_processed.to_csv(args.output_path, index=False)
        
        # Show preview if requested
        if args.preview:
            print("\n" + "="*40)
            print("DATA PREVIEW")
            print("="*40)
            print(df_processed.info())
            print("\nFirst 5 rows:")
            print(df_processed.head())
            print("\nData statistics:")
            print(df_processed.describe())
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print("="*60)
        print("STAGE 2 SUMMARY")
        print("="*60)
        print(f"Input rows: {len(df):,}")
        print(f"Output rows: {len(df_processed):,}")
        print(f"Compression: {((len(df) - len(df_processed)) / len(df)) * 100:.1f}%")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
        print(f"Unique cells: {df_processed['cell_id'].nunique():,}")
        print(f"Output file: {args.output_path}")
        print(f"Output size: {file_size_mb:.1f} MB")
        print("Stage 2 completed successfully!")
        
    except Exception as e:
        print(f"Stage 2 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
