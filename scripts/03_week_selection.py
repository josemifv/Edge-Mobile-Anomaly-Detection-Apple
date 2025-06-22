#!/usr/bin/env python3
"""
03_week_selection.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 3: Reference Week Selection

Selects reference weeks considered as "normal" for each cell using Median Absolute Deviation (MAD) analysis.
These reference weeks will be used for training anomaly detection models in Stage 4.

Usage:
    python scripts/03_week_selection.py <input_path> [--output_path <path>]

Example:
    python scripts/03_week_selection.py data/processed/preprocessed_data.parquet --output_path data/processed/reference_weeks.parquet
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path


def load_preprocessed_data(input_path: str) -> pd.DataFrame:
    """Load the preprocessed data from Stage 2."""
    print(f"Loading data from: {input_path}")
    start_time = time.perf_counter()
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    load_time = time.perf_counter() - start_time
    print(f"  Loaded {len(df):,} rows in {load_time:.2f}s")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Unique cells: {df['cell_id'].nunique():,}")
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features for week-based analysis."""
    print("Adding temporal features...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add ISO week number (proper week numbering)
    iso_calendar = df['timestamp'].dt.isocalendar()
    df['year'] = iso_calendar.year
    df['week'] = iso_calendar.week
    df['year_week'] = df['year'].astype(str) + '_W' + df['week'].astype(str).str.zfill(2)
    
    print(f"  ✓ Added temporal features")
    print(f"  Unique weeks: {df['year_week'].nunique()}")
    print(f"  Week range: {df['year_week'].min()} to {df['year_week'].max()}")
    
    return df


def compute_weekly_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly aggregations per cell."""
    print("Computing weekly aggregations per cell...")
    start_time = time.perf_counter()
    
    # Group by cell_id and year_week, compute statistics
    weekly_agg = df.groupby(['cell_id', 'year_week']).agg({
        'sms_total': ['sum', 'mean', 'std'],
        'calls_total': ['sum', 'mean', 'std'], 
        'internet_traffic': ['sum', 'mean', 'std'],
        'timestamp': ['min', 'max']  # Week boundaries
    }).round(4)
    
    # Flatten column names
    weekly_agg.columns = ['_'.join(col).strip() for col in weekly_agg.columns]
    weekly_agg = weekly_agg.reset_index()
    
    # Rename timestamp columns for clarity
    weekly_agg = weekly_agg.rename(columns={
        'timestamp_min': 'week_start',
        'timestamp_max': 'week_end'
    })
    
    agg_time = time.perf_counter() - start_time
    print(f"  ✓ Weekly aggregations computed in {agg_time:.2f}s")
    print(f"  Weekly data shape: {weekly_agg.shape}")
    
    return weekly_agg


def compute_mad_analysis(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MAD (Median Absolute Deviation) analysis for each cell."""
    print("Computing MAD analysis per cell...")
    start_time = time.perf_counter()
    
    # Activity columns to analyze
    activity_columns = [
        'sms_total_sum', 'calls_total_sum', 'internet_traffic_sum'
    ]
    
    mad_results = []
    
    for cell_id in weekly_df['cell_id'].unique():
        cell_data = weekly_df[weekly_df['cell_id'] == cell_id].copy()
        
        # Skip cells with insufficient data
        if len(cell_data) < 2:
            continue
            
        for col in activity_columns:
            if col in cell_data.columns:
                values = cell_data[col].values
                
                # Remove NaN and infinite values
                values = values[np.isfinite(values)]
                
                if len(values) > 0:
                    median_val = np.median(values)
                    mad_val = np.median(np.abs(values - median_val))
                    
                    # Calculate normalized deviation for each week
                    for _, week_row in cell_data.iterrows():
                        activity_val = week_row[col]
                        if np.isfinite(activity_val):
                            norm_deviation = (activity_val - median_val) / (mad_val + 1e-8)
                            
                            mad_results.append({
                                'cell_id': cell_id,
                                'year_week': week_row['year_week'],
                                'activity_type': col,
                                'activity_value': activity_val,
                                'median_value': median_val,
                                'mad_value': mad_val,
                                'normalized_deviation': norm_deviation
                            })
    
    mad_df = pd.DataFrame(mad_results)
    
    mad_time = time.perf_counter() - start_time
    print(f"  ✓ MAD analysis completed in {mad_time:.2f}s")
    print(f"  MAD data shape: {mad_df.shape}")
    
    return mad_df


def select_reference_weeks(mad_df: pd.DataFrame, num_weeks: int = 4, mad_threshold: float = 1.5) -> pd.DataFrame:
    """Select reference weeks per cell based on MAD analysis."""
    print(f"Selecting {num_weeks} reference weeks per cell (MAD threshold: {mad_threshold})...")
    
    reference_weeks = []
    
    for cell_id in mad_df['cell_id'].unique():
        cell_mad = mad_df[mad_df['cell_id'] == cell_id]
        
        # Find weeks where all activities are below the MAD threshold
        week_scores = []
        for week in cell_mad['year_week'].unique():
            week_data = cell_mad[cell_mad['year_week'] == week]
            
            # Calculate average absolute deviation for this week
            avg_abs_deviation = week_data['normalized_deviation'].abs().mean()
            
            # Check if all deviations are below threshold
            all_below_threshold = (week_data['normalized_deviation'].abs() <= mad_threshold).all()
            
            week_scores.append({
                'cell_id': cell_id,
                'year_week': week,
                'avg_abs_deviation': avg_abs_deviation,
                'is_normal': all_below_threshold
            })
        
        # Sort weeks by average deviation (most normal first)
        week_scores_df = pd.DataFrame(week_scores)
        normal_weeks = week_scores_df[week_scores_df['is_normal']].sort_values('avg_abs_deviation')
        
        # Select top N normal weeks, or all available if fewer than N
        selected_weeks = normal_weeks.head(num_weeks)
        
        for _, week_row in selected_weeks.iterrows():
            reference_weeks.append({
                'cell_id': week_row['cell_id'],
                'reference_week': week_row['year_week'],
                'avg_deviation': week_row['avg_abs_deviation'],
                'selection_rank': len(reference_weeks) % num_weeks + 1
            })
    
    reference_df = pd.DataFrame(reference_weeks)
    
    print(f"  ✓ Selected {len(reference_df)} reference weeks")
    print(f"  Cells with reference weeks: {reference_df['cell_id'].nunique()}")
    print(f"  Average weeks per cell: {len(reference_df) / reference_df['cell_id'].nunique():.1f}")
    
    return reference_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 3 - Week Selection")
    parser.add_argument("input_path", help="Path to preprocessed data file")
    parser.add_argument("--output_path", default="data/processed/reference_weeks.parquet",
                       help="Output file path")
    parser.add_argument("--num_weeks", type=int, default=4, 
                       help="Number of reference weeks per cell")
    parser.add_argument("--mad_threshold", type=float, default=1.5,
                       help="MAD threshold for normal weeks")
    parser.add_argument("--preview", action="store_true", help="Show data preview")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 3: Reference Week Selection")
    print("="*60)
    
    start_time = time.perf_counter()
    
    try:
        # Load preprocessed data
        df = load_preprocessed_data(args.input_path)
        
        # Add temporal features
        df = add_temporal_features(df)
        
        # Compute weekly aggregations
        weekly_df = compute_weekly_aggregations(df)
        
        # Compute MAD analysis
        mad_df = compute_mad_analysis(weekly_df)
        
        # Select reference weeks
        reference_weeks = select_reference_weeks(
            mad_df, 
            num_weeks=args.num_weeks,
            mad_threshold=args.mad_threshold
        )
        
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save reference weeks
        print(f"Saving to {args.output_path}...")
        if output_path.suffix.lower() == '.parquet':
            reference_weeks.to_parquet(args.output_path, index=False)
        else:
            reference_weeks.to_csv(args.output_path, index=False)
        
        # Show preview if requested
        if args.preview:
            print("\n" + "="*40)
            print("REFERENCE WEEKS PREVIEW")
            print("="*40)
            reference_weeks.info()
            print("\nFirst 10 reference weeks:")
            print(reference_weeks.head(10))
            print("\nReference weeks per cell distribution:")
            print(reference_weeks.groupby('cell_id').size().describe())
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print("="*60)
        print("STAGE 3 SUMMARY")
        print("="*60)
        print(f"Input data rows: {len(df):,}")
        print(f"Weekly aggregations: {len(weekly_df):,}")
        print(f"Reference weeks selected: {len(reference_weeks):,}")
        print(f"Cells with reference weeks: {reference_weeks['cell_id'].nunique():,}")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Output file: {args.output_path}")
        print(f"Output size: {file_size_mb:.1f} MB")
        print("Stage 3 completed successfully!")
        
    except Exception as e:
        print(f"Stage 3 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
