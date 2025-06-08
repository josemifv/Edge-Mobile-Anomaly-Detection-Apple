#!/usr/bin/env python3
"""
Week Selection for Anomaly Detection Pipeline

This script computes Median Absolute Deviation (MAD) per cell and week,
then selects reference weeks considered as "normal" for anomaly detection.

Usage:
    python scripts/03_week_selection.py <input_parquet> [options]

Author: Edge Mobile Anomaly Detection Project
Target: Apple Silicon optimization
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(input_path):
    """
    Load the processed telecom data from Parquet file.
    
    Args:
        input_path (str): Path to the input Parquet file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    print(f"Loading processed data from: {input_path}")
    start_time = time.perf_counter()
    
    df = pd.read_parquet(input_path)
    
    load_time = time.perf_counter() - start_time
    print(f"✓ Data loaded in {load_time:.2f}s")
    print(f"  Shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def add_temporal_features(df):
    """
    Add temporal features for week-based analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp column
        
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    print("Adding temporal features...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add week number (ISO week) - using ISO year for correct week numbering
    iso_calendar = df['timestamp'].dt.isocalendar()
    df['year'] = iso_calendar.year  # Use ISO year, not calendar year
    df['week'] = iso_calendar.week
    df['year_week'] = df['year'].astype(str) + '_W' + df['week'].astype(str).str.zfill(2)
    
    # Add day of week and hour for additional analysis
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['hour'] = df['timestamp'].dt.hour
    
    print(f"✓ Temporal features added")
    print(f"  Unique weeks: {df['year_week'].nunique()}")
    print(f"  Week range: {df['year_week'].min()} to {df['year_week'].max()}")
    
    return df


def compute_weekly_aggregations(df):
    """
    Compute weekly aggregations per cell.
    
    Args:
        df (pd.DataFrame): Input DataFrame with temporal features
        
    Returns:
        pd.DataFrame: Weekly aggregated data per cell
    """
    print("Computing weekly aggregations per cell...")
    start_time = time.perf_counter()
    
    # Group by cell_id and year_week, compute aggregations
    weekly_agg = df.groupby(['cell_id', 'year_week']).agg({
        'sms_total': ['sum', 'mean', 'std', 'count'],
        'calls_total': ['sum', 'mean', 'std', 'count'], 
        'internet_traffic': ['sum', 'mean', 'std', 'count'],
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
    print(f"✓ Weekly aggregations computed in {agg_time:.2f}s")
    print(f"  Shape: {weekly_agg.shape}")
    print(f"  Cells with data: {weekly_agg['cell_id'].nunique()}")
    
    return weekly_agg


def compute_mad_per_cell_week(weekly_df):
    """
    Compute Median Absolute Deviation (MAD) for each activity type per cell and week.
    
    Args:
        weekly_df (pd.DataFrame): Weekly aggregated data
        
    Returns:
        pd.DataFrame: DataFrame with MAD values per cell and week
    """
    print("Computing MAD (Median Absolute Deviation) per cell and week...")
    start_time = time.perf_counter()
    
    # Activity columns to compute MAD for
    activity_columns = [
        'sms_total_sum', 'sms_total_mean',
        'calls_total_sum', 'calls_total_mean', 
        'internet_traffic_sum', 'internet_traffic_mean'
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
                    
                    # Add MAD data for each week
                    for _, week_row in cell_data.iterrows():
                        mad_results.append({
                            'cell_id': cell_id,
                            'year_week': week_row['year_week'],
                            'activity_type': col,
                            'activity_value': week_row[col],
                            'median_value': median_val,
                            'mad_value': mad_val,
                            'normalized_deviation': (week_row[col] - median_val) / (mad_val + 1e-8)  # Avoid division by zero
                        })
    
    mad_df = pd.DataFrame(mad_results)
    
    mad_time = time.perf_counter() - start_time
    print(f"✓ MAD computation completed in {mad_time:.2f}s")
    print(f"  MAD data shape: {mad_df.shape}")
    
    return mad_df


def select_reference_weeks_per_cell(mad_df, weekly_df, num_reference_weeks=4, mad_threshold=1.5):
    """
    Select reference weeks per cell considered as "normal" based on MAD analysis.
    
    Args:
        mad_df (pd.DataFrame): MAD analysis results
        weekly_df (pd.DataFrame): Weekly aggregated data
        num_reference_weeks (int): Number of reference weeks to select per cell
        mad_threshold (float): MAD threshold for considering weeks as normal
        
    Returns:
        tuple: (cell_reference_weeks_df, summary_stats)
    """
    print(f"Selecting {num_reference_weeks} reference weeks per cell (MAD threshold: {mad_threshold})...")
    
    cell_reference_weeks = []
    cell_stats = []
    
    unique_cells = mad_df['cell_id'].unique()
    processed_cells = 0
    
    for cell_id in unique_cells:
        cell_mad_data = mad_df[mad_df['cell_id'] == cell_id]
        
        # Calculate average normalized deviation per week for this cell
        week_scores = cell_mad_data.groupby('year_week').agg({
            'normalized_deviation': ['mean', 'std', 'count'],
            'mad_value': 'mean'
        }).round(4)
        
        week_scores.columns = ['avg_norm_dev', 'std_norm_dev', 'count_measurements', 'avg_mad']
        week_scores = week_scores.reset_index()
        week_scores['cell_id'] = cell_id
        
        # Score weeks by stability (low average deviation and low std)
        week_scores['stability_score'] = 1 / (week_scores['avg_norm_dev'].abs() + week_scores['std_norm_dev'] + 1e-8)
        
        # Filter weeks that are below the MAD threshold
        normal_weeks = week_scores[week_scores['avg_norm_dev'].abs() <= mad_threshold]
        
        # Select reference weeks for this cell
        if len(normal_weeks) >= num_reference_weeks:
            selected_weeks = normal_weeks.nlargest(num_reference_weeks, 'stability_score')
        else:
            # If not enough normal weeks, take the most stable available weeks
            selected_weeks = week_scores.nlargest(min(num_reference_weeks, len(week_scores)), 'stability_score')
        
        # Store reference weeks for this cell
        for _, week_row in selected_weeks.iterrows():
            cell_reference_weeks.append({
                'cell_id': cell_id,
                'reference_week': week_row['year_week'],
                'stability_score': week_row['stability_score'],
                'avg_norm_dev': week_row['avg_norm_dev'],
                'std_norm_dev': week_row['std_norm_dev'],
                'count_measurements': week_row['count_measurements']
            })
        
        # Store cell statistics
        cell_stats.append({
            'cell_id': cell_id,
            'total_weeks_available': len(week_scores),
            'normal_weeks_found': len(normal_weeks),
            'reference_weeks_selected': len(selected_weeks),
            'avg_stability_score': selected_weeks['stability_score'].mean()
        })
        
        processed_cells += 1
        if processed_cells % 1000 == 0:
            print(f"  Processed {processed_cells}/{len(unique_cells)} cells...")
    
    reference_weeks_df = pd.DataFrame(cell_reference_weeks)
    cell_stats_df = pd.DataFrame(cell_stats)
    
    print(f"✓ Reference week selection completed")
    print(f"  Processed {len(unique_cells)} cells")
    print(f"  Average reference weeks per cell: {reference_weeks_df.groupby('cell_id').size().mean():.2f}")
    print(f"  Cells with {num_reference_weeks} reference weeks: {(cell_stats_df['reference_weeks_selected'] == num_reference_weeks).sum()}")
    
    return reference_weeks_df, cell_stats_df


def generate_week_selection_report(weekly_df, mad_df, reference_weeks_df, cell_stats_df, output_dir):
    """
    Generate a detailed report of the week selection process.
    
    Args:
        weekly_df (pd.DataFrame): Weekly aggregated data
        mad_df (pd.DataFrame): MAD analysis results
        reference_weeks_df (pd.DataFrame): Selected reference weeks per cell
        cell_stats_df (pd.DataFrame): Cell statistics
        output_dir (str): Output directory for reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"week_selection_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("WEEK SELECTION ANALYSIS REPORT (PER CELL)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"  Total cells: {weekly_df['cell_id'].nunique()}\n")
        f.write(f"  Total weeks: {weekly_df['year_week'].nunique()}\n")
        f.write(f"  Week range: {weekly_df['year_week'].min()} to {weekly_df['year_week'].max()}\n")
        f.write(f"  Total weekly observations: {len(weekly_df)}\n\n")
        
        f.write("MAD ANALYSIS RESULTS:\n")
        f.write(f"  Total MAD measurements: {len(mad_df)}\n")
        f.write(f"  Average MAD value: {mad_df['mad_value'].mean():.4f}\n")
        f.write(f"  Average normalized deviation: {mad_df['normalized_deviation'].mean():.4f}\n\n")
        
        f.write("REFERENCE WEEK SELECTION SUMMARY:\n")
        f.write(f"  Total reference weeks selected: {len(reference_weeks_df)}\n")
        f.write(f"  Average reference weeks per cell: {reference_weeks_df.groupby('cell_id').size().mean():.2f}\n")
        f.write(f"  Cells processed: {len(cell_stats_df)}\n")
        
        req_weeks = cell_stats_df['reference_weeks_selected'].iloc[0] if len(cell_stats_df) > 0 else 0
        f.write(f"  Cells with {req_weeks} reference weeks: {(cell_stats_df['reference_weeks_selected'] == req_weeks).sum()}\n")
        f.write(f"  Average stability score: {reference_weeks_df['stability_score'].mean():.4f}\n\n")
        
        f.write("WEEK DISTRIBUTION:\n")
        week_counts = reference_weeks_df['reference_week'].value_counts().sort_index()
        for week, count in week_counts.items():
            f.write(f"  {week}: {count} cells ({count/len(cell_stats_df)*100:.1f}%)\n")
        
        f.write("\nCELL STATISTICS SUMMARY:\n")
        f.write(f"  Min weeks available per cell: {cell_stats_df['total_weeks_available'].min()}\n")
        f.write(f"  Max weeks available per cell: {cell_stats_df['total_weeks_available'].max()}\n")
        f.write(f"  Average weeks available per cell: {cell_stats_df['total_weeks_available'].mean():.2f}\n")
        f.write(f"  Average normal weeks found per cell: {cell_stats_df['normal_weeks_found'].mean():.2f}\n")
    
    print(f"✓ Week selection report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Week Selection for Anomaly Detection - Compute MAD and select reference weeks"
    )
    parser.add_argument(
        "input_path",
        help="Path to the processed Parquet file"
    )
    parser.add_argument(
        "--num_reference_weeks",
        type=int,
        default=3,
        help="Number of reference weeks to select (default: 3)"
    )
    parser.add_argument(
        "--mad_threshold",
        type=float,
        default=1.5,
        help="MAD threshold for normal week selection (default: 1.5)"
    )
    parser.add_argument(
        "--output_dir",
        default="reports",
        help="Output directory for reports (default: reports)"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate results (weekly aggregations, MAD data)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("WEEK SELECTION FOR ANOMALY DETECTION")
    print("=" * 50)
    print(f"Input: {input_path}")
    print(f"Reference weeks to select: {args.num_reference_weeks}")
    print(f"MAD threshold: {args.mad_threshold}")
    print(f"Output directory: {output_dir}")
    print()
    
    start_time = time.perf_counter()
    
    try:
        # Load processed data
        df = load_processed_data(input_path)
        
        # Add temporal features
        df = add_temporal_features(df)
        
        # Compute weekly aggregations
        weekly_df = compute_weekly_aggregations(df)
        
        # Compute MAD per cell and week
        mad_df = compute_mad_per_cell_week(weekly_df)
        
        # Select reference weeks per cell
        reference_weeks_df, cell_stats_df = select_reference_weeks_per_cell(
            mad_df, weekly_df, args.num_reference_weeks, args.mad_threshold
        )
        
        # Generate report
        generate_week_selection_report(
            weekly_df, mad_df, reference_weeks_df, cell_stats_df, output_dir
        )
        
        # Save intermediate results if requested
        if args.save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            weekly_path = output_dir / f"weekly_aggregations_{timestamp}.parquet"
            weekly_df.to_parquet(weekly_path, index=False)
            print(f"✓ Weekly aggregations saved: {weekly_path}")
            
            mad_path = output_dir / f"mad_analysis_{timestamp}.parquet"
            mad_df.to_parquet(mad_path, index=False)
            print(f"✓ MAD analysis saved: {mad_path}")
            
            ref_path = output_dir / f"reference_weeks_{timestamp}.parquet"
            reference_weeks_df.to_parquet(ref_path, index=False)
            print(f"✓ Reference weeks per cell saved: {ref_path}")
            
            cell_stats_path = output_dir / f"cell_stats_{timestamp}.parquet"
            cell_stats_df.to_parquet(cell_stats_path, index=False)
            print(f"✓ Cell statistics saved: {cell_stats_path}")
        
        total_time = time.perf_counter() - start_time
        print(f"\n✓ Week selection completed in {total_time:.2f}s")
        print(f"Reference weeks per cell: {reference_weeks_df.groupby('cell_id').size().mean():.2f} avg per cell")
        
    except Exception as e:
        print(f"Error during week selection: {str(e)}")
        raise


if __name__ == "__main__":
    main()

