#!/usr/bin/env python3
"""
04_anomaly_detection.py

CMMSE 2025: Stage 4 - OSP Anomaly Detection
Optimized version using a hybrid Polars/NumPy approach for high performance.
"""

import polars as pl
import pandas as pd
import numpy as np
import argparse
import time
import multiprocessing
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']

class OSPDetector:
    """OSP Anomaly Detector using float32 precision and vectorized outputs."""
    def __init__(self, n_components: int = 3, anomaly_threshold: float = 2.0, standardize: bool = True):
        self.n_components, self.anomaly_threshold, self.standardize = n_components, anomaly_threshold, standardize
        self.scaler = StandardScaler() if standardize else None
        self.svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm='randomized', n_iter=3)
        self.error_threshold, self.training_mean_error, self.training_std_error = None, None, None
        
    def fit(self, X: np.ndarray):
        X_fit = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_fit) if self.standardize else X_fit
        self.svd.fit(X_scaled)
        X_reconstructed = self.svd.inverse_transform(self.svd.transform(X_scaled))
        reconstruction_errors = np.linalg.norm(X_scaled - X_reconstructed, axis=1)
        self.training_mean_error, self.training_std_error = np.mean(reconstruction_errors), np.std(reconstruction_errors)
        self.error_threshold = self.training_mean_error + self.anomaly_threshold * self.training_std_error
        
    def predict(self, X: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
        if self.error_threshold is None: raise ValueError("Model must be fitted before prediction.")
        X_predict = X.astype(np.float32)
        X_scaled = self.scaler.transform(X_predict) if self.standardize else X_predict
        X_reconstructed = self.svd.inverse_transform(self.svd.transform(X_scaled))
        anomaly_scores = np.linalg.norm(X_scaled - X_reconstructed, axis=1)
        
        # Ensure X is always 2D for consistent indexing
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Build DataFrame with consistent array shapes
        data_dict = {
            'timestamp': timestamps, 
            'anomaly_score': anomaly_scores
        }
        
        # Add feature columns safely
        for i, col in enumerate(FEATURE_COLUMNS):
            if i < X.shape[1]:
                data_dict[col] = X[:, i]
            else:
                data_dict[col] = np.zeros(X.shape[0])
        
        results_df = pd.DataFrame(data_dict)
        anomalies_df = results_df[results_df['anomaly_score'] > self.error_threshold].copy().reset_index(drop=True)

        if not anomalies_df.empty:
            # Calculate severity scores, ensuring proper broadcasting and handling edge cases
            severity_scores = (anomalies_df['anomaly_score'].values - self.training_mean_error)
            
            # Handle division by zero case (when training_std_error is 0)
            if self.training_std_error != 0:
                severity_scores = severity_scores / self.training_std_error
            else:
                # If std is 0, all training errors were identical, so use a default severity
                severity_scores = severity_scores  # Just use the difference from mean
            
            # Assign severity scores directly as numpy array to avoid index alignment issues
            anomalies_df['severity_score'] = severity_scores
        return anomalies_df

def process_single_cell(cell_tuple: tuple) -> pd.DataFrame:
    """Processes a single cell's data to detect anomalies."""
    cell_id, cell_df = cell_tuple
    detector_params = {'n_components': 3, 'anomaly_threshold': 2.0} # Placeholder, should be passed
    try:
        training_df = cell_df.filter(pl.col('is_reference'))
        if training_df.height == 0: return pd.DataFrame()
        X_train = training_df.select(FEATURE_COLUMNS).to_numpy()
        X_test = cell_df.select(FEATURE_COLUMNS).to_numpy()
        timestamps_test = cell_df['timestamp'].to_numpy()

        detector = OSPDetector(**detector_params)
        detector.fit(X_train)
        anomalies_df = detector.predict(X_test, timestamps_test)
        
        if not anomalies_df.empty:
            # Ensure proper assignment by creating a column with the right length
            anomalies_df = anomalies_df.copy()
            anomalies_df['cell_id'] = [cell_id] * len(anomalies_df)
            # Reorder columns to match expected output format
            output_columns = ['cell_id', 'timestamp', 'anomaly_score'] + FEATURE_COLUMNS + ['severity_score']
            # Only select columns that exist in the DataFrame
            available_columns = [col for col in output_columns if col in anomalies_df.columns]
            return anomalies_df[available_columns]
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing cell {cell_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 4 - OSP Anomaly Detection")
    parser.add_argument("data_path", type=Path, help="Path to preprocessed data file")
    parser.add_argument("reference_weeks_path", type=Path, help="Path to reference weeks file")
    parser.add_argument("--output_path", type=Path, default="outputs/04_individual_anomalies.parquet", help="Output file path")
    parser.add_argument("--n_components", type=int, default=3, help="SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0, help="Anomaly threshold")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes")
    parser.add_argument("--preview", action="store_true", help="Show a preview of the output DataFrame.")
    args = parser.parse_args()

    print("="*60); print("Stage 4: OSP Anomaly Detection"); print("="*60)
    start_time = time.perf_counter()

    data_lazy = pl.scan_parquet(args.data_path)
    ref_weeks_lazy = pl.scan_parquet(args.reference_weeks_path)
    
    # Convert reference_week from "2013_W47" format to integer format (like 201347)
    ref_weeks_processed = ref_weeks_lazy.with_columns(
        pl.col("reference_week")
        .str.replace("_W", "")
        .cast(pl.Int32)
        .alias("year_week")
    ).select(["cell_id", "year_week"])
    
    # Add year_week column to the main data
    data_with_year_week = data_lazy.with_columns(
        (pl.col("timestamp").dt.iso_year() * 100 + pl.col("timestamp").dt.week()).alias("year_week")
    )
    
    # Create a reference weeks set for marking
    ref_set = ref_weeks_processed.collect()
    ref_keys = set(zip(ref_set['cell_id'].to_list(), ref_set['year_week'].to_list()))
    
    # Perform the join and mark reference weeks
    data_with_weeks = data_with_year_week.with_columns(
        pl.struct(["cell_id", "year_week"]).map_elements(
            lambda x: (x["cell_id"], x["year_week"]) in ref_keys,
            return_dtype=pl.Boolean
        ).alias("is_reference")
    ).select([
        "cell_id", "timestamp", "sms_total", "calls_total", "internet_traffic", "is_reference"
    ]).collect()
    
    # Group by cell_id and convert to list of tuples for multiprocessing
    cells_grouped = data_with_weeks.group_by('cell_id')
    cell_tasks = [(cell_id, cell_df) for cell_id, cell_df in cells_grouped]
    
    num_workers = args.max_workers or min(len(cell_tasks), os.cpu_count())
    print(f"Processing {len(cell_tasks)} cells using {num_workers} workers...")
    
    results_dfs = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        for i, result in enumerate(pool.imap(process_single_cell, cell_tasks)):
            results_dfs.append(result)
            # Show progress every 1000 cells
            if (i + 1) % 1000 == 0 or (i + 1) == len(cell_tasks):
                print(f"Processed {i + 1:,}/{len(cell_tasks):,} cells ({(i + 1)/len(cell_tasks)*100:.1f}%)")
    
    all_anomalies_df = pd.concat(results_dfs, ignore_index=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    all_anomalies_df.to_parquet(args.output_path, index=False)
    
    total_time = time.perf_counter() - start_time
    
    if args.preview:
        print("\n--- DATA PREVIEW (Anomalies Found) ---")
        if not all_anomalies_df.empty:
            print(f"Shape: {all_anomalies_df.shape}")
            print("Head(5) of most severe anomalies:"); print(all_anomalies_df.nlargest(5, 'severity_score'))
            print("\nInfo:"); all_anomalies_df.info()
        else:
            print("No anomalies were detected.")

    print("\n--- STAGE 4 PERFORMANCE SUMMARY ---")
    print(f"Total samples analyzed: {len(data_with_weeks):,}")
    print(f"Individual anomalies detected: {len(all_anomalies_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("✅ Stage 4 completed successfully.")

if __name__ == "__main__":
    main()