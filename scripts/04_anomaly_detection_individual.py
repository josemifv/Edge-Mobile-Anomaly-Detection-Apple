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
import os
import multiprocessing
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']

class OSPDetector:
    """OSP Anomaly Detector optimized with float32 precision and vectorized outputs."""
    
    def __init__(self, n_components: int = 3, anomaly_threshold: float = 2.0, standardize: bool = True):
        self.n_components = n_components
        self.anomaly_threshold = anomaly_threshold
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        # OPTIMIZATION: Use faster randomized SVD with fewer iterations
        self.svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm='randomized', n_iter=3)
        self.error_threshold = None
        self.training_mean_error = None
        self.training_std_error = None
        
    def fit(self, X: np.ndarray):
        # OPTIMIZATION: Use float32 for faster computation
        X_fit = X.astype(np.float32)
        X_scaled = self.scaler.fit_transform(X_fit) if self.standardize else X_fit
        self.svd.fit(X_scaled)
        
        X_reconstructed = self.svd.inverse_transform(self.svd.transform(X_scaled))
        residuals = X_scaled - X_reconstructed
        reconstruction_errors = np.linalg.norm(residuals, axis=1)
        
        self.training_mean_error = np.mean(reconstruction_errors)
        self.training_std_error = np.std(reconstruction_errors)
        self.error_threshold = self.training_mean_error + self.anomaly_threshold * self.training_std_error
        
    def predict(self, X: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
        if self.error_threshold is None:
            raise ValueError("Model must be fitted before prediction.")
        
        X_predict = X.astype(np.float32)
        X_scaled = self.scaler.transform(X_predict) if self.standardize else X_predict
        
        X_reconstructed = self.svd.inverse_transform(self.svd.transform(X_scaled))
        residuals = X_scaled - X_reconstructed
        anomaly_scores = np.linalg.norm(residuals, axis=1)
        
        # OPTIMIZATION: Vectorized creation of results, then filter. This avoids all Python loops.
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'anomaly_score': anomaly_scores
        })
        anomalies_df = results_df[results_df['anomaly_score'] > self.error_threshold].copy()

        if not anomalies_df.empty:
            anomalies_df['severity_score'] = (anomalies_df['anomaly_score'] - self.training_mean_error) / self.training_std_error
            # Add original features for context
            original_features_df = pd.DataFrame(X, columns=FEATURE_COLUMNS).loc[anomalies_df.index]
            anomalies_df = pd.concat([anomalies_df, original_features_df], axis=1)
        
        return anomalies_df

def process_single_cell(cell_df: pl.DataFrame, detector_params: dict) -> pd.DataFrame:
    """Processes a single cell's data to detect anomalies."""
    cell_id = cell_df['cell_id'][0]
    
    try:
        # Separate training and test data based on the 'is_reference' flag
        training_df = cell_df.filter(pl.col('is_reference'))
        if training_df.height == 0:
            return pd.DataFrame()

        # Convert to NumPy for scikit-learn
        X_train = training_df.select(FEATURE_COLUMNS).to_numpy()
        X_test = cell_df.select(FEATURE_COLUMNS).to_numpy()
        timestamps_test = cell_df['timestamp'].to_numpy()

        detector = OSPDetector(**detector_params)
        detector.fit(X_train)
        
        anomalies_df = detector.predict(X_test, timestamps_test)
        
        if not anomalies_df.empty:
            anomalies_df['cell_id'] = cell_id
            print(f"Cell {cell_id}: {len(anomalies_df)} anomalies detected.")
            # Reorder columns for final output
            return anomalies_df[['cell_id', 'timestamp', 'anomaly_score', 'severity_score'] + FEATURE_COLUMNS]
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error processing cell {cell_id}: {e}")
        return pd.DataFrame()

def main():
    """Main function to run the anomaly detection stage."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 4 - OSP Anomaly Detection")
    parser.add_argument("data_path", type=Path, help="Path to preprocessed data file")
    parser.add_argument("reference_weeks_path", type=Path, help="Path to reference weeks file")
    parser.add_argument("--output_path", type=Path, default="outputs/04_individual_anomalies.parquet", help="Output file path")
    parser.add_argument("--n_components", type=int, default=3, help="SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0, help="Anomaly threshold")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes")
    args = parser.parse_args()

    print("="*60); print("Stage 4: OSP Anomaly Detection"); print("="*60)
    start_time = time.perf_counter()

    # Use Polars for efficient data loading and preparation
    data_lazy = pl.scan_parquet(args.data_path)
    ref_weeks_lazy = pl.scan_parquet(args.reference_weeks_path)
    
    # Add temporal features and join reference weeks
    print("Adding temporal features and joining reference weeks...")
    data_with_temporal = data_lazy.with_columns(
        (pl.col("timestamp").dt.iso_year().alias("year")),
        (pl.col("timestamp").dt.week().alias("week"))
    ).with_columns(
        (pl.col("year").cast(pl.Utf8) + "_W" + pl.col("week").cast(pl.Utf8).str.zfill(2)).alias("year_week")
    ).collect()
    
    # Load reference weeks and create a set for faster lookup
    ref_weeks_df = ref_weeks_lazy.collect()
    ref_weeks_set = set()
    for row in ref_weeks_df.iter_rows():
        cell_id, ref_week, _ = row
        ref_weeks_set.add((cell_id, ref_week))
    
    print(f"Loaded {len(ref_weeks_set)} reference week combinations")
    
    # Mark reference weeks efficiently
    data_with_weeks = data_with_temporal.with_columns(
        pl.struct(["cell_id", "year_week"]).map_elements(
            lambda x: (x['cell_id'], x['year_week']) in ref_weeks_set,
            return_dtype=pl.Boolean
        ).alias("is_reference")
    )

    detector_params = {'n_components': args.n_components, 'anomaly_threshold': args.anomaly_threshold}
    
    # Group data by cell for parallel processing
    print("Grouping data by cell for parallel processing...")
    grouped_data = [(cell_id, group_df) for cell_id, group_df in data_with_weeks.group_by('cell_id')]
    
    num_workers = args.max_workers or min(len(grouped_data), os.cpu_count())
    print(f"Processing {len(grouped_data)} cells using {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # We pass a tuple of (DataFrame, params) to each worker
        results_dfs = pool.starmap(process_single_cell, [(group_df, detector_params) for _, group_df in grouped_data])
    
    print("Collecting and formatting results...")
    all_anomalies_df = pd.concat(results_dfs, ignore_index=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    all_anomalies_df.to_parquet(args.output_path, index=False)
    
    total_time = time.perf_counter() - start_time
    print("\n--- STAGE 4 PERFORMANCE SUMMARY ---")
    print(f"Total samples analyzed: {len(data_with_weeks):,}")
    print(f"Individual anomalies detected: {len(all_anomalies_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("✅ Stage 4 completed successfully.")

if __name__ == "__main__":
    main()
