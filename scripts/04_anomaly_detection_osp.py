#!/usr/bin/env python3
"""
04_anomaly_detection_osp.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 4: OSP (Orthogonal Subspace Projection) Anomaly Detection

Implements OSP-based anomaly detection using SVD decomposition for per-cell modeling.
Uses reference weeks from Stage 3 to build normal behavior models and detect anomalies.

Usage:
    python scripts/04_anomaly_detection_osp.py <data_path> <reference_weeks_path> [--output_path <path>]

Example:
    python scripts/04_anomaly_detection_osp.py data/processed/preprocessed_data.parquet data/processed/reference_weeks.parquet --output_path results/anomalies.parquet
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Tuple
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')

# Feature columns for OSP analysis
FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']


class OSPAnomalyDetector:
    """Orthogonal Subspace Projection (OSP) Anomaly Detector for telecommunications data."""
    
    def __init__(self, n_components: int = 3, anomaly_threshold: float = 2.0, standardize: bool = True):
        """
        Initialize OSP Anomaly Detector.
        
        Args:
            n_components: Number of SVD components for subspace projection
            anomaly_threshold: Threshold for anomaly detection (in std deviations)
            standardize: Whether to standardize features
        """
        self.n_components = n_components
        self.anomaly_threshold = anomaly_threshold
        self.standardize = standardize
        
        # Model components
        self.scaler = StandardScaler() if standardize else None
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.error_threshold = None
        
    def fit(self, X: np.ndarray) -> 'OSPAnomalyDetector':
        """Fit the OSP model on normal (reference) data."""
        # Standardize if requested
        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
            
        # Fit SVD to get normal subspace
        self.svd.fit(X_scaled)
        
        # Calculate reconstruction errors on training data
        X_projected = self.svd.transform(X_scaled)
        X_reconstructed = self.svd.inverse_transform(X_projected)
        
        # Compute residuals (projection onto orthogonal subspace)
        residuals = X_scaled - X_reconstructed
        reconstruction_errors = np.linalg.norm(residuals, axis=1)
        
        # Set threshold based on training data statistics
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.error_threshold = mean_error + self.anomaly_threshold * std_error
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies in new data."""
        if self.error_threshold is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Standardize if needed
        if self.standardize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
            
        # Project onto normal subspace and reconstruct
        X_projected = self.svd.transform(X_scaled)
        X_reconstructed = self.svd.inverse_transform(X_projected)
        
        # Compute residuals (anomaly scores)
        residuals = X_scaled - X_reconstructed
        anomaly_scores = np.linalg.norm(residuals, axis=1)
        
        # Binary anomaly labels
        anomaly_labels = (anomaly_scores > self.error_threshold).astype(int)
        
        return anomaly_labels, anomaly_scores


def process_single_cell(args: tuple) -> dict:
    """Process anomaly detection for a single cell."""
    cell_id, cell_data, reference_weeks_for_cell, detector_params = args
    
    try:
        start_time = time.perf_counter()
        
        # Check if we have reference weeks for this cell
        if len(reference_weeks_for_cell) == 0:
            return {
                'cell_id': cell_id,
                'status': 'no_reference_weeks',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0
            }
        
        # Get reference week identifiers
        ref_weeks = set(reference_weeks_for_cell['reference_week'].values)
        
        # Add temporal features to cell data
        cell_data = cell_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(cell_data['timestamp']):
            cell_data['timestamp'] = pd.to_datetime(cell_data['timestamp'])
        
        iso_calendar = cell_data['timestamp'].dt.isocalendar()
        cell_data['year_week'] = (iso_calendar.year.astype(str) + '_W' + 
                                 iso_calendar.week.astype(str).str.zfill(2))
        
        # Separate training (reference weeks) and test data
        training_data = cell_data[cell_data['year_week'].isin(ref_weeks)]
        test_data = cell_data.copy()  # Test on all data
        
        if len(training_data) == 0:
            return {
                'cell_id': cell_id,
                'status': 'no_training_data',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0
            }
        
        # Prepare feature matrices
        X_train = training_data[FEATURE_COLUMNS].values
        X_test = test_data[FEATURE_COLUMNS].values
        
        # Remove any rows with NaN values
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)
        
        X_train_clean = X_train[train_mask]
        X_test_clean = X_test[test_mask]
        
        if len(X_train_clean) == 0 or len(X_test_clean) == 0:
            return {
                'cell_id': cell_id,
                'status': 'insufficient_clean_data',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0
            }
        
        # Initialize and train OSP detector
        detector = OSPAnomalyDetector(**detector_params)
        detector.fit(X_train_clean)
        
        # Predict anomalies
        anomaly_labels, anomaly_scores = detector.predict(X_test_clean)
        
        # Create results
        anomalies_detected = np.sum(anomaly_labels)
        anomaly_rate = anomalies_detected / len(anomaly_labels)
        
        processing_time = time.perf_counter() - start_time
        
        return {
            'cell_id': cell_id,
            'status': 'success',
            'total_samples': len(test_data),
            'training_samples': len(X_train_clean),
            'anomalies_detected': int(anomalies_detected),
            'anomaly_rate': float(anomaly_rate),
            'processing_time': processing_time,
            'error_threshold': detector.error_threshold
        }
        
    except Exception as e:
        return {
            'cell_id': cell_id,
            'status': f'error: {str(e)}',
            'total_samples': len(cell_data) if 'cell_data' in locals() else 0,
            'anomalies_detected': 0,
            'anomaly_rate': 0.0,
            'processing_time': 0.0
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 4 - OSP Anomaly Detection")
    parser.add_argument("data_path", help="Path to preprocessed data file")
    parser.add_argument("reference_weeks_path", help="Path to reference weeks file")
    parser.add_argument("--output_path", default="results/anomalies.parquet",
                       help="Output file path")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Number of SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0,
                       help="Anomaly threshold (std deviations)")
    parser.add_argument("--max_workers", type=int,
                       help="Max parallel processes (default: auto)")
    parser.add_argument("--preview", action="store_true", help="Show results preview")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 4: OSP Anomaly Detection")
    print("="*60)
    
    start_time = time.perf_counter()
    
    try:
        # Load data
        print("Loading data...")
        
        if args.data_path.endswith('.parquet'):
            data = pd.read_parquet(args.data_path)
        else:
            data = pd.read_csv(args.data_path)
        
        if args.reference_weeks_path.endswith('.parquet'):
            reference_weeks = pd.read_parquet(args.reference_weeks_path)
        else:
            reference_weeks = pd.read_csv(args.reference_weeks_path)
        
        print(f"  Data: {len(data):,} rows, {data['cell_id'].nunique()} cells")
        print(f"  Reference weeks: {len(reference_weeks):,} weeks for {reference_weeks['cell_id'].nunique()} cells")
        
        # Prepare detector parameters
        detector_params = {
            'n_components': args.n_components,
            'anomaly_threshold': args.anomaly_threshold,
            'standardize': True
        }
        
        print(f"OSP Parameters: {detector_params}")
        
        # Prepare tasks for parallel processing
        print("Preparing tasks for parallel processing...")
        tasks = []
        
        for cell_id in data['cell_id'].unique():
            cell_data = data[data['cell_id'] == cell_id]
            ref_weeks_for_cell = reference_weeks[reference_weeks['cell_id'] == cell_id]
            tasks.append((cell_id, cell_data, ref_weeks_for_cell, detector_params))
        
        # Determine number of workers
        max_workers = args.max_workers or min(len(tasks), multiprocessing.cpu_count())
        print(f"Processing {len(tasks)} cells using {max_workers} workers...")
        
        # Process cells in parallel
        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(process_single_cell, tasks)
        
        # Compile results
        results_df = pd.DataFrame(results)
        
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        print(f"Saving results to {args.output_path}...")
        if output_path.suffix.lower() == '.parquet':
            results_df.to_parquet(args.output_path, index=False)
        else:
            results_df.to_csv(args.output_path, index=False)
        
        # Show preview if requested
        if args.preview:
            print("\n" + "="*40)
            print("RESULTS PREVIEW")
            print("="*40)
            results_df.info()
            print("\nStatus distribution:")
            print(results_df['status'].value_counts())
            print("\nAnomaly rate statistics:")
            successful_results = results_df[results_df['status'] == 'success']
            if len(successful_results) > 0:
                print(successful_results['anomaly_rate'].describe())
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        successful_cells = len(results_df[results_df['status'] == 'success'])
        total_anomalies = results_df['anomalies_detected'].sum()
        total_samples = results_df['total_samples'].sum()
        
        print("="*60)
        print("STAGE 4 SUMMARY")
        print("="*60)
        print(f"Cells processed: {len(results_df)}")
        print(f"Successful cells: {successful_cells}")
        print(f"Total samples analyzed: {total_samples:,}")
        print(f"Total anomalies detected: {total_anomalies:,}")
        print(f"Overall anomaly rate: {(total_anomalies/total_samples)*100:.2f}%")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {total_samples/total_time:,.0f} samples/second")
        print(f"Output file: {args.output_path}")
        print("Stage 4 completed successfully!")
        
    except Exception as e:
        print(f"Stage 4 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
