#!/usr/bin/env python3
"""
04_anomaly_detection_individual.py

CMMSE 2025: Refactored OSP Anomaly Detection Pipeline
Stage 4: Individual Anomaly Detection and Recording

This refactored version outputs individual anomaly records instead of cell-level aggregates.
Analysis will be handled by Stage 5 (analyze_anomalies.py).

Usage:
    python scripts/04_anomaly_detection_individual.py <data_path> <reference_weeks_path> [--output_path <path>]

Example:
    python scripts/04_anomaly_detection_individual.py data/processed/preprocessed_data.parquet data/processed/reference_weeks.parquet --output_path data/processed/individual_anomalies.parquet
"""

import pandas as pd
import numpy as np
import argparse
import time
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Feature columns for OSP analysis
FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']


class OSPAnomalyDetectorIndividual:
    """OSP Anomaly Detector that captures individual anomaly records."""
    
    def __init__(self, n_components: int = 3, anomaly_threshold: float = 2.0, standardize: bool = True):
        """
        Initialize OSP Anomaly Detector for individual record output.
        
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
        self.training_mean_error = None
        self.training_std_error = None
        
    def fit(self, X: np.ndarray) -> 'OSPAnomalyDetectorIndividual':
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
        
        # Store training statistics
        self.training_mean_error = np.mean(reconstruction_errors)
        self.training_std_error = np.std(reconstruction_errors)
        self.error_threshold = self.training_mean_error + self.anomaly_threshold * self.training_std_error
        
        return self
    
    def predict_individual_records(self, X: np.ndarray, timestamps: np.ndarray, 
                                  cell_ids: np.ndarray, original_indices: np.ndarray = None) -> List[Dict]:
        """Predict anomalies and return individual anomaly records."""
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
        
        # Calculate severity scores (how many standard deviations above threshold)
        severity_scores = (anomaly_scores - self.training_mean_error) / self.training_std_error
        
        # Create individual anomaly records
        individual_anomalies = []
        anomaly_indices = np.where(anomaly_labels == 1)[0]
        
        for idx in anomaly_indices:
            record = {
                'cell_id': int(cell_ids[idx]),
                'timestamp': timestamps[idx],
                'anomaly_score': float(anomaly_scores[idx]),
                'severity_score': float(severity_scores[idx]),
                'sms_total': float(X[idx, 0]),
                'calls_total': float(X[idx, 1]),
                'internet_traffic': float(X[idx, 2]),
                'error_threshold': float(self.error_threshold),
                'training_mean_error': float(self.training_mean_error),
                'training_std_error': float(self.training_std_error)
            }
            
            if original_indices is not None:
                record['original_index'] = int(original_indices[idx])
                
            individual_anomalies.append(record)
        
        return individual_anomalies


def process_single_cell_individual(args: tuple) -> List[Dict]:
    """Process anomaly detection for a single cell and return individual anomaly records."""
    cell_id, cell_data, reference_weeks_for_cell, detector_params = args
    
    try:
        # Check if we have reference weeks for this cell
        if len(reference_weeks_for_cell) == 0:
            print(f"Warning: No reference weeks for cell {cell_id}")
            return []
        
        # Get reference week identifiers
        ref_weeks = set(reference_weeks_for_cell['reference_week'].values)
        
        # Add temporal features to cell data
        cell_data = cell_data.copy().reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(cell_data['timestamp']):
            cell_data['timestamp'] = pd.to_datetime(cell_data['timestamp'])
        
        iso_calendar = cell_data['timestamp'].dt.isocalendar()
        cell_data['year_week'] = (iso_calendar.year.astype(str) + '_W' + 
                                 iso_calendar.week.astype(str).str.zfill(2))
        
        # Separate training (reference weeks) and test data
        training_data = cell_data[cell_data['year_week'].isin(ref_weeks)]
        test_data = cell_data.copy()  # Test on all data
        
        if len(training_data) == 0:
            print(f"Warning: No training data for cell {cell_id}")
            return []
        
        # Prepare feature matrices
        X_train = training_data[FEATURE_COLUMNS].values
        X_test = test_data[FEATURE_COLUMNS].values
        
        # Remove any rows with NaN values
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)
        
        X_train_clean = X_train[train_mask]
        X_test_clean = X_test[test_mask]
        test_data_clean = test_data[test_mask].reset_index(drop=True)
        
        if len(X_train_clean) == 0 or len(X_test_clean) == 0:
            print(f"Warning: Insufficient clean data for cell {cell_id}")
            return []
        
        # Initialize and train OSP detector
        detector = OSPAnomalyDetectorIndividual(**detector_params)
        detector.fit(X_train_clean)
        
        # Get individual anomaly records
        individual_anomalies = detector.predict_individual_records(
            X_test_clean,
            timestamps=test_data_clean['timestamp'].values,
            cell_ids=np.full(len(test_data_clean), cell_id),
            original_indices=test_data_clean.index.values
        )
        
        print(f"Cell {cell_id}: {len(individual_anomalies)} anomalies detected from {len(test_data_clean)} samples")
        
        return individual_anomalies
        
    except Exception as e:
        print(f"Error processing cell {cell_id}: {str(e)}")
        return []


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 4 - Individual OSP Anomaly Detection")
    parser.add_argument("data_path", help="Path to preprocessed data file")
    parser.add_argument("reference_weeks_path", help="Path to reference weeks file")
    parser.add_argument("--output_path", default="data/processed/individual_anomalies.parquet",
                       help="Output file path for individual anomalies")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Number of SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0,
                       help="Anomaly threshold (std deviations)")
    parser.add_argument("--max_workers", type=int,
                       help="Max parallel processes (default: auto)")
    parser.add_argument("--sample_cells", type=int,
                       help="Process only N cells for testing")
    parser.add_argument("--preview", action="store_true", help="Show results preview")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 4: Individual OSP Anomaly Detection")
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
        unique_cells = data['cell_id'].unique()
        
        # Sample cells if requested
        if args.sample_cells:
            unique_cells = unique_cells[:args.sample_cells]
            print(f"Sampling {len(unique_cells)} cells for testing")
        
        tasks = []
        for cell_id in unique_cells:
            cell_data = data[data['cell_id'] == cell_id]
            ref_weeks_for_cell = reference_weeks[reference_weeks['cell_id'] == cell_id]
            tasks.append((cell_id, cell_data, ref_weeks_for_cell, detector_params))
        
        # Determine number of workers
        max_workers = args.max_workers or min(len(tasks), multiprocessing.cpu_count())
        print(f"Processing {len(tasks)} cells using {max_workers} workers...")
        
        # Process cells in parallel
        with multiprocessing.Pool(max_workers) as pool:
            results = pool.map(process_single_cell_individual, tasks)
        
        # Flatten results and create DataFrame
        print("Collecting and formatting results...")
        all_anomalies = []
        for cell_anomalies in results:
            all_anomalies.extend(cell_anomalies)
        
        if len(all_anomalies) == 0:
            print("Warning: No anomalies detected across all cells")
            anomalies_df = pd.DataFrame(columns=[
                'cell_id', 'timestamp', 'anomaly_score', 'severity_score',
                'sms_total', 'calls_total', 'internet_traffic',
                'error_threshold', 'training_mean_error', 'training_std_error'
            ])
        else:
            anomalies_df = pd.DataFrame(all_anomalies)
        
        # Create output directory
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save individual anomalies
        print(f"Saving individual anomalies to {args.output_path}...")
        if output_path.suffix.lower() == '.parquet':
            anomalies_df.to_parquet(args.output_path, index=False)
        else:
            anomalies_df.to_csv(args.output_path, index=False)
        
        # Show preview if requested
        if args.preview and len(anomalies_df) > 0:
            print("\n" + "="*40)
            print("INDIVIDUAL ANOMALIES PREVIEW")
            print("="*40)
            anomalies_df.info()
            print("\nTop 5 most severe anomalies:")
            top_5 = anomalies_df.nlargest(5, 'severity_score')
            for i, (_, anomaly) in enumerate(top_5.iterrows(), 1):
                print(f"{i}. Cell {anomaly['cell_id']} | {pd.to_datetime(anomaly['timestamp']).strftime('%Y-%m-%d %H:%M')} | Severity: {anomaly['severity_score']:.2f}σ")
            
            print(f"\nSeverity score statistics:")
            print(anomalies_df['severity_score'].describe())
            print(f"\nCells with most anomalies:")
            print(anomalies_df['cell_id'].value_counts().head())
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        total_samples = len(data) if not args.sample_cells else len(data[data['cell_id'].isin(unique_cells)])
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        
        print("="*60)
        print("STAGE 4 SUMMARY")
        print("="*60)
        print(f"Cells processed: {len(tasks)}")
        print(f"Total samples analyzed: {total_samples:,}")
        print(f"Individual anomalies detected: {len(anomalies_df):,}")
        print(f"Overall anomaly rate: {(len(anomalies_df)/total_samples)*100:.2f}%")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {total_samples/total_time:,.0f} samples/second")
        print(f"Output file: {args.output_path}")
        print(f"Output size: {file_size_mb:.1f} MB")
        print("Stage 4 completed successfully!")
        
        if len(anomalies_df) > 0:
            print(f"\nNext step: Run Stage 5 analysis")
            print(f"python scripts/05_analyze_anomalies.py {args.output_path}")
        
    except Exception as e:
        print(f"Stage 4 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
