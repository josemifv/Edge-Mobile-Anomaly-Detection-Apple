#!/usr/bin/env python3
"""
04_anomaly_detection_osp_detailed.py

CMMSE 2025: Enhanced OSP Anomaly Detection with Individual Anomaly Tracking
Stage 4: OSP (Orthogonal Subspace Projection) Anomaly Detection with Detailed Results

This enhanced version captures individual anomaly scores and details for severity analysis.

Usage:
    python scripts/04_anomaly_detection_osp_detailed.py <data_path> <reference_weeks_path> [--output_path <path>]

Example:
    python scripts/04_anomaly_detection_osp_detailed.py data/processed/preprocessed_data.parquet data/processed/reference_weeks.parquet --output_path results/detailed_anomalies.parquet
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Feature columns for OSP analysis
FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']


class OSPAnomalyDetectorDetailed:
    """Enhanced OSP Anomaly Detector that captures individual anomaly scores."""
    
    def __init__(self, n_components: int = 3, anomaly_threshold: float = 2.0, standardize: bool = True):
        """
        Initialize Enhanced OSP Anomaly Detector.
        
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
        
    def fit(self, X: np.ndarray) -> 'OSPAnomalyDetectorDetailed':
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
    
    def predict_detailed(self, X: np.ndarray, timestamps: np.ndarray = None, 
                        original_indices: np.ndarray = None) -> Dict:
        """Predict anomalies with detailed individual scores."""
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
        
        # Compile detailed results
        results = {
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'severity_scores': severity_scores,
            'timestamps': timestamps,
            'original_indices': original_indices,
            'original_features': X,
            'error_threshold': self.error_threshold,
            'training_mean_error': self.training_mean_error,
            'training_std_error': self.training_std_error
        }
        
        return results


def process_single_cell_detailed(args: tuple) -> Dict:
    """Process anomaly detection for a single cell with detailed results."""
    cell_id, cell_data, reference_weeks_for_cell, detector_params, return_details = args
    
    try:
        start_time = time.time()
        
        # Check if we have reference weeks for this cell
        if len(reference_weeks_for_cell) == 0:
            return {
                'cell_id': cell_id,
                'status': 'no_reference_weeks',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0,
                'individual_anomalies': []
            }
        
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
            return {
                'cell_id': cell_id,
                'status': 'no_training_data',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0,
                'individual_anomalies': []
            }
        
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
            return {
                'cell_id': cell_id,
                'status': 'insufficient_clean_data',
                'total_samples': len(cell_data),
                'anomalies_detected': 0,
                'anomaly_rate': 0.0,
                'processing_time': 0.0,
                'individual_anomalies': []
            }
        
        # Initialize and train OSP detector
        detector = OSPAnomalyDetectorDetailed(**detector_params)
        detector.fit(X_train_clean)
        
        # Predict anomalies with detailed results
        detailed_results = detector.predict_detailed(
            X_test_clean, 
            timestamps=test_data_clean['timestamp'].values,
            original_indices=test_data_clean.index.values
        )
        
        # Extract anomaly information
        anomaly_labels = detailed_results['anomaly_labels']
        anomaly_scores = detailed_results['anomaly_scores']
        severity_scores = detailed_results['severity_scores']
        
        # Create individual anomaly records
        individual_anomalies = []
        if return_details:
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            for idx in anomaly_indices:
                individual_anomalies.append({
                    'cell_id': cell_id,
                    'timestamp': detailed_results['timestamps'][idx],
                    'anomaly_score': float(anomaly_scores[idx]),
                    'severity_score': float(severity_scores[idx]),
                    'sms_total': float(detailed_results['original_features'][idx, 0]),
                    'calls_total': float(detailed_results['original_features'][idx, 1]),
                    'internet_traffic': float(detailed_results['original_features'][idx, 2]),
                    'error_threshold': float(detailed_results['error_threshold']),
                    'original_index': int(detailed_results['original_indices'][idx])
                })
        
        # Create summary results
        anomalies_detected = np.sum(anomaly_labels)
        anomaly_rate = anomalies_detected / len(anomaly_labels)
        processing_time = time.time() - start_time
        
        return {
            'cell_id': cell_id,
            'status': 'success',
            'total_samples': len(test_data),
            'training_samples': len(X_train_clean),
            'anomalies_detected': int(anomalies_detected),
            'anomaly_rate': float(anomaly_rate),
            'processing_time': processing_time,
            'error_threshold': detector.error_threshold,
            'training_mean_error': detector.training_mean_error,
            'training_std_error': detector.training_std_error,
            'max_severity_score': float(np.max(severity_scores[anomaly_labels == 1])) if anomalies_detected > 0 else 0.0,
            'individual_anomalies': individual_anomalies
        }
        
    except Exception as e:
        return {
            'cell_id': cell_id,
            'status': f'error: {str(e)}',
            'total_samples': len(cell_data) if 'cell_data' in locals() else 0,
            'anomalies_detected': 0,
            'anomaly_rate': 0.0,
            'processing_time': 0.0,
            'individual_anomalies': []
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Enhanced OSP Anomaly Detection with Individual Tracking")
    parser.add_argument("data_path", help="Path to preprocessed data file")
    parser.add_argument("reference_weeks_path", help="Path to reference weeks file")
    parser.add_argument("--output_path", default="results/detailed_anomalies.parquet",
                       help="Output file path for summary results")
    parser.add_argument("--individual_output_path", default="results/individual_anomalies.parquet",
                       help="Output file path for individual anomalies")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Number of SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0,
                       help="Anomaly threshold (std deviations)")
    parser.add_argument("--max_workers", type=int,
                       help="Max parallel processes (default: auto)")
    parser.add_argument("--preview", action="store_true", help="Show results preview")
    parser.add_argument("--top_anomalies", type=int, default=10,
                       help="Number of top anomalies to display")
    parser.add_argument("--sample_cells", type=int, 
                       help="Process only a sample of cells for testing")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Enhanced Mobile Network Anomaly Detection")
    print("Stage 4: OSP Anomaly Detection with Individual Tracking")
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
        
        # Sample cells if requested
        unique_cells = data['cell_id'].unique()
        if args.sample_cells and args.sample_cells < len(unique_cells):
            sampled_cells = np.random.choice(unique_cells, size=args.sample_cells, replace=False)
            data = data[data['cell_id'].isin(sampled_cells)]
            reference_weeks = reference_weeks[reference_weeks['cell_id'].isin(sampled_cells)]
            print(f"  Sampling {args.sample_cells} cells for processing")
        
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
            tasks.append((cell_id, cell_data, ref_weeks_for_cell, detector_params, True))  # True = return details
        
        # Determine number of workers
        max_workers = args.max_workers or min(len(tasks), multiprocessing.cpu_count())
        print(f"Processing {len(tasks)} cells using {max_workers} workers...")
        
        # Process cells in parallel
        with multiprocessing.Pool(max_workers) as pool:
            results = pool.map(process_single_cell_detailed, tasks)
        
        # Compile results
        summary_results = []
        individual_anomalies = []
        
        for result in results:
            # Summary results
            summary_result = {k: v for k, v in result.items() if k != 'individual_anomalies'}
            summary_results.append(summary_result)
            
            # Individual anomalies
            individual_anomalies.extend(result['individual_anomalies'])
        
        summary_df = pd.DataFrame(summary_results)
        individual_df = pd.DataFrame(individual_anomalies)
        
        # Create output directories
        output_path = Path(args.output_path)
        individual_output_path = Path(args.individual_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        individual_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        print(f"Saving summary results to {args.output_path}...")
        if output_path.suffix.lower() == '.parquet':
            summary_df.to_parquet(args.output_path, index=False)
        else:
            summary_df.to_csv(args.output_path, index=False)
        
        print(f"Saving individual anomalies to {args.individual_output_path}...")
        if individual_output_path.suffix.lower() == '.parquet':
            individual_df.to_parquet(args.individual_output_path, index=False)
        else:
            individual_df.to_csv(args.individual_output_path, index=False)
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        successful_cells = len(summary_df[summary_df['status'] == 'success'])
        total_anomalies = summary_df['anomalies_detected'].sum()
        total_samples = summary_df['total_samples'].sum()
        
        print("="*60)
        print("ENHANCED STAGE 4 SUMMARY")
        print("="*60)
        print(f"Cells processed: {len(summary_df)}")
        print(f"Successful cells: {successful_cells}")
        print(f"Total samples analyzed: {total_samples:,}")
        print(f"Total anomalies detected: {total_anomalies:,}")
        print(f"Individual anomaly records: {len(individual_df):,}")
        print(f"Overall anomaly rate: {(total_anomalies/total_samples)*100:.2f}%")
        print(f"Processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {total_samples/total_time:,.0f} samples/second")
        
        # Show top anomalies
        if len(individual_df) > 0:
            print("\n" + "="*60)
            print(f"TOP {args.top_anomalies} MOST SEVERE ANOMALIES")
            print("="*60)
            
            top_anomalies = individual_df.nlargest(args.top_anomalies, 'severity_score')
            
            for i, (_, anomaly) in enumerate(top_anomalies.iterrows(), 1):
                print(f"\n{i:2d}. Cell {anomaly['cell_id']:4d} | {pd.to_datetime(anomaly['timestamp']).strftime('%Y-%m-%d %H:%M')} | Severity: {anomaly['severity_score']:6.2f}σ")
                print(f"    Score: {anomaly['anomaly_score']:.6f} | Threshold: {anomaly['error_threshold']:.6f}")
                print(f"    SMS: {anomaly['sms_total']:8.1f} | Calls: {anomaly['calls_total']:8.1f} | Internet: {anomaly['internet_traffic']:8.1f}")
        
        # Show preview if requested
        if args.preview and len(individual_df) > 0:
            print("\n" + "="*40)
            print("INDIVIDUAL ANOMALIES PREVIEW")
            print("="*40)
            print(individual_df.info())
            print("\nSeverity score statistics:")
            print(individual_df['severity_score'].describe())
            print("\nCells with most anomalies:")
            print(individual_df['cell_id'].value_counts().head())
        
        print(f"\nOutput files:")
        print(f"  Summary: {args.output_path}")
        print(f"  Individual anomalies: {args.individual_output_path}")
        print("Enhanced Stage 4 completed successfully!")
        
    except Exception as e:
        print(f"Enhanced Stage 4 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
