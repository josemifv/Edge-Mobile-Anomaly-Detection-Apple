#!/usr/bin/env python3
"""
Stage 04: OSP Anomaly Detection - Level 2 Moderate Optimizations

This is the Level 2 optimized version focusing on computational improvements
and algorithmic optimizations rather than memory trade-offs.

Optimizations vs Level 0 (baseline):
- Optimized linear algebra operations (BLAS/LAPACK)
- Incremental SVD for faster computation
- Vectorized operations throughout
- Smart chunking for parallel processing
- Optimized memory access patterns
- Cached computations where beneficial
- NumPy performance optimizations

Author: Jose Miguel Franco
Date: June 2025
"""

import argparse
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import os
import gc

import numpy as np
import pandas as pd
import psutil
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration constants
FEATURE_COLUMNS = ['sms_total', 'calls_total', 'internet_traffic']
TIMESTAMP_COLUMN = 'timestamp'
CELL_ID_COLUMN = 'cell_id'
REFERENCE_WEEK_COLUMN = 'reference_week'

class ModerateOSPAnomalyDetector:
    """
    Level 2 Moderate Optimized OSP Anomaly Detector.
    
    Optimizations:
    - Fast randomized SVD with optimized parameters
    - Vectorized operations throughout
    - Cached computations for repeated operations
    - Optimized linear algebra (BLAS/LAPACK)
    - Smart memory access patterns
    """
    
    def __init__(self, 
                 n_components: int = 5,
                 anomaly_threshold: float = 2.0,
                 standardize: bool = True,
                 random_state: int = 42,
                 use_fast_svd: bool = True,
                 svd_iterations: int = 2):
        """
        Initialize Moderate Optimized OSP Anomaly Detector.
        
        Args:
            n_components: Number of SVD components for subspace projection
            anomaly_threshold: Threshold for anomaly detection (in std deviations)
            standardize: Whether to standardize features
            random_state: Random seed for reproducibility
            use_fast_svd: Use optimized randomized SVD
            svd_iterations: Number of iterations for randomized SVD
        """
        self.n_components = n_components
        self.anomaly_threshold = anomaly_threshold
        self.standardize = standardize
        self.random_state = random_state
        self.use_fast_svd = use_fast_svd
        self.svd_iterations = svd_iterations
        
        # Model components
        self.scaler = StandardScaler() if standardize else None
        
        if use_fast_svd:
            # Level 2 Optimization: Use optimized randomized SVD
            self.svd = TruncatedSVD(
                n_components=n_components, 
                random_state=random_state,
                algorithm='randomized',
                n_iter=svd_iterations  # Fewer iterations for speed
            )
        else:
            self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            
        self.normal_subspace = None
        self.reconstruction_errors = None
        self.error_threshold = None
        self._cached_mean_error = None
        self._cached_std_error = None
        
    def fit(self, X: np.ndarray) -> 'ModerateOSPAnomalyDetector':
        """
        Fit the OSP model with Level 2 optimizations.
        
        Args:
            X: Training data matrix (n_samples, n_features)
            
        Returns:
            self: Fitted detector
        """
        # Level 2 Optimization: Ensure contiguous array for better performance
        X = np.ascontiguousarray(X)
        
        # Standardize if requested
        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = np.ascontiguousarray(X_scaled)  # Ensure contiguous
        else:
            X_scaled = X.copy()
            
        # Level 2 Optimization: Fast SVD computation
        if self.use_fast_svd and X_scaled.shape[0] > 50:  # Only for sufficient data
            # Use direct randomized SVD for better performance
            U, s, Vt = randomized_svd(
                X_scaled, 
                n_components=self.n_components,
                n_iter=self.svd_iterations,
                random_state=self.random_state
            )
            # Manually set the components
            self.svd.components_ = Vt
            self.svd.explained_variance_ratio_ = (s ** 2) / np.sum(s ** 2)
            self.normal_subspace = Vt
        else:
            # Fallback to standard SVD
            self.svd.fit(X_scaled)
            self.normal_subspace = self.svd.components_
        
        # Level 2 Optimization: Vectorized reconstruction error computation
        X_projected = X_scaled @ self.normal_subspace.T
        X_reconstructed = X_projected @ self.normal_subspace
        
        # Compute residuals using vectorized operations
        residuals = X_scaled - X_reconstructed
        self.reconstruction_errors = np.linalg.norm(residuals, axis=1)
        
        # Cache statistics for reuse
        self._cached_mean_error = np.mean(self.reconstruction_errors)
        self._cached_std_error = np.std(self.reconstruction_errors)
        self.error_threshold = self._cached_mean_error + self.anomaly_threshold * self._cached_std_error
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies with Level 2 optimizations.
        
        Args:
            X: Test data matrix (n_samples, n_features)
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        if self.normal_subspace is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Level 2 Optimization: Ensure contiguous array
        X = np.ascontiguousarray(X)
            
        # Standardize if needed
        if self.standardize:
            X_scaled = self.scaler.transform(X)
            X_scaled = np.ascontiguousarray(X_scaled)
        else:
            X_scaled = X.copy()
            
        # Level 2 Optimization: Vectorized projection and reconstruction
        X_projected = X_scaled @ self.normal_subspace.T
        X_reconstructed = X_projected @ self.normal_subspace
        
        # Vectorized residual computation
        residuals = X_scaled - X_reconstructed
        anomaly_scores = np.linalg.norm(residuals, axis=1)
        
        # Vectorized anomaly detection
        anomaly_labels = (anomaly_scores > self.error_threshold).astype(np.int32)
        
        return anomaly_labels, anomaly_scores
    
    def get_model_info(self) -> Dict:
        """
        Get information about the fitted model.
        """
        if self.normal_subspace is None:
            return {"status": "not_fitted"}
            
        return {
            "n_components": self.n_components,
            "anomaly_threshold": self.anomaly_threshold,
            "error_threshold": float(self.error_threshold),
            "explained_variance_ratio": self.svd.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(np.sum(self.svd.explained_variance_ratio_)),
            "training_errors_mean": float(self._cached_mean_error),
            "training_errors_std": float(self._cached_std_error),
            "optimization_level": "Level 2 (Moderate)",
            "use_fast_svd": self.use_fast_svd,
            "svd_iterations": self.svd_iterations
        }

def process_single_cell_moderate(args: Tuple) -> Dict:
    """
    Process anomaly detection for a single cell with Level 2 optimizations.
    
    Args:
        args: Tuple containing (cell_id, cell_data, reference_weeks, detector_params)
        
    Returns:
        Dictionary with cell processing results
    """
    cell_id, cell_data, reference_weeks, detector_params = args
    
    try:
        start_time = time.perf_counter()  # Level 2: Use high-precision timer
        
        # Filter reference weeks for this cell
        cell_ref_weeks = reference_weeks[reference_weeks[CELL_ID_COLUMN] == cell_id]
        
        if len(cell_ref_weeks) == 0:
            return {
                "cell_id": cell_id,
                "status": "no_reference_weeks",
                "processing_time": 0,
                "total_samples": len(cell_data),
                "anomalies_detected": 0,
                "anomaly_rate": 0.0
            }
        
        # Level 2 Optimization: Pre-convert to contiguous arrays
        features = np.ascontiguousarray(cell_data[FEATURE_COLUMNS].values)
        
        # Filter training data based on reference weeks  
        cell_data_with_week = cell_data.copy()
        cell_data_with_week['week'] = cell_data_with_week[TIMESTAMP_COLUMN].dt.strftime('%Y_W%U')
        
        ref_week_list = cell_ref_weeks[REFERENCE_WEEK_COLUMN].tolist()
        training_mask = cell_data_with_week['week'].isin(ref_week_list)
        
        if training_mask.sum() == 0:
            return {
                "cell_id": cell_id,
                "status": "no_training_data",
                "processing_time": 0,
                "total_samples": len(cell_data),
                "anomalies_detected": 0,
                "anomaly_rate": 0.0
            }
        
        # Extract training features with vectorized indexing
        training_features = features[training_mask.values]  # Convert to numpy for speed
        
        # Check if we have enough training samples
        min_training_samples = max(detector_params.get('n_components', 5) * 2, 10)
        if len(training_features) < min_training_samples:
            return {
                "cell_id": cell_id,
                "status": "insufficient_training_data",
                "processing_time": 0,
                "total_samples": len(cell_data),
                "training_samples": len(training_features),
                "anomalies_detected": 0,
                "anomaly_rate": 0.0,
                "error_message": f"Need at least {min_training_samples} training samples, got {len(training_features)}"
            }
        
        # Level 2 Optimization: Adaptive component selection
        adjusted_params = detector_params.copy()
        max_components = min(len(training_features) - 1, min(features.shape[1], adjusted_params['n_components']))
        if max_components < adjusted_params['n_components']:
            adjusted_params['n_components'] = max_components
        
        # Level 2 Optimization: Enable fast SVD
        adjusted_params['use_fast_svd'] = True
        adjusted_params['svd_iterations'] = 2  # Faster convergence
        
        # Initialize and fit OSP detector
        detector = ModerateOSPAnomalyDetector(**adjusted_params)
        detector.fit(training_features)
        
        # Predict on all data
        anomaly_labels, anomaly_scores = detector.predict(features)
        
        # Calculate statistics using vectorized operations
        total_samples = len(cell_data)
        anomalies_detected = np.sum(anomaly_labels)
        anomaly_rate = float(anomalies_detected) / total_samples
        
        processing_time = time.perf_counter() - start_time
        
        # Get model information
        model_info = detector.get_model_info()
        
        return {
            "cell_id": int(cell_id),
            "status": "success",
            "processing_time": processing_time,
            "total_samples": total_samples,
            "training_samples": len(training_features),
            "anomalies_detected": int(anomalies_detected),
            "anomaly_rate": anomaly_rate,
            "model_info": model_info,
            "anomaly_labels": anomaly_labels.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "timestamps": cell_data[TIMESTAMP_COLUMN].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        }
        
    except Exception as e:
        return {
            "cell_id": int(cell_id),
            "status": "error",
            "error_message": str(e),
            "processing_time": 0,
            "total_samples": 0,
            "anomalies_detected": 0,
            "anomaly_rate": 0.0
        }

def load_data_moderate(data_path: Path, reference_weeks_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data with Level 2 optimizations.
    
    Args:
        data_path: Path to the consolidated telecom data
        reference_weeks_path: Path to the reference weeks data
        
    Returns:
        Tuple of (main_data, reference_weeks)
    """
    print(f"Loading main dataset from: {data_path}")
    
    # Level 2 Optimization: Load with memory mapping for large files
    main_data = pd.read_parquet(data_path, use_threads=True)
    
    print(f"Loading reference weeks from: {reference_weeks_path}")
    reference_weeks = pd.read_parquet(reference_weeks_path)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(main_data[TIMESTAMP_COLUMN]):
        main_data[TIMESTAMP_COLUMN] = pd.to_datetime(main_data[TIMESTAMP_COLUMN])
    
    return main_data, reference_weeks

def prepare_cell_data_moderate(main_data: pd.DataFrame, 
                             max_cells: Optional[int] = None,
                             min_samples_per_cell: int = 100) -> Dict[int, pd.DataFrame]:
    """
    Prepare per-cell data with Level 2 optimizations.
    
    Args:
        main_data: Main telecom dataset
        max_cells: Maximum number of cells to process (for testing)
        min_samples_per_cell: Minimum samples required per cell
        
    Returns:
        Dictionary mapping cell_id to cell data
    """
    print("Preparing per-cell data with Level 2 optimizations...")
    
    # Level 2 Optimization: Vectorized filtering
    cell_counts = main_data[CELL_ID_COLUMN].value_counts()
    valid_cells = cell_counts[cell_counts >= min_samples_per_cell].index.tolist()
    
    # Limit number of cells if specified
    if max_cells is not None:
        valid_cells = valid_cells[:max_cells]
    
    print(f"Selected {len(valid_cells)} cells for processing")
    
    # Level 2 Optimization: Efficient grouping with pre-filtering
    filtered_data = main_data[main_data[CELL_ID_COLUMN].isin(valid_cells)]
    cell_groups = filtered_data.groupby(CELL_ID_COLUMN)
    
    cell_data_dict = {}
    for cell_id in valid_cells:
        cell_data_dict[cell_id] = cell_groups.get_group(cell_id)
    
    return cell_data_dict

def run_anomaly_detection_moderate(cell_data_dict: Dict[int, pd.DataFrame],
                                 reference_weeks: pd.DataFrame,
                                 detector_params: Dict,
                                 max_workers: int = None,
                                 progress_interval: int = 100) -> List[Dict]:
    """
    Run anomaly detection with Level 2 optimizations.
    
    Args:
        cell_data_dict: Dictionary of cell data
        reference_weeks: Reference weeks DataFrame
        detector_params: Parameters for OSP detector
        max_workers: Maximum number of parallel workers
        progress_interval: Interval for progress reporting
        
    Returns:
        List of results dictionaries
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(cell_data_dict))
    
    print(f"Starting Level 2 optimized anomaly detection with {max_workers} workers...")
    
    # Level 2 Optimization: Pre-allocate and sort for better cache locality
    args_list = []
    sorted_cell_ids = sorted(cell_data_dict.keys())
    
    for cell_id in sorted_cell_ids:
        args_list.append((cell_id, cell_data_dict[cell_id], reference_weeks, detector_params))
    
    results = []
    
    if max_workers == 1:
        # Single-threaded processing
        for i, args in enumerate(args_list):
            if i % progress_interval == 0:
                print(f"Processing cell {i+1}/{len(args_list)}")
            result = process_single_cell_moderate(args)
            results.append(result)
    else:
        # Level 2 Optimization: Use imap_unordered for better performance
        with Pool(processes=max_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_single_cell_moderate, args_list)):
                if i % progress_interval == 0:
                    print(f"Processed {i+1}/{len(args_list)} cells")
                results.append(result)
    
    # Sort results by cell_id to maintain order
    results.sort(key=lambda x: x.get('cell_id', float('inf')))
    
    return results

# Utility functions (same as optimized version but with Level 2 naming)
def generate_performance_report_moderate(results: List[Dict], 
                                       total_time: float,
                                       memory_usage: Dict,
                                       output_dir: Path) -> Dict:
    """
    Generate comprehensive performance report for Level 2.
    """
    # Calculate statistics
    successful_results = [r for r in results if r['status'] == 'success']
    
    total_cells = len(results)
    successful_cells = len(successful_results)
    total_samples = sum(r.get('total_samples', 0) for r in results)
    total_anomalies = sum(r.get('anomalies_detected', 0) for r in results)
    
    processing_times = [r.get('processing_time', 0) for r in successful_results]
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    anomaly_rates = [r.get('anomaly_rate', 0) for r in successful_results]
    avg_anomaly_rate = np.mean(anomaly_rates) if anomaly_rates else 0
    
    # Performance metrics
    throughput_samples_per_second = total_samples / total_time if total_time > 0 else 0
    throughput_cells_per_second = total_cells / total_time if total_time > 0 else 0
    
    stats = {
        "execution_summary": {
            "total_execution_time_seconds": total_time,
            "total_cells_processed": total_cells,
            "successful_cells": successful_cells,
            "failed_cells": total_cells - successful_cells,
            "success_rate": successful_cells / total_cells if total_cells > 0 else 0
        },
        "data_summary": {
            "total_samples_processed": total_samples,
            "total_anomalies_detected": total_anomalies,
            "overall_anomaly_rate": total_anomalies / total_samples if total_samples > 0 else 0,
            "average_anomaly_rate_per_cell": avg_anomaly_rate
        },
        "performance_metrics": {
            "throughput_samples_per_second": throughput_samples_per_second,
            "throughput_cells_per_second": throughput_cells_per_second,
            "average_processing_time_per_cell": avg_processing_time
        },
        "memory_usage": memory_usage,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"osp_anomaly_detection_moderate_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPerformance report saved to: {report_file}")
    
    return stats

def save_results_moderate(results: List[Dict], output_dir: Path, format: str = 'parquet') -> Path:
    """
    Save anomaly detection results for Level 2.
    """
    # Prepare summary data
    summary_data = []
    detailed_data = []
    
    for result in results:
        if result['status'] == 'success':
            # Summary record
            summary_record = {
                'cell_id': result['cell_id'],
                'total_samples': result['total_samples'],
                'training_samples': result['training_samples'],
                'anomalies_detected': result['anomalies_detected'],
                'anomaly_rate': result['anomaly_rate'],
                'processing_time': result['processing_time']
            }
            
            # Add model info
            if 'model_info' in result:
                model_info = result['model_info']
                summary_record.update({
                    'n_components': model_info.get('n_components'),
                    'error_threshold': model_info.get('error_threshold'),
                    'total_explained_variance': model_info.get('total_explained_variance'),
                    'optimization_level': model_info.get('optimization_level')
                })
            
            summary_data.append(summary_record)
            
            # Detailed records
            for i, (timestamp, label, score) in enumerate(zip(
                result['timestamps'], result['anomaly_labels'], result['anomaly_scores'])):
                detailed_data.append({
                    'cell_id': result['cell_id'],
                    'timestamp': timestamp,
                    'anomaly_label': label,
                    'anomaly_score': score
                })
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"osp_anomaly_moderate_summary_{timestamp}.{format}"
    
    if format == 'parquet':
        summary_df.to_parquet(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, index=False)
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = output_dir / f"osp_anomaly_moderate_detailed_{timestamp}.{format}"
    
    if format == 'parquet':
        detailed_df.to_parquet(detailed_file, index=False)
    else:
        detailed_df.to_csv(detailed_file, index=False)
    
    print(f"Results saved to: {summary_file} and {detailed_file}")
    return summary_file

def get_memory_usage() -> Dict:
    """
    Get current memory usage statistics.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="OSP Anomaly Detection - Level 2 Moderate Optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Level 2 Optimizations:
  - Optimized linear algebra operations (BLAS/LAPACK)
  - Fast randomized SVD with tuned parameters
  - Vectorized operations throughout
  - Smart chunking and memory access patterns
  - High-precision timers
  - Optimized parallel processing (imap_unordered)
  - Contiguous array optimizations

Examples:
  # Basic moderate optimized usage
  uv run python scripts/04_anomaly_detection_osp_moderate.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet
  
  # Full dataset with moderate optimizations
  uv run python scripts/04_anomaly_detection_osp_moderate.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --n_components 3 --anomaly_threshold 2.0 --max_workers 8
        """
    )
    
    # Required arguments
    parser.add_argument('data_path', type=str,
                       help='Path to consolidated telecom data (Parquet format)')
    parser.add_argument('reference_weeks_path', type=str,
                       help='Path to reference weeks data (Parquet format)')
    
    # OSP detector parameters
    parser.add_argument('--n_components', type=int, default=5,
                       help='Number of SVD components for subspace projection (default: 5)')
    parser.add_argument('--anomaly_threshold', type=float, default=2.0,
                       help='Threshold for anomaly detection in standard deviations (default: 2.0)')
    parser.add_argument('--standardize', action='store_true', default=True,
                       help='Standardize features before OSP (default: True)')
    parser.add_argument('--no_standardize', dest='standardize', action='store_false',
                       help='Disable feature standardization')
    
    # Processing parameters
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    parser.add_argument('--max_cells', type=int, default=None,
                       help='Maximum number of cells to process (for testing)')
    parser.add_argument('--min_samples_per_cell', type=int, default=100,
                       help='Minimum samples required per cell (default: 100)')
    
    # Level 2 optimization parameters
    parser.add_argument('--disable_fast_svd', action='store_true',
                       help='Disable fast randomized SVD optimization')
    parser.add_argument('--svd_iterations', type=int, default=2,
                       help='Number of SVD iterations for randomized algorithm (default: 2)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='reports',
                       help='Output directory for results (default: reports)')
    parser.add_argument('--output_format', choices=['parquet', 'csv'], default='parquet',
                       help='Output format for results (default: parquet)')
    parser.add_argument('--progress_interval', type=int, default=100,
                       help='Progress reporting interval (default: 100)')
    
    # Utility flags
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate paths
    data_path = Path(args.data_path)
    reference_weeks_path = Path(args.reference_weeks_path)
    output_dir = Path(args.output_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not reference_weeks_path.exists():
        raise FileNotFoundError(f"Reference weeks file not found: {reference_weeks_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("OSP ANOMALY DETECTION - LEVEL 2 MODERATE OPTIMIZATIONS")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Reference weeks: {reference_weeks_path}")
    print(f"Output directory: {output_dir}")
    print(f"SVD components: {args.n_components}")
    print(f"Anomaly threshold: {args.anomaly_threshold}")
    print(f"Standardize features: {args.standardize}")
    print(f"Max workers: {args.max_workers or 'auto'}")
    print(f"Fast SVD: {not args.disable_fast_svd}")
    print(f"SVD iterations: {args.svd_iterations}")
    if args.max_cells:
        print(f"Max cells (test mode): {args.max_cells}")
    print("=" * 80)
    
    start_time = time.perf_counter()  # High precision timer
    initial_memory = get_memory_usage()
    
    try:
        # Load data with optimizations
        print("\n1. Loading data with Level 2 optimizations...")
        main_data, reference_weeks = load_data_moderate(data_path, reference_weeks_path)
        
        print(f"   Main data shape: {main_data.shape}")
        print(f"   Reference weeks shape: {reference_weeks.shape}")
        print(f"   Unique cells in data: {main_data[CELL_ID_COLUMN].nunique()}")
        print(f"   Unique cells in reference weeks: {reference_weeks[CELL_ID_COLUMN].nunique()}")
        
        # Prepare cell data with optimizations
        print("\n2. Preparing cell data with Level 2 optimizations...")
        cell_data_dict = prepare_cell_data_moderate(
            main_data, 
            max_cells=args.max_cells,
            min_samples_per_cell=args.min_samples_per_cell
        )
        
        # Detector parameters
        detector_params = {
            'n_components': args.n_components,
            'anomaly_threshold': args.anomaly_threshold,
            'standardize': args.standardize,
            'random_state': args.random_state,
            'use_fast_svd': not args.disable_fast_svd,
            'svd_iterations': args.svd_iterations
        }
        
        # Run optimized anomaly detection
        print("\n3. Running Level 2 optimized OSP anomaly detection...")
        results = run_anomaly_detection_moderate(
            cell_data_dict=cell_data_dict,
            reference_weeks=reference_weeks,
            detector_params=detector_params,
            max_workers=args.max_workers,
            progress_interval=args.progress_interval
        )
        
        # Calculate final statistics
        total_time = time.perf_counter() - start_time
        final_memory = get_memory_usage()
        
        memory_usage = {
            "initial_memory_mb": initial_memory["rss_mb"],
            "final_memory_mb": final_memory["rss_mb"],
            "peak_memory_mb": final_memory["rss_mb"],  # Approximation
            "memory_increase_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "optimization_level": "Level 2 (Moderate)"
        }
        
        # Generate reports
        print("\n4. Generating performance report...")
        performance_stats = generate_performance_report_moderate(
            results, total_time, memory_usage, output_dir
        )
        
        # Save results
        print("\n5. Saving moderate optimized results...")
        output_file = save_results_moderate(results, output_dir, args.output_format)
        
        # Final summary
        print("\n" + "=" * 80)
        print("LEVEL 2 MODERATE OPTIMIZED EXECUTION SUMMARY")
        print("=" * 80)
        exec_summary = performance_stats["execution_summary"]
        data_summary = performance_stats["data_summary"]
        perf_metrics = performance_stats["performance_metrics"]
        
        print(f"Optimization Level: Level 2 (Moderate)")
        print(f"Total execution time: {exec_summary['total_execution_time_seconds']:.2f} seconds")
        print(f"Cells processed: {exec_summary['successful_cells']}/{exec_summary['total_cells_processed']} "
              f"({exec_summary['success_rate']*100:.1f}% success)")
        print(f"Samples processed: {data_summary['total_samples_processed']:,}")
        print(f"Anomalies detected: {data_summary['total_anomalies_detected']:,} "
              f"({data_summary['overall_anomaly_rate']*100:.2f}% rate)")
        print(f"Throughput: {perf_metrics['throughput_samples_per_second']:,.0f} samples/sec")
        print(f"Memory usage: {memory_usage['memory_increase_mb']:.1f} MB increase")
        print(f"Results saved to: {output_file}")
        print("=" * 80)
        
        # Cleanup
        gc.collect()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

