#!/usr/bin/env python3
"""
Benchmarking Script for Week Selection Parameter Sweeps

This script performs comprehensive parameter sweep testing for the week selection stage
to find optimal configurations for anomaly detection.

Usage:
    python scripts/benchmark_week_selection.py <input_parquet> [options]

Author: Edge Mobile Anomaly Detection Project
Target: Apple Silicon optimization
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import itertools
from datetime import datetime
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Import functions from the week selection script
import sys
sys.path.append(str(Path(__file__).parent))

from importlib import import_module
week_selection = import_module('03_week_selection')


class WeekSelectionBenchmark:
    """
    Benchmarking class for week selection parameter optimization.
    """
    
    def __init__(self, input_path, output_dir="reports/benchmarks"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data once
        print("Loading and preparing data for benchmarking...")
        self.df = self._load_and_prepare_data()
        print(f"✓ Data prepared: {self.df.shape}")
        
        # Benchmark results storage
        self.benchmark_results = []
        
    def _load_and_prepare_data(self):
        """Load and prepare data with temporal features."""
        df = week_selection.load_processed_data(self.input_path)
        df = week_selection.add_temporal_features(df)
        return df
    
    def run_parameter_sweep(self, param_grid, sample_cells=None, parallel=True, max_workers=None):
        """
        Run parameter sweep testing across different configurations.
        
        Args:
            param_grid (dict): Parameter grid for testing
            sample_cells (int): Number of cells to sample for faster testing (None = all cells)
            parallel (bool): Whether to run tests in parallel
            max_workers (int): Maximum number of parallel workers
        """
        print("\nSTARTING PARAMETER SWEEP BENCHMARK")
        print("=" * 60)
        
        # Generate all parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        print(f"Testing {len(param_combinations)} parameter combinations:")
        for name, values in param_grid.items():
            print(f"  {name}: {values}")
        
        if sample_cells:
            print(f"\nUsing sample of {sample_cells} cells for faster testing")
        
        print(f"Parallel processing: {'Enabled' if parallel else 'Disabled'}")
        print()
        
        start_time = time.perf_counter()
        
        if parallel and len(param_combinations) > 1:
            self._run_parallel_sweep(param_combinations, param_names, sample_cells, max_workers)
        else:
            self._run_sequential_sweep(param_combinations, param_names, sample_cells)
        
        total_time = time.perf_counter() - start_time
        print(f"\n✓ Parameter sweep completed in {total_time:.2f}s")
        print(f"  Total configurations tested: {len(self.benchmark_results)}")
        
        return self.benchmark_results
    
    def _run_sequential_sweep(self, param_combinations, param_names, sample_cells):
        """Run parameter sweep sequentially."""
        for i, params in enumerate(param_combinations, 1):
            param_dict = dict(zip(param_names, params))
            print(f"Testing configuration {i}/{len(param_combinations)}: {param_dict}")
            
            result = self._test_single_configuration(param_dict, sample_cells)
            self.benchmark_results.append(result)
    
    def _run_parallel_sweep(self, param_combinations, param_names, sample_cells, max_workers):
        """Run parameter sweep in parallel."""
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, len(param_combinations))
        
        print(f"Using {max_workers} parallel workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {}
            for i, params in enumerate(param_combinations, 1):
                param_dict = dict(zip(param_names, params))
                future = executor.submit(self._test_single_configuration_parallel, 
                                       param_dict, sample_cells, self.input_path)
                future_to_params[future] = (i, param_dict)
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                i, param_dict = future_to_params[future]
                try:
                    result = future.result()
                    self.benchmark_results.append(result)
                    print(f"✓ Completed {i}/{len(param_combinations)}: {param_dict}")
                except Exception as e:
                    print(f"✗ Failed {i}/{len(param_combinations)}: {param_dict} - {str(e)}")
    
    @staticmethod
    def _test_single_configuration_parallel(param_dict, sample_cells, input_path):
        """Test a single configuration in parallel (static method for pickling)."""
        # Reload data in each process
        df = week_selection.load_processed_data(input_path)
        df = week_selection.add_temporal_features(df)
        
        benchmark = WeekSelectionBenchmark.__new__(WeekSelectionBenchmark)
        benchmark.df = df
        
        return benchmark._test_single_configuration(param_dict, sample_cells)
    
    def _test_single_configuration(self, param_dict, sample_cells):
        """Test a single parameter configuration."""
        start_time = time.perf_counter()
        
        try:
            # Sample cells if requested
            df_test = self.df.copy()
            if sample_cells and sample_cells < df_test['cell_id'].nunique():
                sample_cell_ids = np.random.choice(
                    df_test['cell_id'].unique(), 
                    size=sample_cells, 
                    replace=False
                )
                df_test = df_test[df_test['cell_id'].isin(sample_cell_ids)]
            
            # Run the pipeline with current parameters
            weekly_df = week_selection.compute_weekly_aggregations(df_test)
            mad_df = week_selection.compute_mad_per_cell_week(weekly_df)
            
            reference_weeks_df, cell_stats_df = week_selection.select_reference_weeks_per_cell(
                mad_df, weekly_df, 
                num_reference_weeks=param_dict['num_reference_weeks'],
                mad_threshold=param_dict['mad_threshold']
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(reference_weeks_df, cell_stats_df, weekly_df, mad_df)
            
            processing_time = time.perf_counter() - start_time
            
            result = {
                'parameters': param_dict,
                'processing_time': processing_time,
                'success': True,
                'error': None,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            result = {
                'parameters': param_dict,
                'processing_time': processing_time,
                'success': False,
                'error': str(e),
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
        
        return result
    
    def _calculate_metrics(self, reference_weeks_df, cell_stats_df, weekly_df, mad_df):
        """Calculate comprehensive metrics for the configuration."""
        total_cells = len(cell_stats_df)
        
        metrics = {
            # Basic statistics
            'total_cells': total_cells,
            'total_reference_weeks': len(reference_weeks_df),
            'avg_reference_weeks_per_cell': reference_weeks_df.groupby('cell_id').size().mean(),
            
            # Week selection quality
            'cells_with_target_weeks': (cell_stats_df['reference_weeks_selected'] == 
                                      cell_stats_df['reference_weeks_selected'].iloc[0]).sum(),
            'cells_with_target_weeks_pct': (cell_stats_df['reference_weeks_selected'] == 
                                          cell_stats_df['reference_weeks_selected'].iloc[0]).mean() * 100,
            
            # Stability metrics
            'avg_stability_score': reference_weeks_df['stability_score'].mean(),
            'stability_score_std': reference_weeks_df['stability_score'].std(),
            'avg_normalized_deviation': reference_weeks_df['avg_norm_dev'].mean(),
            'normalized_deviation_std': reference_weeks_df['avg_norm_dev'].std(),
            
            # Week distribution metrics
            'unique_weeks_selected': reference_weeks_df['reference_week'].nunique(),
            'week_distribution_entropy': self._calculate_entropy(reference_weeks_df['reference_week']),
            'most_popular_week_pct': reference_weeks_df['reference_week'].value_counts().iloc[0] / total_cells * 100,
            
            # MAD analysis metrics
            'avg_mad_value': mad_df['mad_value'].mean(),
            'mad_value_std': mad_df['mad_value'].std(),
            'avg_normal_weeks_per_cell': cell_stats_df['normal_weeks_found'].mean(),
            'normal_weeks_coverage': cell_stats_df['normal_weeks_found'].sum() / (total_cells * weekly_df['year_week'].nunique()),
            
            # Data coverage metrics
            'weeks_available_min': cell_stats_df['total_weeks_available'].min(),
            'weeks_available_max': cell_stats_df['total_weeks_available'].max(),
            'weeks_available_avg': cell_stats_df['total_weeks_available'].mean(),
        }
        
        return metrics
    
    def _calculate_entropy(self, series):
        """Calculate Shannon entropy of a series."""
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-8))
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"week_selection_benchmark_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        benchmark_summary = {
            'benchmark_info': {
                'input_file': str(self.input_path),
                'total_configurations': len(self.benchmark_results),
                'successful_configurations': sum(1 for r in self.benchmark_results if r['success']),
                'failed_configurations': sum(1 for r in self.benchmark_results if not r['success']),
                'benchmark_timestamp': datetime.now().isoformat()
            },
            'results': self.benchmark_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(benchmark_summary, f, indent=2, default=str)
        
        print(f"✓ Benchmark results saved: {filepath}")
        return filepath
    
    def generate_analysis_report(self, filename=None):
        """Generate a comprehensive analysis report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"week_selection_analysis_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        successful_results = [r for r in self.benchmark_results if r['success']]
        
        with open(filepath, 'w') as f:
            f.write("WEEK SELECTION PARAMETER SWEEP ANALYSIS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BENCHMARK SUMMARY:\n")
            f.write(f"  Total configurations tested: {len(self.benchmark_results)}\n")
            f.write(f"  Successful configurations: {len(successful_results)}\n")
            f.write(f"  Failed configurations: {len(self.benchmark_results) - len(successful_results)}\n\n")
            
            if successful_results:
                f.write("PERFORMANCE ANALYSIS:\n")
                processing_times = [r['processing_time'] for r in successful_results]
                f.write(f"  Average processing time: {np.mean(processing_times):.2f}s\n")
                f.write(f"  Min processing time: {np.min(processing_times):.2f}s\n")
                f.write(f"  Max processing time: {np.max(processing_times):.2f}s\n\n")
                
                f.write("TOP 5 CONFIGURATIONS BY STABILITY SCORE:\n")
                sorted_by_stability = sorted(successful_results, 
                                           key=lambda x: x['metrics'].get('avg_stability_score', 0), 
                                           reverse=True)
                for i, result in enumerate(sorted_by_stability[:5], 1):
                    f.write(f"  {i}. {result['parameters']}\n")
                    f.write(f"     Stability Score: {result['metrics']['avg_stability_score']:.4f}\n")
                    f.write(f"     Processing Time: {result['processing_time']:.2f}s\n")
                    f.write(f"     Normal Week Coverage: {result['metrics']['normal_weeks_coverage']:.3f}\n\n")
                
                f.write("TOP 5 CONFIGURATIONS BY PROCESSING TIME:\n")
                sorted_by_time = sorted(successful_results, key=lambda x: x['processing_time'])
                for i, result in enumerate(sorted_by_time[:5], 1):
                    f.write(f"  {i}. {result['parameters']}\n")
                    f.write(f"     Processing Time: {result['processing_time']:.2f}s\n")
                    f.write(f"     Stability Score: {result['metrics']['avg_stability_score']:.4f}\n")
                    f.write(f"     Normal Week Coverage: {result['metrics']['normal_weeks_coverage']:.3f}\n\n")
                
                # Parameter impact analysis
                f.write("PARAMETER IMPACT ANALYSIS:\n")
                param_names = list(successful_results[0]['parameters'].keys())
                for param in param_names:
                    f.write(f"\n  {param.upper()} IMPACT:\n")
                    param_groups = {}
                    for result in successful_results:
                        param_val = result['parameters'][param]
                        if param_val not in param_groups:
                            param_groups[param_val] = []
                        param_groups[param_val].append(result)
                    
                    for param_val, group in param_groups.items():
                        avg_stability = np.mean([r['metrics']['avg_stability_score'] for r in group])
                        avg_time = np.mean([r['processing_time'] for r in group])
                        f.write(f"    {param_val}: Avg Stability={avg_stability:.4f}, Avg Time={avg_time:.2f}s\n")
        
        print(f"✓ Analysis report saved: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Week Selection Parameter Sweeps for Optimal Configuration"
    )
    parser.add_argument(
        "input_path",
        help="Path to the processed Parquet file"
    )
    parser.add_argument(
        "--num_reference_weeks",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="List of reference week counts to test (default: [2,3,4,5,6])"
    )
    parser.add_argument(
        "--mad_threshold", 
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0, 2.5],
        help="List of MAD thresholds to test (default: [1.0,1.5,2.0,2.5])"
    )
    parser.add_argument(
        "--sample_cells",
        type=int,
        help="Number of cells to sample for faster testing (default: all cells)"
    )
    parser.add_argument(
        "--output_dir",
        default="reports/benchmarks",
        help="Output directory for benchmark results (default: reports/benchmarks)"
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Maximum number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    np.random.seed(args.seed)
    
    # Validate input
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Define parameter grid
    param_grid = {
        'num_reference_weeks': args.num_reference_weeks,
        'mad_threshold': args.mad_threshold
    }
    
    print("WEEK SELECTION PARAMETER SWEEP BENCHMARK")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample cells: {args.sample_cells or 'All cells'}")
    print(f"Random seed: {args.seed}")
    print(f"Parallel processing: {'Disabled' if args.no_parallel else 'Enabled'}")
    
    # Initialize benchmark
    benchmark = WeekSelectionBenchmark(input_path, args.output_dir)
    
    # Run parameter sweep
    results = benchmark.run_parameter_sweep(
        param_grid=param_grid,
        sample_cells=args.sample_cells,
        parallel=not args.no_parallel,
        max_workers=args.max_workers
    )
    
    # Save results and generate reports
    json_file = benchmark.save_results()
    report_file = benchmark.generate_analysis_report()
    
    print(f"\n✓ Benchmark completed successfully!")
    print(f"  Results saved: {json_file}")
    print(f"  Analysis report: {report_file}")
    print(f"  Total configurations: {len(results)}")
    print(f"  Successful configurations: {sum(1 for r in results if r['success'])}")


if __name__ == "__main__":
    main()

