#!/usr/bin/env python3
"""
enhanced_benchmark_runner.py

CMMSE 2025: Enhanced Pipeline Benchmark Runner with Throughput Metrics
=====================================================================

Extended version of pipeline_benchmark_runner.py that automatically computes
comprehensive throughput and compression metrics for each benchmark run:

• Rows/sec per stage: using known row counts divided by stage time
• Compression ratio: ingested bytes → preprocessed → individual anomalies  
• CPU efficiency: mean & peak CPU usage vs wall time
• Thermal headroom: max temperature reached
• Persists all metrics in per-run JSON format

This script combines the existing benchmark execution with automatic metrics
computation to provide a complete performance analysis workflow.

Usage:
    python scripts/utils/enhanced_benchmark_runner.py data/raw/ [options]
    
Example:
    python scripts/utils/enhanced_benchmark_runner.py data/raw/ \
        --runs 5 --output_dir outputs/benchmarks/ \
        --n_components 3 --anomaly_threshold 2.0 --verbose

Author: José Miguel Franco-Valiente
Created: December 2024
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import the existing benchmark runner functionality
try:
    from pipeline_benchmark_runner import (
        setup_logging, load_env_defaults, get_system_info,
        run_single_benchmark, generate_summary_report
    )
except ImportError:
    # Fallback for when running as standalone script
    sys.path.append(str(Path(__file__).parent))
    from pipeline_benchmark_runner import (
        setup_logging, load_env_defaults, get_system_info,
        run_single_benchmark, generate_summary_report
    )

# Import the metrics calculator
try:
    from compute_throughput_metrics import ThroughputMetricsCalculator
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from compute_throughput_metrics import ThroughputMetricsCalculator

# Configure logging
logger = logging.getLogger('enhanced_benchmark_runner')


class EnhancedBenchmarkRunner:
    """
    Enhanced benchmark runner that includes automatic throughput metrics computation.
    
    Extends the basic benchmark functionality to automatically compute:
    - Stage throughput (rows/second)
    - Compression ratios between stages
    - CPU efficiency metrics
    - Thermal performance analysis
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, **kwargs):
        """
        Initialize the enhanced benchmark runner.
        
        Args:
            input_dir: Input data directory
            output_dir: Output directory for benchmark results
            **kwargs: Additional benchmark parameters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.benchmark_params = kwargs
        
        # Create timestamped benchmark directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_dir = self.output_dir / timestamp
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.benchmark_dir / "enhanced_benchmark_execution.log"
        self.logger = setup_logging(log_file, kwargs.get('verbose', False))
        
        self.logger.info(f"Enhanced benchmark runner initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Benchmark directory: {self.benchmark_dir}")
        
    def run_benchmarks(self, num_runs: int) -> List[Dict]:
        """
        Execute multiple benchmark runs with metrics computation.
        
        Args:
            num_runs: Number of benchmark runs to execute
            
        Returns:
            List of run results with basic metrics
        """
        self.logger.info(f"Starting {num_runs} benchmark runs...")
        
        # Collect system information
        system_info = get_system_info()
        self.logger.info(f"System: {system_info['platform']} - {system_info['processor']}")
        self.logger.info(f"Apple Silicon: {system_info.get('apple_silicon', False)}")
        
        run_results = []
        
        for run_id in range(1, num_runs + 1):
            self.logger.info(f"=" * 60)
            self.logger.info(f"STARTING RUN {run_id}/{num_runs}")
            self.logger.info(f"=" * 60)
            
            # Create run directory
            run_output_dir = self.benchmark_dir / f"run_{run_id}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Execute single benchmark run
                run_metrics = run_single_benchmark(
                    run_id=run_id,
                    input_dir=self.input_dir,
                    run_output_dir=run_output_dir,
                    max_workers=self.benchmark_params.get('max_workers'),
                    n_components=self.benchmark_params.get('n_components', 3),
                    anomaly_threshold=self.benchmark_params.get('anomaly_threshold', 2.0),
                    keep_tmp=self.benchmark_params.get('keep_tmp', False),
                    logger=self.logger
                )
                
                run_results.append(run_metrics)
                
                # Log run completion
                if run_metrics['success']:
                    execution_time = run_metrics['execution_time_seconds']
                    self.logger.info(f"✅ Run {run_id} completed successfully in {execution_time:.2f}s")
                else:
                    self.logger.error(f"❌ Run {run_id} failed: {run_metrics.get('error_message', 'Unknown error')}")
                
            except Exception as e:
                error_msg = f"Critical error in run {run_id}: {str(e)}"
                self.logger.error(error_msg)
                
                # Create minimal error result
                run_results.append({
                    'run_id': run_id,
                    'success': False,
                    'error_message': error_msg,
                    'execution_time_seconds': 0
                })
            
            # Brief pause between runs
            if run_id < num_runs:
                time.sleep(2)
        
        self.logger.info(f"Completed all {num_runs} benchmark runs")
        
        # Generate basic summary report
        try:
            summary_dir = self.benchmark_dir / "summary"
            generate_summary_report(run_results, summary_dir, system_info, self.logger)
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
        
        return run_results
    
    def compute_throughput_metrics(self) -> Optional[Path]:
        """
        Compute comprehensive throughput and compression metrics for all runs.
        
        Returns:
            Path to consolidated metrics file or None if computation failed
        """
        self.logger.info("Computing comprehensive throughput and compression metrics...")
        
        try:
            # Initialize metrics calculator
            calculator = ThroughputMetricsCalculator(self.benchmark_dir)
            
            # Load run data
            self.logger.info("Loading run data for metrics computation...")
            calculator.load_run_data()
            
            if not calculator.runs_data:
                self.logger.error("No valid run data found for metrics computation!")
                return None
            
            # Compute all metrics
            self.logger.info(f"Computing metrics for {len(calculator.runs_data)} runs...")
            all_metrics = calculator.compute_all_metrics()
            
            # Save metrics to JSON files
            output_file = calculator.save_metrics_to_json(all_metrics)
            
            # Log results
            successful_runs = sum(1 for m in all_metrics if m.get('success', False))
            self.logger.info(f"✅ Throughput metrics computed successfully!")
            self.logger.info(f"Processed {len(all_metrics)} runs ({successful_runs} successful)")
            self.logger.info(f"Metrics saved to: {output_file.parent}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error computing throughput metrics: {e}")
            if self.benchmark_params.get('verbose', False):
                import traceback
                self.logger.error(traceback.format_exc())
            return None
    
    def run_complete_benchmark(self, num_runs: int) -> Dict[str, any]:
        """
        Run complete benchmark workflow including metrics computation.
        
        Args:
            num_runs: Number of benchmark runs to execute
            
        Returns:
            Dictionary with benchmark results and metrics information
        """
        start_time = time.perf_counter()
        
        self.logger.info("=" * 70)
        self.logger.info("ENHANCED PIPELINE BENCHMARK EXECUTION")
        self.logger.info("=" * 70)
        
        # Execute benchmark runs
        run_results = self.run_benchmarks(num_runs)
        
        # Compute throughput metrics
        metrics_file = self.compute_throughput_metrics()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Prepare final results
        final_results = {
            'benchmark_directory': str(self.benchmark_dir),
            'total_runs': len(run_results),
            'successful_runs': sum(1 for r in run_results if r.get('success', False)),
            'total_benchmark_time_seconds': round(total_time, 2),
            'total_benchmark_time_minutes': round(total_time / 60, 2),
            'metrics_computed': metrics_file is not None,
            'metrics_file': str(metrics_file) if metrics_file else None,
            'benchmark_parameters': self.benchmark_params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save final results
        results_file = self.benchmark_dir / "enhanced_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Log final summary
        self.logger.info("=" * 70)
        self.logger.info("ENHANCED BENCHMARK COMPLETED")
        self.logger.info("=" * 70)
        self.logger.info(f"Benchmark directory: {self.benchmark_dir}")
        self.logger.info(f"Total runs: {final_results['total_runs']}")
        self.logger.info(f"Successful runs: {final_results['successful_runs']}")
        self.logger.info(f"Total time: {final_results['total_benchmark_time_minutes']:.2f} minutes")
        self.logger.info(f"Metrics computed: {'✅ YES' if final_results['metrics_computed'] else '❌ NO'}")
        
        if final_results['metrics_computed']:
            self.logger.info(f"Comprehensive metrics available in: {metrics_file.parent}")
        
        return final_results


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Enhanced pipeline benchmark runner with automatic throughput metrics computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic enhanced benchmark with 10 runs
    python scripts/utils/enhanced_benchmark_runner.py data/raw/
    
    # Custom configuration with metrics
    python scripts/utils/enhanced_benchmark_runner.py data/raw/ \
        --runs 5 --output_dir outputs/benchmarks/ \
        --n_components 3 --anomaly_threshold 2.0 --verbose
    
    # Performance-focused benchmark
    python scripts/utils/enhanced_benchmark_runner.py data/raw/ \
        --runs 3 --max_workers 8 --keep_tmp --verbose
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Input directory containing raw data files'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='Number of benchmark runs to execute (default: 10)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs/benchmarks/'),
        help='Output directory for benchmark results (default: outputs/benchmarks/)'
    )
    
    parser.add_argument(
        '--max_workers',
        type=int,
        help='Maximum number of worker processes (default: auto-detect)'
    )
    
    parser.add_argument(
        '--n_components',
        type=int,
        default=3,
        help='Number of OSP components (default: 3)'
    )
    
    parser.add_argument(
        '--anomaly_threshold',
        type=float,
        default=2.0,
        help='Anomaly detection threshold (default: 2.0)'
    )
    
    parser.add_argument(
        '--keep_tmp',
        action='store_true',
        help='Keep temporary files after execution'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load environment defaults
    env_defaults = load_env_defaults()
    
    # Apply environment defaults if not specified
    if args.max_workers is None and 'MAX_WORKERS' in env_defaults:
        try:
            args.max_workers = int(env_defaults['MAX_WORKERS'])
        except ValueError:
            pass
    
    if 'N_COMPONENTS' in env_defaults:
        try:
            args.n_components = int(env_defaults['N_COMPONENTS'])
        except ValueError:
            pass
    
    if 'ANOMALY_THRESHOLD' in env_defaults:
        try:
            args.anomaly_threshold = float(env_defaults['ANOMALY_THRESHOLD'])
        except ValueError:
            pass
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    try:
        # Initialize enhanced benchmark runner
        runner = EnhancedBenchmarkRunner(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            n_components=args.n_components,
            anomaly_threshold=args.anomaly_threshold,
            keep_tmp=args.keep_tmp,
            verbose=args.verbose
        )
        
        # Run complete benchmark with metrics
        results = runner.run_complete_benchmark(args.runs)
        
        # Print final summary to stdout
        print("\n" + "=" * 70)
        print("ENHANCED BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Benchmark Directory: {results['benchmark_directory']}")
        print(f"Total Runs: {results['total_runs']}")
        print(f"Successful Runs: {results['successful_runs']}")
        print(f"Success Rate: {results['successful_runs']/results['total_runs']*100:.1f}%")
        print(f"Total Time: {results['total_benchmark_time_minutes']:.2f} minutes")
        print(f"Metrics Computed: {'✅ YES' if results['metrics_computed'] else '❌ NO'}")
        
        if results['metrics_computed']:
            print(f"Metrics File: {results['metrics_file']}")
        
        print("=" * 70)
        
        # Exit with appropriate code
        if results['successful_runs'] == 0:
            print("❌ No successful runs - exiting with error")
            sys.exit(1)
        elif results['successful_runs'] < results['total_runs']:
            print("⚠️  Some runs failed - check logs for details")
            sys.exit(0)
        else:
            print("✅ All runs completed successfully")
            sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
