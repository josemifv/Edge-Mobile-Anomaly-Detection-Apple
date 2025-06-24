#!/usr/bin/env python3
"""
robust_pipeline_runner.py

CMMSE 2025: Robust Pipeline Runner with Error Handling and Cleanup
================================================================

This script provides robust execution of the 5-stage anomaly detection pipeline with:
• Try/except wrapper for each run with error capture and status marking
• Temporary file cleanup based on --keep_tmp flag (removes bulky parquet artifacts)
• Monitor thread termination guarantee with finally blocks
• Rotating file handler for comprehensive logging
• Graceful error handling and recovery

Features:
- Wraps each pipeline run in comprehensive try/except blocks
- Captures detailed error logs and marks run status on failure
- Continues execution even if individual runs fail
- Cleans up temporary artifacts if --keep_tmp not specified
- Ensures monitor threads always terminate properly
- Uses rotating file handlers to prevent log file bloat
- Comprehensive error reporting and recovery mechanisms

Usage:
    python scripts/utils/robust_pipeline_runner.py data/raw/ [options]

Example:
    python scripts/utils/robust_pipeline_runner.py data/raw/ \
        --runs 10 --output_dir outputs/benchmarks/ \
        --n_components 3 --anomaly_threshold 2.0 \
        --keep_tmp --verbose
"""

import argparse
import json
import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
import psutil

# Import monitoring utilities with fallback
try:
    from .monitoring import start_monitor, stop_monitor
except ImportError:
    try:
        from monitoring import start_monitor, stop_monitor
    except ImportError:
        print("Warning: Could not import monitoring utilities. Resource monitoring will be limited.")
        start_monitor = None
        stop_monitor = None


class RobustPipelineRunner:
    """
    Robust pipeline runner with comprehensive error handling and cleanup.
    
    This class manages the execution of multiple pipeline runs with:
    - Individual run error handling and recovery
    - Temporary file cleanup management
    - Monitor thread lifecycle management
    - Rotating log file management
    - Comprehensive status tracking
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, logger: logging.Logger):
        """
        Initialize the robust pipeline runner.
        
        Args:
            input_dir: Directory containing raw data files
            output_dir: Base output directory for benchmark results
            logger: Configured logger instance
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = logger
        self.successful_runs = []
        self.failed_runs = []
        self.active_monitors = []  # Track active monitoring threads
        
        # Create benchmark session directory
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = output_dir / f"robust_benchmark_{self.session_timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized robust pipeline runner - Session: {self.session_timestamp}")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {self.session_dir}")
    
    def setup_run_directory(self, run_id: int) -> Tuple[Path, Path, Path]:
        """
        Set up directory structure for a single run.
        
        Args:
            run_id: Unique identifier for this run
            
        Returns:
            Tuple of (run_dir, data_dir, logs_dir)
        """
        run_dir = self.session_dir / f"run_{run_id}"
        data_dir = run_dir / "data"
        logs_dir = run_dir / "logs"
        
        # Create directories
        for directory in [run_dir, data_dir, logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        return run_dir, data_dir, logs_dir
    
    def execute_single_run(self, run_id: int, pipeline_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single pipeline run with comprehensive error handling.
        
        Args:
            run_id: Unique identifier for this run
            pipeline_args: Arguments to pass to the pipeline
            
        Returns:
            Dict containing run results and status information
        """
        run_dir, data_dir, logs_dir = self.setup_run_directory(run_id)
        
        # Initialize run metrics
        run_metrics = {
            'run_id': run_id,
            'status': 'unknown',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'execution_time_seconds': 0.0,
            'error_message': None,
            'error_traceback': None,
            'pipeline_stdout': None,
            'pipeline_stderr': None,
            'stage_timings': {},
            'file_sizes': {},
            'monitoring_data_available': False
        }
        
        monitor = None
        
        try:
            self.logger.info(f"Starting run {run_id}")
            start_time = time.perf_counter()
            
            # Start monitoring if available
            if start_monitor is not None:
                try:
                    monitor = start_monitor(interval=1.0, enable_apple_metrics=True)
                    self.active_monitors.append(monitor)
                    self.logger.debug(f"Started monitoring for run {run_id}")
                    run_metrics['monitoring_data_available'] = True
                except Exception as e:
                    self.logger.warning(f"Failed to start monitoring for run {run_id}: {e}")
                    monitor = None
            
            # Prepare pipeline command
            cmd = self._build_pipeline_command(data_dir, pipeline_args)
            self.logger.debug(f"Pipeline command: {' '.join(map(str, cmd))}")
            
            # Execute pipeline with comprehensive capture
            try:
                process_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                    timeout=pipeline_args.get('timeout', 3600)  # 1 hour default timeout
                )
                
                run_metrics['pipeline_stdout'] = process_result.stdout
                run_metrics['pipeline_stderr'] = process_result.stderr
                
                # Check if pipeline succeeded
                if process_result.returncode == 0:
                    run_metrics['status'] = 'success'
                    self.logger.info(f"Run {run_id} completed successfully")
                    
                    # Parse timing information from stdout
                    run_metrics['stage_timings'] = self._parse_pipeline_timing(process_result.stdout)
                    
                    # Collect file sizes
                    run_metrics['file_sizes'] = self._collect_file_sizes(data_dir)
                    
                    self.successful_runs.append(run_id)
                    
                else:
                    run_metrics['status'] = 'pipeline_failure'
                    run_metrics['error_message'] = f"Pipeline exited with code {process_result.returncode}"
                    self.logger.error(f"Run {run_id} failed - Pipeline exit code: {process_result.returncode}")
                    self.failed_runs.append(run_id)
                
            except subprocess.TimeoutExpired as e:
                run_metrics['status'] = 'timeout'
                run_metrics['error_message'] = f"Pipeline execution timeout after {e.timeout} seconds"
                self.logger.error(f"Run {run_id} timed out after {e.timeout} seconds")
                self.failed_runs.append(run_id)
                
            except Exception as e:
                run_metrics['status'] = 'execution_error'
                run_metrics['error_message'] = str(e)
                run_metrics['error_traceback'] = traceback.format_exc()
                self.logger.error(f"Run {run_id} failed with execution error: {e}")
                self.logger.debug(f"Run {run_id} traceback: {traceback.format_exc()}")
                self.failed_runs.append(run_id)
            
            # Calculate execution time
            end_time = time.perf_counter()
            run_metrics['execution_time_seconds'] = end_time - start_time
            run_metrics['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            # Catch-all for any unexpected errors
            run_metrics['status'] = 'unexpected_error'
            run_metrics['error_message'] = f"Unexpected error: {str(e)}"
            run_metrics['error_traceback'] = traceback.format_exc()
            run_metrics['end_time'] = datetime.now().isoformat()
            
            self.logger.error(f"Run {run_id} failed with unexpected error: {e}")
            self.logger.debug(f"Run {run_id} unexpected error traceback: {traceback.format_exc()}")
            self.failed_runs.append(run_id)
            
        finally:
            # Ensure monitor is always stopped and cleaned up
            if monitor is not None:
                try:
                    if stop_monitor is not None:
                        monitoring_df = stop_monitor(monitor)
                        
                        # Save monitoring data if available
                        if not monitoring_df.empty:
                            monitor_file = logs_dir / "resource_monitor.csv"
                            monitoring_df.to_csv(monitor_file, index=False)
                            self.logger.debug(f"Saved monitoring data for run {run_id}: {len(monitoring_df)} samples")
                    
                    # Remove from active monitors list
                    if monitor in self.active_monitors:
                        self.active_monitors.remove(monitor)
                        
                except Exception as e:
                    self.logger.warning(f"Error stopping monitor for run {run_id}: {e}")
            
            # Save run metrics
            try:
                metrics_file = run_dir / "run_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(run_metrics, f, indent=2, default=str)
                self.logger.debug(f"Saved run metrics for run {run_id}")
            except Exception as e:
                self.logger.error(f"Failed to save run metrics for run {run_id}: {e}")
            
            # Save pipeline logs
            try:
                if run_metrics.get('pipeline_stdout'):
                    stdout_file = logs_dir / "pipeline_stdout.log"
                    with open(stdout_file, 'w') as f:
                        f.write(run_metrics['pipeline_stdout'])
                
                if run_metrics.get('pipeline_stderr'):
                    stderr_file = logs_dir / "pipeline_stderr.log"
                    with open(stderr_file, 'w') as f:
                        f.write(run_metrics['pipeline_stderr'])
            except Exception as e:
                self.logger.warning(f"Failed to save pipeline logs for run {run_id}: {e}")
            
            # Cleanup temporary files if requested
            if not pipeline_args.get('keep_tmp', False):
                self._cleanup_temporary_files(data_dir, run_id)
        
        return run_metrics
    
    def _build_pipeline_command(self, data_dir: Path, pipeline_args: Dict[str, Any]) -> List[str]:
        """Build the pipeline command with appropriate arguments."""
        cmd = [
            sys.executable,
            "scripts/run_pipeline.py",
            str(self.input_dir),
            "--output_dir", str(data_dir),
            "--reports_dir", str(data_dir / "reports")
        ]
        
        # Add optional arguments
        if pipeline_args.get('n_components'):
            cmd.extend(["--n_components", str(pipeline_args['n_components'])])
        
        if pipeline_args.get('anomaly_threshold'):
            cmd.extend(["--anomaly_threshold", str(pipeline_args['anomaly_threshold'])])
        
        if pipeline_args.get('max_workers'):
            cmd.extend(["--max_workers", str(pipeline_args['max_workers'])])
        
        if pipeline_args.get('preview', False):
            cmd.append("--preview")
        
        return cmd
    
    def _parse_pipeline_timing(self, stdout_content: str) -> Dict[str, float]:
        """Parse stage timing information from pipeline stdout."""
        timing_dict = {}
        
        # Extract stage timings
        stage_pattern = re.compile(r'✅ Stage (\d+) completed successfully in ([\d.]+) seconds')
        for match in stage_pattern.finditer(stdout_content):
            stage_num = int(match.group(1))
            duration = float(match.group(2))
            timing_dict[f'stage{stage_num}_time'] = duration
        
        # Extract total pipeline time
        total_pattern = re.compile(r'Total Pipeline Time:\s+([\d.]+) seconds')
        total_match = total_pattern.search(stdout_content)
        if total_match:
            timing_dict['total_pipeline_time'] = float(total_match.group(1))
        
        return timing_dict
    
    def _collect_file_sizes(self, data_dir: Path) -> Dict[str, Dict[str, float]]:
        """Collect file size information for pipeline artifacts."""
        file_sizes = {}
        
        # Expected stage outputs
        stage_files = {
            'stage1': '01_ingested_data.parquet',
            'stage2': '02_preprocessed_data.parquet',
            'stage3': '03_reference_weeks.parquet',
            'stage4': '04_individual_anomalies.parquet'
        }
        
        for stage, filename in stage_files.items():
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    size_bytes = file_path.stat().st_size
                    file_sizes[stage] = {
                        'filename': filename,
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024 * 1024), 2),
                        'size_gb': round(size_bytes / (1024 * 1024 * 1024), 3)
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to get size for {filename}: {e}")
        
        # Collect reports directory size if it exists
        reports_dir = data_dir / "reports"
        if reports_dir.exists():
            try:
                total_size = sum(f.stat().st_size for f in reports_dir.rglob('*') if f.is_file())
                file_sizes['reports'] = {
                    'dirname': 'reports',
                    'size_bytes': total_size,
                    'size_mb': round(total_size / (1024 * 1024), 2),
                    'size_gb': round(total_size / (1024 * 1024 * 1024), 3)
                }
            except Exception as e:
                self.logger.warning(f"Failed to calculate reports directory size: {e}")
        
        return file_sizes
    
    def _cleanup_temporary_files(self, data_dir: Path, run_id: int):
        """
        Clean up bulky parquet artifacts while keeping metrics and logs.
        
        This removes large parquet files but preserves:
        - Small metadata files
        - Log files
        - Metrics JSON files
        - Reports directory (contains analysis results)
        """
        try:
            # Files to remove (bulky parquet artifacts)
            files_to_remove = [
                '01_ingested_data.parquet',
                '02_preprocessed_data.parquet',
                '04_individual_anomalies.parquet'  # Keep reference weeks (small)
            ]
            
            bytes_cleaned = 0
            files_removed = 0
            
            for filename in files_to_remove:
                file_path = data_dir / filename
                if file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        bytes_cleaned += file_size
                        files_removed += 1
                        self.logger.debug(f"Removed {filename} ({file_size / (1024*1024):.1f} MB) from run {run_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {filename} from run {run_id}: {e}")
            
            if files_removed > 0:
                mb_cleaned = bytes_cleaned / (1024 * 1024)
                self.logger.info(f"Cleaned up {files_removed} files ({mb_cleaned:.1f} MB) from run {run_id}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup for run {run_id}: {e}")
    
    def ensure_all_monitors_stopped(self):
        """Ensure all monitoring threads are properly terminated."""
        if not self.active_monitors:
            return
        
        self.logger.info(f"Ensuring {len(self.active_monitors)} monitoring threads are stopped...")
        
        for i, monitor in enumerate(self.active_monitors[:]):  # Copy list to avoid modification during iteration
            try:
                if stop_monitor is not None:
                    stop_monitor(monitor)
                    self.logger.debug(f"Stopped monitor {i}")
                self.active_monitors.remove(monitor)
            except Exception as e:
                self.logger.warning(f"Error stopping monitor {i}: {e}")
        
        self.logger.info("All monitoring threads terminated")
    
    def generate_summary_report(self, all_run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary report for all runs."""
        summary = {
            'session_info': {
                'timestamp': self.session_timestamp,
                'total_runs': len(all_run_metrics),
                'successful_runs': len(self.successful_runs),
                'failed_runs': len(self.failed_runs),
                'success_rate': len(self.successful_runs) / len(all_run_metrics) if all_run_metrics else 0
            },
            'execution_summary': {},
            'error_summary': {},
            'performance_summary': {}
        }
        
        if self.successful_runs:
            # Calculate execution time statistics for successful runs
            successful_metrics = [m for m in all_run_metrics if m['status'] == 'success']
            execution_times = [m['execution_time_seconds'] for m in successful_metrics]
            
            summary['execution_summary'] = {
                'mean_execution_time': float(pd.Series(execution_times).mean()),
                'median_execution_time': float(pd.Series(execution_times).median()),
                'std_execution_time': float(pd.Series(execution_times).std()),
                'min_execution_time': float(pd.Series(execution_times).min()),
                'max_execution_time': float(pd.Series(execution_times).max())
            }
            
            # Aggregate stage timings
            stage_timings = {}
            for metrics in successful_metrics:
                for stage, timing in metrics.get('stage_timings', {}).items():
                    if stage not in stage_timings:
                        stage_timings[stage] = []
                    stage_timings[stage].append(timing)
            
            for stage, timings in stage_timings.items():
                summary['performance_summary'][f'{stage}_mean'] = float(pd.Series(timings).mean())
                summary['performance_summary'][f'{stage}_std'] = float(pd.Series(timings).std())
        
        if self.failed_runs:
            # Analyze failure patterns
            failed_metrics = [m for m in all_run_metrics if m['status'] != 'success']
            error_types = {}
            
            for metrics in failed_metrics:
                error_type = metrics.get('status', 'unknown')
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            summary['error_summary'] = {
                'error_types': error_types,
                'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            }
        
        return summary


def setup_rotating_logger(log_file: Path, verbose: bool = False) -> logging.Logger:
    """
    Set up rotating file handler for comprehensive logging.
    
    This prevents log files from growing too large by automatically
    rotating them when they exceed a certain size.
    """
    logger = logging.getLogger('robust_pipeline_runner')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Rotating file handler (10MB max, keep 5 backup files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def main():
    """Main execution function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="CMMSE 2025: Robust Pipeline Benchmark Runner with Error Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with error handling
  python scripts/utils/robust_pipeline_runner.py data/raw/ --runs 5

  # Full benchmark with cleanup disabled
  python scripts/utils/robust_pipeline_runner.py data/raw/ \\
      --runs 10 --keep_tmp --verbose

  # Custom configuration with timeout
  python scripts/utils/robust_pipeline_runner.py data/raw/ \\
      --runs 3 --timeout 7200 --n_components 5 --anomaly_threshold 2.5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw .txt data files"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs to execute (default: 10)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Base directory for benchmark outputs (default: outputs/benchmarks)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for each pipeline run in seconds (default: 3600)"
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--n_components",
        type=int,
        default=3,
        help="OSP SVD components (default: 3)"
    )
    
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=2.0,
        help="OSP anomaly threshold (default: 2.0)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Maximum parallel processes for applicable stages"
    )
    
    # Cleanup and debugging options
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary parquet files (do not clean up bulky artifacts)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Enable data previews in pipeline stages"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up rotating logger
    log_file = args.output_dir / "robust_pipeline_execution.log"
    logger = setup_rotating_logger(log_file, args.verbose)
    
    logger.info("="*80)
    logger.info("CMMSE 2025: ROBUST PIPELINE BENCHMARK RUNNER - START")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of runs: {args.runs}")
    logger.info(f"Keep temporary files: {args.keep_tmp}")
    logger.info(f"Timeout per run: {args.timeout} seconds")
    
    # Initialize pipeline runner
    runner = RobustPipelineRunner(args.input_dir, args.output_dir, logger)
    
    # Prepare pipeline arguments
    pipeline_args = {
        'n_components': args.n_components,
        'anomaly_threshold': args.anomaly_threshold,
        'max_workers': args.max_workers,
        'preview': args.preview,
        'keep_tmp': args.keep_tmp,
        'timeout': args.timeout
    }
    
    # Execute benchmark runs
    all_run_metrics = []
    
    try:
        for run_id in range(1, args.runs + 1):
            logger.info(f"Executing run {run_id}/{args.runs}")
            
            try:
                run_metrics = runner.execute_single_run(run_id, pipeline_args)
                all_run_metrics.append(run_metrics)
                
                # Log run result
                if run_metrics['status'] == 'success':
                    logger.info(f"Run {run_id} completed successfully in {run_metrics['execution_time_seconds']:.2f} seconds")
                else:
                    logger.error(f"Run {run_id} failed with status: {run_metrics['status']}")
                    if run_metrics.get('error_message'):
                        logger.error(f"Error message: {run_metrics['error_message']}")
                
            except Exception as e:
                logger.error(f"Unexpected error executing run {run_id}: {e}")
                logger.debug(f"Run {run_id} unexpected error traceback: {traceback.format_exc()}")
                
                # Create failure record
                failure_metrics = {
                    'run_id': run_id,
                    'status': 'runner_error',
                    'error_message': str(e),
                    'error_traceback': traceback.format_exc(),
                    'execution_time_seconds': 0.0,
                    'start_time': datetime.now().isoformat(),
                    'end_time': datetime.now().isoformat()
                }
                all_run_metrics.append(failure_metrics)
                runner.failed_runs.append(run_id)
    
    finally:
        # Ensure all monitors are stopped
        logger.info("Ensuring all monitoring threads are terminated...")
        runner.ensure_all_monitors_stopped()
        
        # Generate summary report
        try:
            logger.info("Generating summary report...")
            summary = runner.generate_summary_report(all_run_metrics)
            
            # Save summary to file
            summary_file = runner.session_dir / "benchmark_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Save detailed metrics
            detailed_file = runner.session_dir / "all_run_metrics.json"
            with open(detailed_file, 'w') as f:
                json.dump(all_run_metrics, f, indent=2, default=str)
            
            logger.info(f"Summary saved to: {summary_file}")
            logger.info(f"Detailed metrics saved to: {detailed_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
    
    # Final status report
    logger.info("="*80)
    logger.info("ROBUST PIPELINE BENCHMARK RUNNER - COMPLETE")
    logger.info("="*80)
    logger.info(f"Total runs executed: {len(all_run_metrics)}")
    logger.info(f"Successful runs: {len(runner.successful_runs)}")
    logger.info(f"Failed runs: {len(runner.failed_runs)}")
    
    if runner.successful_runs:
        success_rate = len(runner.successful_runs) / len(all_run_metrics) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    if runner.failed_runs:
        logger.warning(f"Failed run IDs: {runner.failed_runs}")
    
    logger.info(f"Session directory: {runner.session_dir}")
    logger.info("Robust pipeline benchmark execution completed.")
    
    # Exit with appropriate code
    if not runner.successful_runs:
        sys.exit(1)  # All runs failed
    elif runner.failed_runs:
        sys.exit(2)  # Some runs failed
    else:
        sys.exit(0)  # All runs succeeded


if __name__ == "__main__":
    main()
