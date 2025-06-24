#!/usr/bin/env python3
"""
pipeline_benchmark_runner.py

CMMSE 2025: Pipeline Benchmark Runner
=====================================

Benchmarks the complete 5-stage anomaly detection pipeline by running multiple iterations
and collecting comprehensive performance metrics and outputs.

This script:
• Wraps the existing run_pipeline.py with fixed parameters
• Executes multiple benchmark runs with different configurations
• Organizes outputs in timestamped directory hierarchy
• Aggregates performance metrics and generates summary reports

Output Hierarchy:
outputs/benchmarks/YYYYMMDD_HHMMSS/
├── run_1/                    # Individual run outputs
│   ├── logs/                 # Execution logs
│   ├── data/                 # Pipeline outputs (parquet files)
│   └── reports/              # Analysis reports
├── run_2/
│   └── ...
├── run_N/
│   └── ...
└── summary/                  # Aggregated results
    ├── benchmark_summary.json
    ├── performance_plot.png
    ├── timing_analysis.csv
    └── benchmark_report.md

Usage:
    python scripts/utils/pipeline_benchmark_runner.py data/raw/ [options]

Example:
    python scripts/utils/pipeline_benchmark_runner.py data/raw/ \\
        --runs 5 --output_dir outputs/benchmarks/ \\
        --n_components 3 --anomaly_threshold 2.0 --verbose
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import psutil

# Import monitoring utilities
try:
    from .monitoring import start_monitor, stop_monitor
except ImportError:
    try:
        from monitoring import start_monitor, stop_monitor
    except ImportError:
        print("Warning: Could not import monitoring utilities. Resource monitoring will be limited.")
        start_monitor = None
        stop_monitor = None


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Set up logging configuration for benchmark execution."""
    logger = logging.getLogger('benchmark_runner')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


def load_env_defaults(env_file: Path = Path('.env')) -> Dict[str, str]:
    """Load default configuration values from .env file if present."""
    defaults = {}
    
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        defaults[key.strip()] = value
        except Exception as e:
            print(f"Warning: Could not read .env file: {e}")
    
    return defaults


def get_system_info() -> Dict[str, str]:
    """Collect system information for benchmark context."""
    import platform
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'timestamp': datetime.now().isoformat()
    }
    
    # Apple Silicon detection
    if 'arm64' in platform.machine() or 'aarch64' in platform.machine():
        info['apple_silicon'] = True
        info['architecture'] = 'ARM64'
    else:
        info['apple_silicon'] = False
        info['architecture'] = platform.machine()
    
    return info


def parse_pipeline_timing(stdout_content: str) -> Dict[str, float]:
    """
    Parse per-stage timing information from pipeline stdout.
    
    The run_pipeline.py script prints timing information in the format:
    "✅ Stage X completed successfully in Y.ZZ seconds."
    
    Args:
        stdout_content: Complete stdout content from pipeline execution
        
    Returns:
        Dict with stage timing information
    """
    timing_dict = {}
    
    # Regex pattern to extract stage timing information
    # Matches: "✅ Stage 1 completed successfully in 123.45 seconds."
    stage_pattern = re.compile(r'✅ Stage (\d+) completed successfully in ([\d.]+) seconds')
    
    for match in stage_pattern.finditer(stdout_content):
        stage_num = int(match.group(1))
        duration = float(match.group(2))
        timing_dict[f'stage{stage_num}_time'] = duration
    
    # Also look for total pipeline time
    # Matches: "Total Pipeline Time:             123.45 seconds (2.06 minutes)"
    total_pattern = re.compile(r'Total Pipeline Time:\s+([\d.]+) seconds')
    total_match = total_pattern.search(stdout_content)
    
    if total_match:
        timing_dict['total_pipeline_time'] = float(total_match.group(1))
    
    return timing_dict


def collect_stage_artifacts_sizes(data_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Collect file sizes for each stage artifact.
    
    Args:
        data_dir: Directory containing pipeline output files
        
    Returns:
        Dict with file information for each stage
    """
    artifacts = {}
    
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
                artifacts[stage] = {
                    'filename': filename,
                    'size_bytes': size_bytes,
                    'size_mb': round(size_bytes / (1024**2), 2),
                    'size_gb': round(size_bytes / (1024**3), 3)
                }
            except Exception as e:
                artifacts[stage] = {
                    'filename': filename,
                    'error': str(e)
                }
        else:
            artifacts[stage] = {
                'filename': filename,
                'exists': False
            }
    
    # Also check for reports directory
    reports_dir = data_dir.parent / 'reports'
    if reports_dir.exists():
        report_files = {}
        for report_file in reports_dir.glob('*'):
            if report_file.is_file():
                try:
                    size_bytes = report_file.stat().st_size
                    report_files[report_file.name] = {
                        'size_bytes': size_bytes,
                        'size_mb': round(size_bytes / (1024**2), 2)
                    }
                except Exception:
                    pass
        
        if report_files:
            artifacts['reports'] = report_files
    
    return artifacts


def run_single_benchmark(
    run_id: int,
    input_dir: Path,
    run_output_dir: Path,
    max_workers: Optional[int],
    n_components: int,
    anomaly_threshold: float,
    keep_tmp: bool,
    logger: logging.Logger
) -> Dict:
    """Execute a single benchmark run and collect metrics."""
    
    logger.info(f"Starting benchmark run {run_id}")
    
    # Create run directory structure
    run_data_dir = run_output_dir / "data"
    run_reports_dir = run_output_dir / "reports"
    run_logs_dir = run_output_dir / "logs"
    
    for dir_path in [run_data_dir, run_reports_dir, run_logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare command arguments
    cmd = [
        sys.executable, 
        "scripts/run_pipeline.py",
        str(input_dir),
        "--output_dir", str(run_data_dir),
        "--reports_dir", str(run_reports_dir),
        "--n_components", str(n_components),
        "--anomaly_threshold", str(anomaly_threshold)
    ]
    
    if max_workers:
        cmd.extend(["--max_workers", str(max_workers)])
    
    # Set up execution environment
    env = os.environ.copy()
    
    # Record system state before execution
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**2)  # MB
    initial_cpu_percent = psutil.cpu_percent()
    
    # Execute pipeline
    start_time = time.perf_counter()
    start_timestamp = datetime.now()
    
    try:
        logger.debug(f"Executing: {' '.join(cmd)}")
        
        # Capture output for both logging and timing extraction
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        
        # Save complete output to log file
        log_file = run_logs_dir / "pipeline_execution.log"
        with open(log_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        
        success = True
        error_message = None
        stdout_content = result.stdout
        
    except subprocess.CalledProcessError as e:
        success = False
        error_message = f"Pipeline execution failed with exit code {e.returncode}"
        logger.error(error_message)
        stdout_content = getattr(e, 'stdout', '') or ''
        
    except Exception as e:
        success = False
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message)
        logger.debug(traceback.format_exc())
        stdout_content = ''
    
    end_time = time.perf_counter()
    end_timestamp = datetime.now()
    execution_time = end_time - start_time
    
    # Record system state after execution
    final_memory = process.memory_info().rss / (1024**2)  # MB
    final_cpu_percent = psutil.cpu_percent()
    
    # Parse per-stage timing from stdout
    stage_timings = {}
    if success and stdout_content:
        stage_timings = parse_pipeline_timing(stdout_content)
        logger.debug(f"Extracted stage timings: {stage_timings}")
    
    # Collect stage artifact file sizes
    stage_artifacts = {}
    if success:
        stage_artifacts = collect_stage_artifacts_sizes(run_data_dir)
        logger.debug(f"Collected artifact information for {len(stage_artifacts)} stages")
    
    # Prepare run metrics
    run_metrics = {
        'run_id': run_id,
        'success': success,
        'execution_time_seconds': round(execution_time, 2),
        'execution_time_minutes': round(execution_time / 60, 2),
        'start_timestamp': start_timestamp.isoformat(),
        'end_timestamp': end_timestamp.isoformat(),
        'parameters': {
            'n_components': n_components,
            'anomaly_threshold': anomaly_threshold,
            'max_workers': max_workers,
            'input_dir': str(input_dir)
        },
        'system_metrics': {
            'initial_memory_mb': round(initial_memory, 2),
            'final_memory_mb': round(final_memory, 2),
            'memory_delta_mb': round(final_memory - initial_memory, 2),
            'initial_cpu_percent': initial_cpu_percent,
            'final_cpu_percent': final_cpu_percent
        },
        'stage_timings': stage_timings,
        'stage_artifacts': stage_artifacts,
        'error_message': error_message
    }
    
    # Save run metrics
    metrics_file = run_output_dir / "run_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(run_metrics, f, indent=2)
    
    # Clean up temporary files if requested
    if not keep_tmp:
        # Remove intermediate parquet files but keep final results
        for temp_file in run_data_dir.glob("0[1-3]_*.parquet"):
            try:
                temp_file.unlink()
                logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove {temp_file}: {e}")
    
    logger.info(f"Completed run {run_id} - {'SUCCESS' if success else 'FAILED'} "
                f"in {execution_time:.2f}s")
    
    return run_metrics


def generate_summary_report(
    benchmark_results: List[Dict],
    summary_dir: Path,
    system_info: Dict,
    logger: logging.Logger
) -> None:
    """Generate comprehensive summary report and visualizations."""
    
    logger.info("Generating benchmark summary report")
    
    # Calculate summary statistics
    successful_runs = [r for r in benchmark_results if r['success']]
    failed_runs = [r for r in benchmark_results if not r['success']]
    
    if not successful_runs:
        logger.error("No successful runs - cannot generate meaningful summary")
        return
    
    execution_times = [r['execution_time_seconds'] for r in successful_runs]
    
    # Calculate stage timing statistics
    stage_timing_stats = {}
    stage_keys = set()
    for r in successful_runs:
        stage_keys.update(r.get('stage_timings', {}).keys())
    
    for stage_key in stage_keys:
        stage_times = [r.get('stage_timings', {}).get(stage_key, 0) for r in successful_runs if r.get('stage_timings', {}).get(stage_key, 0) > 0]
        if stage_times:
            stage_timing_stats[stage_key] = {
                'mean_seconds': round(sum(stage_times) / len(stage_times), 2),
                'min_seconds': round(min(stage_times), 2),
                'max_seconds': round(max(stage_times), 2),
                'std_seconds': round(pd.Series(stage_times).std(), 2) if len(stage_times) > 1 else 0
            }
    
    summary_stats = {
        'total_runs': len(benchmark_results),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'success_rate': len(successful_runs) / len(benchmark_results),
        'execution_time_stats': {
            'mean_seconds': round(sum(execution_times) / len(execution_times), 2),
            'min_seconds': round(min(execution_times), 2),
            'max_seconds': round(max(execution_times), 2),
            'std_seconds': round(pd.Series(execution_times).std(), 2) if len(execution_times) > 1 else 0
        },
        'stage_timing_stats': stage_timing_stats,
        'system_info': system_info,
        'benchmark_timestamp': datetime.now().isoformat()
    }
    
    # Save JSON summary
    summary_file = summary_dir / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'individual_runs': benchmark_results
        }, f, indent=2)
    
    # Create timing analysis CSV with stage timings
    timing_data = []
    for r in benchmark_results:
        row = {
            'run_id': r['run_id'],
            'success': r['success'],
            'execution_time_seconds': r['execution_time_seconds'],
            'execution_time_minutes': r['execution_time_minutes'],
            'memory_delta_mb': r['system_metrics']['memory_delta_mb'],
            'n_components': r['parameters']['n_components'],
            'anomaly_threshold': r['parameters']['anomaly_threshold'],
            'max_workers': r['parameters']['max_workers']
        }
        
        # Add stage timing information if available
        stage_timings = r.get('stage_timings', {})
        for stage_key, timing in stage_timings.items():
            row[stage_key] = timing
        
        timing_data.append(row)
    
    timing_df = pd.DataFrame(timing_data)
    
    timing_csv = summary_dir / "timing_analysis.csv"
    timing_df.to_csv(timing_csv, index=False)
    
    # Generate performance visualization
    if len(successful_runs) > 1:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Execution time plot
            run_indices = [r['run_id'] for r in successful_runs]
            exec_times = [r['execution_time_seconds'] for r in successful_runs]
            
            ax1.plot(run_indices, exec_times, 'b-o', linewidth=2, markersize=6)
            ax1.axhline(summary_stats['execution_time_stats']['mean_seconds'], 
                       color='r', linestyle='--', alpha=0.7, label='Mean')
            ax1.set_xlabel('Run ID')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title('Pipeline Execution Time by Run')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Memory usage plot
            memory_deltas = [r['system_metrics']['memory_delta_mb'] for r in successful_runs]
            ax2.bar(run_indices, memory_deltas, alpha=0.7, color='green')
            ax2.set_xlabel('Run ID')
            ax2.set_ylabel('Memory Delta (MB)')
            ax2.set_title('Memory Usage Change by Run')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = summary_dir / "performance_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance plot saved to: {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate performance plot: {e}")
    
    # Generate markdown report
    report_md = summary_dir / "benchmark_report.md"
    with open(report_md, 'w') as f:
        f.write("# Pipeline Benchmark Report\\n\\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## Summary Statistics\\n\\n")
        f.write(f"- **Total Runs:** {summary_stats['total_runs']}\\n")
        f.write(f"- **Successful Runs:** {summary_stats['successful_runs']}\\n")
        f.write(f"- **Failed Runs:** {summary_stats['failed_runs']}\\n")
        f.write(f"- **Success Rate:** {summary_stats['success_rate']:.1%}\\n\\n")
        
        if successful_runs:
            stats = summary_stats['execution_time_stats']
            f.write("## Execution Time Statistics\\n\\n")
            f.write(f"- **Mean:** {stats['mean_seconds']:.2f} seconds ({stats['mean_seconds']/60:.2f} minutes)\\n")
            f.write(f"- **Minimum:** {stats['min_seconds']:.2f} seconds\\n")
            f.write(f"- **Maximum:** {stats['max_seconds']:.2f} seconds\\n")
            f.write(f"- **Std Dev:** {stats['std_seconds']:.2f} seconds\\n\\n")
        
        f.write("## System Information\\n\\n")
        for key, value in system_info.items():
            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\\n")
        
        if failed_runs:
            f.write("\\n## Failed Runs\\n\\n")
            for run in failed_runs:
                f.write(f"- **Run {run['run_id']}:** {run['error_message']}\\n")
    
    logger.info(f"Benchmark report saved to: {report_md}")


def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="CMMSE 2025: Pipeline Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with 10 runs
  python scripts/utils/pipeline_benchmark_runner.py data/raw/
  
  # Custom configuration
  python scripts/utils/pipeline_benchmark_runner.py data/raw/ \\
      --runs 5 --output_dir outputs/benchmarks/ \\
      --n_components 3 --anomaly_threshold 2.0 --verbose
  
  # With worker limit and temporary file cleanup
  python scripts/utils/pipeline_benchmark_runner.py data/raw/ \\
      --runs 3 --max_workers 4 --keep_tmp --verbose
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw .txt data files"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs to execute (default: 10)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Base directory for benchmark outputs (default: outputs/benchmarks/)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Maximum number of worker processes for parallel stages"
    )
    
    parser.add_argument(
        "--n_components",
        type=int,
        help="OSP SVD components for anomaly detection (default: 3)"
    )
    
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        help="OSP anomaly detection threshold (default: 2.0)"
    )
    
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep temporary intermediate files after benchmark completion"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Load environment defaults
    env_defaults = load_env_defaults()
    
    # Apply defaults from .env file if not specified via CLI
    if not args.output_dir:
        args.output_dir = Path(env_defaults.get('BENCHMARK_OUTPUT_DIR', 'outputs/benchmarks'))
    
    if not args.n_components:
        args.n_components = int(env_defaults.get('OSP_N_COMPONENTS', '3'))
        
    if not args.anomaly_threshold:
        args.anomaly_threshold = float(env_defaults.get('OSP_ANOMALY_THRESHOLD', '2.0'))
        
    if not args.max_workers and 'MAX_WORKERS' in env_defaults:
        args.max_workers = int(env_defaults['MAX_WORKERS'])
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create timestamped benchmark directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = args.output_dir / timestamp
    summary_dir = benchmark_dir / "summary"
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = benchmark_dir / "benchmark_execution.log"
    logger = setup_logging(log_file, args.verbose)
    
    # Log benchmark configuration
    logger.info("="*60)
    logger.info("PIPELINE BENCHMARK RUNNER - START")
    logger.info("="*60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {benchmark_dir}")
    logger.info(f"Number of runs: {args.runs}")
    logger.info(f"OSP components: {args.n_components}")
    logger.info(f"Anomaly threshold: {args.anomaly_threshold}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Keep temporary files: {args.keep_tmp}")
    logger.info(f"Verbose logging: {args.verbose}")
    
    # Collect system information
    system_info = get_system_info()
    logger.info(f"System: {system_info['platform']} ({system_info['architecture']})")
    logger.info(f"CPU cores: {system_info['cpu_count']} physical, {system_info['cpu_count_logical']} logical")
    logger.info(f"Memory: {system_info['memory_total_gb']} GB")
    
    # Execute benchmark runs
    benchmark_results = []
    total_start_time = time.perf_counter()
    
    for run_id in range(1, args.runs + 1):
        run_output_dir = benchmark_dir / f"run_{run_id}"
        
        try:
            # Start monitoring if available
            monitor = None
            if start_monitor:
                monitor = start_monitor()

            # Run benchmark
            run_metrics = run_single_benchmark(
                run_id=run_id,
                input_dir=args.input_dir,
                run_output_dir=run_output_dir,
                max_workers=args.max_workers,
                n_components=args.n_components,
                anomaly_threshold=args.anomaly_threshold,
                keep_tmp=args.keep_tmp,
                logger=logger
            )
            benchmark_results.append(run_metrics)

            # Stop monitoring and save CSV if available
            if monitor:
                monitor_df = stop_monitor(monitor)
                monitor_csv_path = run_output_dir / "resource_monitor.csv"
                monitor_df.to_csv(monitor_csv_path, index=False)
                logger.info(f"Resource monitoring data saved to {monitor_csv_path}")
            
        except KeyboardInterrupt:
            logger.warning(f"Benchmark interrupted during run {run_id}")
            break
        except Exception as e:
            logger.error(f"Unexpected error during run {run_id}: {e}")
            logger.debug(traceback.format_exc())
            # Continue with remaining runs
            continue
    
    total_time = time.perf_counter() - total_start_time
    
    # Generate summary report
    if benchmark_results:
        generate_summary_report(benchmark_results, summary_dir, system_info, logger)
    
    # Log final summary
    successful_runs = len([r for r in benchmark_results if r['success']])
    logger.info("="*60)
    logger.info("BENCHMARK EXECUTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total benchmark time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Completed runs: {len(benchmark_results)}/{args.runs}")
    logger.info(f"Successful runs: {successful_runs}/{len(benchmark_results)}")
    logger.info(f"Results saved to: {benchmark_dir}")
    
    if successful_runs == 0:
        logger.error("No successful benchmark runs completed!")
        sys.exit(1)
    
    print(f"\\n✅ Benchmark completed successfully!")
    print(f"Results available in: {benchmark_dir}")
    print(f"Summary report: {summary_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
