#!/usr/bin/env python3
"""
aggregate_statistics.py

CMMSE 2025: Aggregate Statistics Across Runs (Step 5)
=====================================================

After benchmark loop finishes, this script loads list of run dicts into pandas DataFrame
and computes comprehensive aggregate statistics:

• Mean, median, std, min, max for every numeric metric
• Derives coefficient of variation (std/mean) for stability analysis  
• Calculates 95% confidence intervals using t-distribution
• Stores results to summary_stats.csv & summary_stats.json

Key Metrics Analyzed:
- Timings: stage1_time, stage2_time, stage3_time, stage4_time, stage5_time, total_pipeline_time
- CPU: cpu_percent metrics (mean, peak, per-core)
- Memory: memory_rss, memory_vms, memory_delta_mb, peak_memory
- Temperature: max_temperature_celsius, thermal headroom
- Throughput: rows_per_second per stage, compression ratios
- Compression: compression_ratio, space_saved_percent

Usage:
    python scripts/utils/aggregate_statistics.py <benchmark_results_dir>
    
Example:
    python scripts/utils/aggregate_statistics.py outputs/benchmarks/20241215_143022/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('aggregate_stats')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_benchmark_results(benchmark_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Load benchmark results from multiple sources in benchmark directory.
    
    Args:
        benchmark_dir: Directory containing benchmark results
        logger: Logger instance
        
    Returns:
        List of run dictionaries with all metrics
    """
    results = []
    
    # Try to load from summary JSON first (contains all runs)
    summary_json = benchmark_dir / "summary" / "benchmark_summary.json"
    if summary_json.exists():
        logger.info(f"Loading benchmark results from: {summary_json}")
        try:
            with open(summary_json, 'r') as f:
                data = json.load(f)
                if 'individual_runs' in data:
                    results = data['individual_runs']
                    logger.info(f"Loaded {len(results)} runs from summary JSON")
                    return results
        except Exception as e:
            logger.warning(f"Could not load from summary JSON: {e}")
    
    # Fallback: Load individual run metrics
    logger.info("Loading individual run metrics...")
    run_dirs = sorted([d for d in benchmark_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    for run_dir in run_dirs:
        try:
            # Load basic run metrics
            run_metrics_file = run_dir / "run_metrics.json"
            if run_metrics_file.exists():
                with open(run_metrics_file, 'r') as f:
                    run_data = json.load(f)
                    
                # Try to load throughput metrics if available
                throughput_files = list((benchmark_dir / "summary").glob(f"{run_dir.name}_throughput_metrics.json"))
                if throughput_files:
                    with open(throughput_files[0], 'r') as f:
                        throughput_data = json.load(f)
                        # Merge throughput metrics into run data
                        run_data.update(throughput_data)
                
                results.append(run_data)
                
        except Exception as e:
            logger.warning(f"Could not load metrics from {run_dir}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} individual run results")
    return results


def extract_numeric_metrics(run_results: List[Dict], logger: logging.Logger) -> pd.DataFrame:
    """
    Extract all numeric metrics from run results into pandas DataFrame.
    
    Args:
        run_results: List of run dictionaries
        logger: Logger instance
        
    Returns:
        DataFrame with numeric metrics as columns
    """
    logger.info("Extracting numeric metrics from run results...")
    
    flattened_data = []
    
    for run in run_results:
        if not run.get('success', True):
            logger.debug(f"Skipping failed run: {run.get('run_id', 'unknown')}")
            continue
            
        row = {'run_id': run.get('run_id', 0)}
        
        # Basic execution metrics
        row['execution_time_seconds'] = run.get('execution_time_seconds', 0)
        row['execution_time_minutes'] = run.get('execution_time_minutes', 0)
        
        # Stage timings
        stage_timings = run.get('stage_timings', {})
        for stage_key, timing in stage_timings.items():
            if isinstance(timing, (int, float)):
                row[stage_key] = timing
        
        # System metrics
        system_metrics = run.get('system_metrics', {})
        for metric_key, value in system_metrics.items():
            if isinstance(value, (int, float)):
                row[f"system_{metric_key}"] = value
        
        # Memory metrics (various possible sources)
        memory_metrics = [
            'memory_rss', 'memory_vms', 'memory_percent', 'memory_delta_mb',
            'peak_memory_tracemalloc', 'peak_memory_rusage', 'system_memory_percent'
        ]
        for metric in memory_metrics:
            if metric in run:
                row[metric] = run[metric]
            elif metric in system_metrics:
                row[metric] = system_metrics[metric]
        
        # CPU metrics (from system or direct)
        cpu_metrics = [
            'cpu_percent', 'system_cpu_percent', 'cpu_cores_avg',
            'mean_cpu_utilization_percent', 'peak_cpu_utilization_percent'
        ]
        for metric in cpu_metrics:
            if metric in run:
                row[metric] = run[metric]
            elif metric in system_metrics:
                row[metric] = system_metrics[metric]
        
        # Per-core CPU metrics (up to 8 cores for Apple Silicon)
        for i in range(8):
            core_key = f'cpu_core_{i}'
            if core_key in run:
                row[core_key] = run[core_key]
        
        # Temperature metrics
        temp_metrics = [
            'max_temperature_celsius', 'cpu_die_temperature', 'max_temperature_reached_celsius',
            'headroom_celsius', 'headroom_percent'
        ]
        for metric in temp_metrics:
            if metric in run:
                row[metric] = run[metric]
        
        # Throughput metrics (per stage)
        throughput_prefixes = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']
        throughput_suffixes = ['_rows_per_second', '_rows_per_minute', '_throughput']
        
        for prefix in throughput_prefixes:
            for suffix in throughput_suffixes:
                metric_key = f"{prefix}{suffix}"
                if metric_key in run:
                    row[metric_key] = run[metric_key]
        
        # Overall throughput metrics
        overall_throughput_metrics = [
            'overall_throughput_rows_per_second', 'overall_throughput_rows_per_minute',
            'pipeline_throughput_average', 'efficiency_score'
        ]
        for metric in overall_throughput_metrics:
            if metric in run:
                row[metric] = run[metric]
        
        # Compression metrics
        compression_metrics = [
            'compression_ratio', 'compression_ratio_stage1_to_stage2', 
            'compression_ratio_stage2_to_stage4', 'compression_ratio_end_to_end',
            'space_saved_percent', 'space_saved_percent_stage1_to_stage2',
            'space_saved_percent_stage2_to_stage4', 'space_saved_percent_end_to_end'
        ]
        for metric in compression_metrics:
            if metric in run:
                row[metric] = run[metric]
        
        # File size metrics
        stage_artifacts = run.get('stage_artifacts', {})
        for stage, artifacts in stage_artifacts.items():
            if isinstance(artifacts, dict):
                for size_key in ['size_bytes', 'size_mb', 'size_gb']:
                    if size_key in artifacts:
                        row[f"{stage}_{size_key}"] = artifacts[size_key]
        
        flattened_data.append(row)
    
    if not flattened_data:
        raise ValueError("No successful runs found with numeric data")
    
    df = pd.DataFrame(flattened_data)
    
    # Convert to numeric where possible, keeping NaN for missing values
    numeric_columns = []
    for col in df.columns:
        if col != 'run_id':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():  # Only keep columns with some numeric data
                    numeric_columns.append(col)
            except:
                continue
    
    # Select only numeric columns plus run_id
    df = df[['run_id'] + numeric_columns]
    
    logger.info(f"Extracted {len(numeric_columns)} numeric metrics from {len(df)} successful runs")
    logger.debug(f"Numeric columns: {numeric_columns}")
    
    return df


def compute_aggregate_statistics(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive aggregate statistics for all numeric metrics.
    
    Args:
        df: DataFrame with numeric metrics
        logger: Logger instance
        
    Returns:
        Dictionary with statistics for each metric
    """
    logger.info("Computing aggregate statistics...")
    
    stats_dict = {}
    numeric_columns = [col for col in df.columns if col != 'run_id']
    
    for col in numeric_columns:
        # Skip columns with all NaN values
        series = df[col].dropna()
        if len(series) == 0:
            logger.debug(f"Skipping column {col} - no valid data")
            continue
        
        if len(series) < 2:
            logger.debug(f"Skipping column {col} - insufficient data for statistics")
            continue
        
        try:
            # Basic statistics
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std())
            min_val = float(series.min())
            max_val = float(series.max())
            count = int(len(series))
            
            # Coefficient of variation for stability analysis
            cv = std_val / mean_val if mean_val != 0 else np.inf
            
            # 95% confidence interval using t-distribution
            confidence_level = 0.95
            alpha = 1 - confidence_level
            degrees_freedom = count - 1
            t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
            margin_error = t_critical * (std_val / np.sqrt(count))
            ci_lower = mean_val - margin_error
            ci_upper = mean_val + margin_error
            
            stats_dict[col] = {
                'count': count,
                'mean': round(mean_val, 6),
                'median': round(median_val, 6),
                'std': round(std_val, 6),
                'min': round(min_val, 6),
                'max': round(max_val, 6),
                'coefficient_of_variation': round(cv, 6),
                'ci_95_lower': round(ci_lower, 6),
                'ci_95_upper': round(ci_upper, 6),
                'margin_of_error': round(margin_error, 6)
            }
            
        except Exception as e:
            logger.warning(f"Could not compute statistics for {col}: {e}")
            continue
    
    logger.info(f"Computed aggregate statistics for {len(stats_dict)} metrics")
    
    return stats_dict


def generate_stability_analysis(stats_dict: Dict[str, Dict[str, float]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Generate stability analysis based on coefficient of variation.
    
    Args:
        stats_dict: Dictionary with statistics for each metric
        logger: Logger instance
        
    Returns:
        Dictionary with stability analysis results
    """
    logger.info("Generating stability analysis...")
    
    # Classification thresholds for coefficient of variation
    cv_thresholds = {
        'excellent': 0.05,  # CV <= 5%
        'good': 0.10,       # CV <= 10%
        'moderate': 0.20,   # CV <= 20%
        'high': 0.50,       # CV <= 50%
        'very_high': float('inf')  # CV > 50%
    }
    
    stability_categories = {category: [] for category in cv_thresholds.keys()}
    
    for metric, stats in stats_dict.items():
        cv = stats['coefficient_of_variation']
        
        # Skip infinite CV values
        if np.isinf(cv):
            continue
            
        # Classify stability
        for category, threshold in cv_thresholds.items():
            if cv <= threshold:
                stability_categories[category].append({
                    'metric': metric,
                    'cv': cv,
                    'mean': stats['mean'],
                    'std': stats['std']
                })
                break
    
    # Summary statistics
    all_cvs = [stats['coefficient_of_variation'] for stats in stats_dict.values() 
               if not np.isinf(stats['coefficient_of_variation'])]
    
    stability_summary = {
        'total_metrics': len(stats_dict),
        'metrics_with_valid_cv': len(all_cvs),
        'mean_cv_across_metrics': round(np.mean(all_cvs), 4) if all_cvs else 0,
        'median_cv_across_metrics': round(np.median(all_cvs), 4) if all_cvs else 0,
        'categories': {cat: len(metrics) for cat, metrics in stability_categories.items()},
        'detailed_categories': stability_categories
    }
    
    logger.info(f"Stability analysis: {stability_summary['categories']}")
    
    return stability_summary


def save_results(stats_dict: Dict[str, Dict[str, float]], 
                stability_analysis: Dict[str, Any],
                output_dir: Path, 
                logger: logging.Logger) -> None:
    """
    Save aggregate statistics and stability analysis to CSV and JSON files.
    
    Args:
        stats_dict: Dictionary with statistics for each metric
        stability_analysis: Stability analysis results
        output_dir: Directory to save results
        logger: Logger instance
    """
    logger.info("Saving aggregate statistics...")
    
    # Prepare DataFrame for CSV export
    stats_rows = []
    for metric, stats in stats_dict.items():
        row = {'metric': metric}
        row.update(stats)
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    
    # Sort by coefficient of variation for easy interpretation
    stats_df = stats_df.sort_values('coefficient_of_variation')
    
    # Save CSV
    csv_file = output_dir / "summary_stats.csv"
    stats_df.to_csv(csv_file, index=False, float_format='%.6f')
    logger.info(f"Summary statistics saved to: {csv_file}")
    
    # Prepare comprehensive JSON output
    json_output = {
        'metadata': {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'total_metrics': len(stats_dict),
            'description': 'Aggregate statistics across benchmark runs with 95% confidence intervals'
        },
        'aggregate_statistics': stats_dict,
        'stability_analysis': stability_analysis
    }
    
    # Save JSON
    json_file = output_dir / "summary_stats.json"
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"Summary statistics saved to: {json_file}")
    
    # Print summary to console
    print(f"\n✅ Aggregate Statistics Summary")
    print(f"=" * 50)
    print(f"Total metrics analyzed: {len(stats_dict)}")
    print(f"Stability categories: {stability_analysis['categories']}")
    print(f"Mean CV across metrics: {stability_analysis['mean_cv_across_metrics']:.4f}")
    print(f"Results saved to:")
    print(f"  - CSV: {csv_file}")
    print(f"  - JSON: {json_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark statistics across runs (Step 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/utils/aggregate_statistics.py outputs/benchmarks/20241215_143022/
    
    # With verbose logging
    python scripts/utils/aggregate_statistics.py outputs/benchmarks/20241215_143022/ --verbose
    
    # Custom output directory
    python scripts/utils/aggregate_statistics.py outputs/benchmarks/20241215_143022/ \\
        --output_dir outputs/final_stats/
"""
    )
    
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Directory containing benchmark results (e.g., outputs/benchmarks/20241215_143022/)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save aggregate statistics (default: same as benchmark_dir/summary)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.benchmark_dir.exists():
        print(f"Error: Benchmark directory does not exist: {args.benchmark_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.benchmark_dir / "summary"
        output_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("AGGREGATE STATISTICS ACROSS RUNS - START")
    logger.info("=" * 60)
    logger.info(f"Benchmark directory: {args.benchmark_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Load benchmark results
        run_results = load_benchmark_results(args.benchmark_dir, logger)
        if not run_results:
            raise ValueError("No benchmark results found")
        
        # Step 2: Extract numeric metrics into DataFrame
        df = extract_numeric_metrics(run_results, logger)
        
        # Step 3: Compute aggregate statistics
        stats_dict = compute_aggregate_statistics(df, logger)
        
        # Step 4: Generate stability analysis
        stability_analysis = generate_stability_analysis(stats_dict, logger)
        
        # Step 5: Save results
        save_results(stats_dict, stability_analysis, output_dir, logger)
        
        logger.info("=" * 60)
        logger.info("AGGREGATE STATISTICS COMPUTATION COMPLETE")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during aggregate statistics computation: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
