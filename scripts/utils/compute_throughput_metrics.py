#!/usr/bin/env python3
"""
compute_throughput_metrics.py

CMMSE 2025: Throughput and Compression Metrics Calculator
========================================================

Computes comprehensive throughput and compression metrics for each pipeline run:
• Rows/sec per stage: using known row counts divided by stage time
• Compression ratio: ingested bytes → preprocessed → individual anomalies  
• CPU efficiency: mean & peak CPU usage vs wall time
• Thermal headroom: max temperature reached
• Persists results in per-run JSON format

This script analyzes run metrics and resource monitoring data to compute:
1. Stage-specific throughput (rows processed per second)
2. Data compression efficiency between pipeline stages
3. CPU utilization efficiency vs execution time
4. Thermal performance and headroom analysis

Usage:
    python scripts/utils/compute_throughput_metrics.py <benchmark_dir>
    
Example:
    python scripts/utils/compute_throughput_metrics.py outputs/benchmarks/20241215_143022/

Author: José Miguel Franco-Valiente
Created: December 2024
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known row counts from pipeline stages (stored constants)
STAGE_ROW_COUNTS = {
    'stage1': 319_900_000,  # 319.9M rows from data ingestion
    'stage2': 89_200_000,   # 89.2M rows after preprocessing (72% compression)
    'stage3': 39_400,       # 39.4K reference weeks selected
    'stage4': 5_650_000,    # 5.65M individual anomalies detected (estimated)
    'stage5': 10_000        # 10K cells analyzed
}

# Stage names for better reporting
STAGE_NAMES = {
    'stage1': 'Data Ingestion',
    'stage2': 'Data Preprocessing', 
    'stage3': 'Reference Week Selection',
    'stage4': 'Individual Anomaly Detection',
    'stage5': 'Comprehensive Analysis'
}


class ThroughputMetricsCalculator:
    """
    Calculator for comprehensive throughput and compression metrics.
    
    Analyzes benchmark run data to compute:
    - Stage throughput (rows/second)
    - Compression ratios between stages
    - CPU efficiency metrics
    - Thermal performance analysis
    """
    
    def __init__(self, benchmark_dir: Path):
        """
        Initialize the metrics calculator.
        
        Args:
            benchmark_dir: Path to benchmark directory containing run folders
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.runs_data = []
        
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
            
        logger.info(f"Initialized metrics calculator for: {benchmark_dir}")
    
    def load_run_data(self) -> List[Dict]:
        """
        Load run metrics and resource monitoring data for all runs.
        
        Returns:
            List of run data dictionaries with metrics and monitoring info
        """
        run_dirs = sorted([d for d in self.benchmark_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run_')])
        
        logger.info(f"Found {len(run_dirs)} run directories")
        
        for run_dir in run_dirs:
            try:
                run_data = self._load_single_run_data(run_dir)
                if run_data:
                    self.runs_data.append(run_data)
                    logger.debug(f"Loaded data for {run_dir.name}")
                else:
                    logger.warning(f"No valid data found for {run_dir.name}")
            except Exception as e:
                logger.error(f"Error loading data for {run_dir.name}: {e}")
        
        logger.info(f"Successfully loaded data for {len(self.runs_data)} runs")
        return self.runs_data
    
    def _load_single_run_data(self, run_dir: Path) -> Optional[Dict]:
        """
        Load data for a single run including metrics and monitoring.
        
        Args:
            run_dir: Path to individual run directory
            
        Returns:
            Dictionary with run data or None if loading failed
        """
        # Load run metrics
        metrics_file = run_dir / "run_metrics.json"
        if not metrics_file.exists():
            logger.warning(f"No run_metrics.json found in {run_dir}")
            return None
        
        with open(metrics_file, 'r') as f:
            run_metrics = json.load(f)
        
        # Load resource monitoring data if available
        monitor_file = run_dir / "resource_monitor.csv"
        monitoring_data = None
        if monitor_file.exists():
            try:
                monitoring_data = pd.read_csv(monitor_file)
                logger.debug(f"Loaded {len(monitoring_data)} monitoring samples for {run_dir.name}")
            except Exception as e:
                logger.warning(f"Could not load monitoring data for {run_dir.name}: {e}")
        
        # Load stage artifacts information
        data_dir = run_dir / "data"
        artifacts_info = {}
        if data_dir.exists():
            artifacts_info = self._collect_artifacts_info(data_dir)
        
        return {
            'run_dir': run_dir,
            'run_id': run_metrics.get('run_id', run_dir.name),
            'success': run_metrics.get('success', False),
            'run_metrics': run_metrics,
            'monitoring_data': monitoring_data,
            'artifacts_info': artifacts_info
        }
    
    def _collect_artifacts_info(self, data_dir: Path) -> Dict[str, Dict]:
        """
        Collect file size information for stage artifacts.
        
        Args:
            data_dir: Directory containing pipeline stage outputs
            
        Returns:
            Dictionary with artifact file information
        """
        artifacts = {}
        
        # Expected stage output files
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
                    logger.warning(f"Error getting size for {filename}: {e}")
        
        return artifacts
    
    def compute_throughput_metrics(self, run_data: Dict) -> Dict[str, Any]:
        """
        Compute throughput metrics for a single run.
        
        Args:
            run_data: Run data dictionary with metrics and monitoring
            
        Returns:
            Dictionary with computed throughput metrics
        """
        metrics = {}
        
        # Extract stage timings
        stage_timings = run_data['run_metrics'].get('stage_timings', {})
        
        # Compute rows/sec per stage
        throughput_metrics = {}
        for stage, row_count in STAGE_ROW_COUNTS.items():
            timing_key = f'{stage}_time'
            if timing_key in stage_timings:
                stage_time = stage_timings[timing_key]
                if stage_time > 0:
                    rows_per_sec = row_count / stage_time
                    throughput_metrics[stage] = {
                        'stage_name': STAGE_NAMES.get(stage, stage),
                        'total_rows': row_count,
                        'execution_time_seconds': stage_time,
                        'rows_per_second': round(rows_per_sec, 0),
                        'rows_per_minute': round(rows_per_sec * 60, 0),
                        'throughput_category': self._categorize_throughput(rows_per_sec)
                    }
        
        metrics['throughput_per_stage'] = throughput_metrics
        
        # Compute overall pipeline throughput
        total_time = run_data['run_metrics'].get('execution_time_seconds', 0)
        total_rows_processed = sum(STAGE_ROW_COUNTS.values())
        if total_time > 0:
            overall_throughput = total_rows_processed / total_time
            metrics['overall_throughput'] = {
                'total_rows_processed': total_rows_processed,
                'total_execution_time_seconds': total_time,
                'average_rows_per_second': round(overall_throughput, 0),
                'average_rows_per_minute': round(overall_throughput * 60, 0)
            }
        
        return metrics
    
    def compute_compression_metrics(self, run_data: Dict) -> Dict[str, Any]:
        """
        Compute compression ratios between pipeline stages.
        
        Args:
            run_data: Run data dictionary with artifacts information
            
        Returns:
            Dictionary with compression metrics
        """
        metrics = {}
        artifacts = run_data.get('artifacts_info', {})
        
        # Compression pipeline: ingested → preprocessed → individual anomalies
        compression_chain = ['stage1', 'stage2', 'stage4']
        stage_sizes = {}
        
        for stage in compression_chain:
            if stage in artifacts:
                stage_sizes[stage] = artifacts[stage]['size_bytes']
        
        # Compute compression ratios
        compression_ratios = {}
        
        if 'stage1' in stage_sizes and 'stage2' in stage_sizes:
            # Ingested → Preprocessed compression
            ratio_1_2 = stage_sizes['stage1'] / stage_sizes['stage2']
            compression_ratios['ingested_to_preprocessed'] = {
                'original_size_bytes': stage_sizes['stage1'],
                'compressed_size_bytes': stage_sizes['stage2'],
                'compression_ratio': round(ratio_1_2, 2),
                'space_saved_percent': round((1 - 1/ratio_1_2) * 100, 1),
                'original_size_gb': round(stage_sizes['stage1'] / (1024**3), 3),
                'compressed_size_gb': round(stage_sizes['stage2'] / (1024**3), 3)
            }
        
        if 'stage2' in stage_sizes and 'stage4' in stage_sizes:
            # Preprocessed → Individual anomalies compression
            ratio_2_4 = stage_sizes['stage2'] / stage_sizes['stage4']
            compression_ratios['preprocessed_to_anomalies'] = {
                'original_size_bytes': stage_sizes['stage2'],
                'compressed_size_bytes': stage_sizes['stage4'],
                'compression_ratio': round(ratio_2_4, 2),
                'space_saved_percent': round((1 - 1/ratio_2_4) * 100, 1),
                'original_size_gb': round(stage_sizes['stage2'] / (1024**3), 3),
                'compressed_size_gb': round(stage_sizes['stage4'] / (1024**3), 3)
            }
        
        if 'stage1' in stage_sizes and 'stage4' in stage_sizes:
            # End-to-end compression (ingested → anomalies)
            ratio_1_4 = stage_sizes['stage1'] / stage_sizes['stage4']
            compression_ratios['end_to_end'] = {
                'original_size_bytes': stage_sizes['stage1'],
                'final_size_bytes': stage_sizes['stage4'],
                'compression_ratio': round(ratio_1_4, 2),
                'space_saved_percent': round((1 - 1/ratio_1_4) * 100, 1),
                'original_size_gb': round(stage_sizes['stage1'] / (1024**3), 3),
                'final_size_gb': round(stage_sizes['stage4'] / (1024**3), 3)
            }
        
        metrics['compression_ratios'] = compression_ratios
        metrics['stage_file_sizes'] = {
            stage: {
                'size_bytes': size,
                'size_mb': round(size / (1024**2), 2),
                'size_gb': round(size / (1024**3), 3)
            }
            for stage, size in stage_sizes.items()
        }
        
        return metrics
    
    def compute_cpu_efficiency_metrics(self, run_data: Dict) -> Dict[str, Any]:
        """
        Compute CPU efficiency metrics: mean & peak CPU usage vs wall time.
        
        Args:
            run_data: Run data dictionary with monitoring data
            
        Returns:
            Dictionary with CPU efficiency metrics
        """
        metrics = {}
        monitoring_data = run_data.get('monitoring_data')
        
        if monitoring_data is None or monitoring_data.empty:
            logger.warning(f"No monitoring data available for run {run_data['run_id']}")
            return {'cpu_efficiency': 'monitoring_data_unavailable'}
        
        # Extract CPU metrics from monitoring data
        cpu_columns = [col for col in monitoring_data.columns if 'cpu' in col.lower()]
        
        # System-wide CPU metrics
        if 'system_cpu_percent' in monitoring_data.columns:
            system_cpu = monitoring_data['system_cpu_percent'].dropna()
            metrics['system_cpu'] = {
                'mean_percent': round(system_cpu.mean(), 2),
                'peak_percent': round(system_cpu.max(), 2),
                'min_percent': round(system_cpu.min(), 2),
                'std_percent': round(system_cpu.std(), 2)
            }
        
        # Process-specific CPU metrics
        if 'cpu_percent' in monitoring_data.columns:
            process_cpu = monitoring_data['cpu_percent'].dropna()
            metrics['process_cpu'] = {
                'mean_percent': round(process_cpu.mean(), 2),
                'peak_percent': round(process_cpu.max(), 2),
                'min_percent': round(process_cpu.min(), 2),
                'std_percent': round(process_cpu.std(), 2)
            }
        
        # Per-core CPU analysis
        core_columns = [col for col in monitoring_data.columns if col.startswith('cpu_core_')]
        if core_columns:
            core_metrics = {}
            for col in core_columns:
                core_data = monitoring_data[col].dropna()
                if not core_data.empty:
                    core_metrics[col] = {
                        'mean_percent': round(core_data.mean(), 2),
                        'peak_percent': round(core_data.max(), 2)
                    }
            if core_metrics:
                metrics['per_core_cpu'] = core_metrics
        
        # Compute CPU efficiency scores
        execution_time = run_data['run_metrics'].get('execution_time_seconds', 0)
        if execution_time > 0 and 'system_cpu' in metrics:
            mean_cpu = metrics['system_cpu']['mean_percent']
            peak_cpu = metrics['system_cpu']['peak_percent']
            
            # CPU efficiency: work done per CPU utilization
            # Higher efficiency = more work with less CPU usage
            metrics['cpu_efficiency'] = {
                'mean_cpu_utilization_percent': mean_cpu,
                'peak_cpu_utilization_percent': peak_cpu,
                'execution_time_seconds': execution_time,
                'cpu_time_product': round(mean_cpu * execution_time, 2),
                'efficiency_score': round(100 / max(mean_cpu, 1), 2),  # Higher is better
                'peak_efficiency_score': round(100 / max(peak_cpu, 1), 2)
            }
        
        return metrics
    
    def compute_thermal_metrics(self, run_data: Dict) -> Dict[str, Any]:
        """
        Compute thermal headroom: max temperature reached.
        
        Args:
            run_data: Run data dictionary with monitoring data
            
        Returns:
            Dictionary with thermal metrics
        """
        metrics = {}
        monitoring_data = run_data.get('monitoring_data')
        
        if monitoring_data is None or monitoring_data.empty:
            logger.warning(f"No monitoring data available for thermal analysis of run {run_data['run_id']}")
            return {'thermal_headroom': 'monitoring_data_unavailable'}
        
        # Look for temperature columns
        temp_columns = [col for col in monitoring_data.columns if 'temperature' in col.lower()]
        
        if not temp_columns:
            logger.info(f"No temperature data available for run {run_data['run_id']}")
            return {'thermal_headroom': 'temperature_data_unavailable'}
        
        thermal_data = {}
        for col in temp_columns:
            temp_data = monitoring_data[col].dropna()
            if not temp_data.empty:
                thermal_data[col] = {
                    'max_temperature_celsius': round(temp_data.max(), 2),
                    'mean_temperature_celsius': round(temp_data.mean(), 2),
                    'min_temperature_celsius': round(temp_data.min(), 2),
                    'temperature_range': round(temp_data.max() - temp_data.min(), 2)
                }
        
        # Compute thermal headroom (assuming typical Apple Silicon thermal limit around 100°C)
        apple_silicon_thermal_limit = 100.0  # °C
        
        if thermal_data:
            max_temps = [data['max_temperature_celsius'] for data in thermal_data.values()]
            overall_max_temp = max(max_temps)
            
            metrics['thermal_headroom'] = {
                'max_temperature_reached_celsius': overall_max_temp,
                'thermal_limit_celsius': apple_silicon_thermal_limit,
                'headroom_celsius': round(apple_silicon_thermal_limit - overall_max_temp, 2),
                'headroom_percent': round((apple_silicon_thermal_limit - overall_max_temp) / apple_silicon_thermal_limit * 100, 1),
                'thermal_efficiency': 'good' if overall_max_temp < 80 else 'moderate' if overall_max_temp < 90 else 'high'
            }
            
            metrics['temperature_details'] = thermal_data
        else:
            metrics['thermal_headroom'] = 'no_temperature_sensors_detected'
        
        return metrics
    
    def _categorize_throughput(self, rows_per_sec: float) -> str:
        """Categorize throughput performance."""
        if rows_per_sec >= 1_000_000:
            return 'excellent'
        elif rows_per_sec >= 500_000:
            return 'good'
        elif rows_per_sec >= 100_000:
            return 'moderate'
        else:
            return 'low'
    
    def compute_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Compute all metrics for all loaded runs.
        
        Returns:
            List of dictionaries with complete metrics for each run
        """
        if not self.runs_data:
            logger.error("No run data loaded. Call load_run_data() first.")
            return []
        
        all_metrics = []
        
        for run_data in self.runs_data:
            logger.info(f"Computing metrics for run {run_data['run_id']}")
            
            run_metrics = {
                'run_id': run_data['run_id'],
                'success': run_data['success'],
                'timestamp': datetime.now().isoformat(),
                'benchmark_dir': str(self.benchmark_dir)
            }
            
            try:
                # Compute all metric categories
                run_metrics['throughput_metrics'] = self.compute_throughput_metrics(run_data)
                run_metrics['compression_metrics'] = self.compute_compression_metrics(run_data)
                run_metrics['cpu_efficiency_metrics'] = self.compute_cpu_efficiency_metrics(run_data)
                run_metrics['thermal_metrics'] = self.compute_thermal_metrics(run_data)
                
                # Add execution summary
                execution_time = run_data['run_metrics'].get('execution_time_seconds', 0)
                run_metrics['execution_summary'] = {
                    'total_execution_time_seconds': execution_time,
                    'total_execution_time_minutes': round(execution_time / 60, 2),
                    'parameters': run_data['run_metrics'].get('parameters', {}),
                    'success': run_data['success']
                }
                
                logger.info(f"Successfully computed metrics for run {run_data['run_id']}")
                
            except Exception as e:
                logger.error(f"Error computing metrics for run {run_data['run_id']}: {e}")
                run_metrics['error'] = str(e)
            
            all_metrics.append(run_metrics)
        
        return all_metrics
    
    def save_metrics_to_json(self, metrics: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> Path:
        """
        Save computed metrics to JSON files.
        
        Args:
            metrics: List of run metrics
            output_dir: Output directory (defaults to benchmark_dir/summary)
            
        Returns:
            Path to saved metrics file
        """
        if output_dir is None:
            output_dir = self.benchmark_dir / "summary"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual run metrics
        for run_metrics in metrics:
            run_id = run_metrics['run_id']
            filename = f"run_{run_id}_throughput_metrics.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(run_metrics, f, indent=2)
            
            logger.debug(f"Saved metrics for run {run_id} to {filepath}")
        
        # Save consolidated metrics
        consolidated_file = output_dir / "all_runs_throughput_metrics.json"
        with open(consolidated_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved consolidated metrics to {consolidated_file}")
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(metrics)
        summary_file = output_dir / "throughput_metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Saved summary statistics to {summary_file}")
        
        return consolidated_file
    
    def _generate_summary_statistics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics across all runs."""
        successful_runs = [m for m in metrics if m.get('success', False)]
        
        if not successful_runs:
            return {'error': 'No successful runs to analyze'}
        
        summary = {
            'total_runs': len(metrics),
            'successful_runs': len(successful_runs),
            'analysis_timestamp': datetime.now().isoformat(),
            'summary_statistics': {}
        }
        
        # Throughput statistics
        stage_throughputs = {}
        for stage in STAGE_ROW_COUNTS.keys():
            throughputs = []
            for run in successful_runs:
                stage_metrics = run.get('throughput_metrics', {}).get('throughput_per_stage', {})
                if stage in stage_metrics:
                    throughputs.append(stage_metrics[stage]['rows_per_second'])
            
            if throughputs:
                stage_throughputs[stage] = {
                    'mean_rows_per_second': round(np.mean(throughputs), 0),
                    'median_rows_per_second': round(np.median(throughputs), 0),
                    'std_rows_per_second': round(np.std(throughputs), 0),
                    'min_rows_per_second': round(np.min(throughputs), 0),
                    'max_rows_per_second': round(np.max(throughputs), 0)
                }
        
        summary['summary_statistics']['throughput_per_stage'] = stage_throughputs
        
        # CPU efficiency statistics
        cpu_efficiencies = []
        for run in successful_runs:
            cpu_metrics = run.get('cpu_efficiency_metrics', {})
            if 'cpu_efficiency' in cpu_metrics and isinstance(cpu_metrics['cpu_efficiency'], dict):
                efficiency = cpu_metrics['cpu_efficiency'].get('efficiency_score')
                if efficiency is not None:
                    cpu_efficiencies.append(efficiency)
        
        if cpu_efficiencies:
            summary['summary_statistics']['cpu_efficiency'] = {
                'mean_efficiency_score': round(np.mean(cpu_efficiencies), 2),
                'median_efficiency_score': round(np.median(cpu_efficiencies), 2),
                'std_efficiency_score': round(np.std(cpu_efficiencies), 2)
            }
        
        # Thermal statistics
        max_temperatures = []
        for run in successful_runs:
            thermal_metrics = run.get('thermal_metrics', {})
            if 'thermal_headroom' in thermal_metrics and isinstance(thermal_metrics['thermal_headroom'], dict):
                max_temp = thermal_metrics['thermal_headroom'].get('max_temperature_reached_celsius')
                if max_temp is not None:
                    max_temperatures.append(max_temp)
        
        if max_temperatures:
            summary['summary_statistics']['thermal_performance'] = {
                'mean_max_temperature_celsius': round(np.mean(max_temperatures), 2),
                'peak_temperature_celsius': round(np.max(max_temperatures), 2),
                'min_temperature_celsius': round(np.min(max_temperatures), 2)
            }
        
        return summary


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Compute throughput and compression metrics for pipeline benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compute metrics for a benchmark directory
    python scripts/utils/compute_throughput_metrics.py outputs/benchmarks/20241215_143022/
    
    # With verbose output
    python scripts/utils/compute_throughput_metrics.py outputs/benchmarks/20241215_143022/ --verbose
    
    # Custom output directory
    python scripts/utils/compute_throughput_metrics.py outputs/benchmarks/20241215_143022/ --output_dir results/metrics/
        """
    )
    
    parser.add_argument(
        'benchmark_dir',
        type=Path,
        help='Path to benchmark directory containing run folders'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory for metrics files (default: benchmark_dir/summary)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize calculator
        calculator = ThroughputMetricsCalculator(args.benchmark_dir)
        
        # Load run data
        logger.info("Loading run data...")
        calculator.load_run_data()
        
        if not calculator.runs_data:
            logger.error("No valid run data found!")
            sys.exit(1)
        
        # Compute all metrics
        logger.info("Computing throughput and compression metrics...")
        all_metrics = calculator.compute_all_metrics()
        
        # Save results
        logger.info("Saving metrics to JSON files...")
        output_file = calculator.save_metrics_to_json(all_metrics, args.output_dir)
        
        logger.info(f"✅ Metrics computation completed successfully!")
        logger.info(f"Results saved to: {output_file.parent}")
        logger.info(f"Processed {len(all_metrics)} runs")
        
        # Print summary
        successful_runs = sum(1 for m in all_metrics if m.get('success', False))
        logger.info(f"Successful runs: {successful_runs}/{len(all_metrics)}")
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
