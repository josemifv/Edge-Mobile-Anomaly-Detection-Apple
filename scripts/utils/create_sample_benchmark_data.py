#!/usr/bin/env python3
"""
create_sample_benchmark_data.py

Creates sample benchmark data for testing aggregate statistics functionality.
This script generates realistic benchmark results similar to what would be produced
by the pipeline_benchmark_runner.py script.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def create_sample_run_data(run_id: int) -> Dict:
    """Create sample data for a single benchmark run."""
    
    # Base execution time with some variation (around 600-900 seconds)
    base_time = 750 + random.normalvariate(0, 50)
    
    # Stage timings (proportional to typical pipeline)
    stage1_time = base_time * 0.15 + random.normalvariate(0, 5)  # ~15% for data ingestion
    stage2_time = base_time * 0.20 + random.normalvariate(0, 8)  # ~20% for preprocessing  
    stage3_time = base_time * 0.05 + random.normalvariate(0, 2)  # ~5% for week selection
    stage4_time = base_time * 0.55 + random.normalvariate(0, 15) # ~55% for anomaly detection
    stage5_time = base_time * 0.05 + random.normalvariate(0, 2)  # ~5% for analysis
    
    total_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time
    
    # Memory metrics (MB)
    initial_memory = 2000 + random.normalvariate(0, 100)
    memory_delta = 3000 + random.normalvariate(0, 200)
    final_memory = initial_memory + memory_delta
    
    # CPU metrics
    mean_cpu = 45 + random.normalvariate(0, 10)
    peak_cpu = mean_cpu + 20 + random.normalvariate(0, 5)
    
    # Temperature (Celsius) - Apple Silicon typical range
    max_temp = 65 + random.normalvariate(0, 8)
    headroom = 100 - max_temp  # Thermal limit is 100°C
    
    # Throughput metrics (rows per second)
    stage1_throughput = 1.8e6 + random.normalvariate(0, 0.2e6)
    stage2_throughput = 0.6e6 + random.normalvariate(0, 0.1e6)
    stage3_throughput = 0.8e6 + random.normalvariate(0, 0.1e6)
    stage4_throughput = 0.2e6 + random.normalvariate(0, 0.05e6)
    stage5_throughput = 1.0e4 + random.normalvariate(0, 0.2e4)
    
    # Compression ratios
    compression_stage1_to_stage2 = 2.2 + random.normalvariate(0, 0.1)
    compression_stage2_to_stage4 = 13.5 + random.normalvariate(0, 0.5)
    compression_end_to_end = compression_stage1_to_stage2 * compression_stage2_to_stage4
    
    # File sizes (MB)
    stage1_size = 4400 + random.normalvariate(0, 100)
    stage2_size = stage1_size / compression_stage1_to_stage2
    stage3_size = 0.5 + random.normalvariate(0, 0.05)
    stage4_size = stage2_size / (compression_stage2_to_stage4 * 0.9)  # Slight variation
    
    success = random.random() > 0.05  # 95% success rate
    
    run_data = {
        'run_id': run_id,
        'success': success,
        'execution_time_seconds': round(total_time, 2),
        'execution_time_minutes': round(total_time / 60, 2),
        'start_timestamp': (datetime.now()).isoformat(),
        'end_timestamp': (datetime.now()).isoformat(),
        'parameters': {
            'n_components': 3,
            'anomaly_threshold': 2.0,
            'max_workers': 8,
            'input_dir': 'data/raw/'
        },
        'system_metrics': {
            'initial_memory_mb': round(initial_memory, 2),
            'final_memory_mb': round(final_memory, 2),
            'memory_delta_mb': round(memory_delta, 2),
            'initial_cpu_percent': 5.0,
            'final_cpu_percent': 8.0
        },
        'stage_timings': {
            'stage1_time': round(stage1_time, 2),
            'stage2_time': round(stage2_time, 2),
            'stage3_time': round(stage3_time, 2),
            'stage4_time': round(stage4_time, 2),
            'stage5_time': round(stage5_time, 2),
            'total_pipeline_time': round(total_time, 2)
        },
        'stage_artifacts': {
            'stage1': {'size_bytes': int(stage1_size * 1024 * 1024), 'size_mb': round(stage1_size, 2), 'size_gb': round(stage1_size / 1024, 3)},
            'stage2': {'size_bytes': int(stage2_size * 1024 * 1024), 'size_mb': round(stage2_size, 2), 'size_gb': round(stage2_size / 1024, 3)},
            'stage3': {'size_bytes': int(stage3_size * 1024 * 1024), 'size_mb': round(stage3_size, 2), 'size_gb': round(stage3_size / 1024, 3)},
            'stage4': {'size_bytes': int(stage4_size * 1024 * 1024), 'size_mb': round(stage4_size, 2), 'size_gb': round(stage4_size / 1024, 3)}
        },
        # Additional throughput and efficiency metrics
        'mean_cpu_utilization_percent': round(mean_cpu, 1),
        'peak_cpu_utilization_percent': round(peak_cpu, 1),
        'max_temperature_celsius': round(max_temp, 1),
        'headroom_celsius': round(headroom, 1),
        'headroom_percent': round((headroom / 100) * 100, 1),
        'stage1_rows_per_second': int(stage1_throughput),
        'stage2_rows_per_second': int(stage2_throughput),
        'stage3_rows_per_second': int(stage3_throughput),
        'stage4_rows_per_second': int(stage4_throughput),
        'stage5_rows_per_second': int(stage5_throughput),
        'compression_ratio_stage1_to_stage2': round(compression_stage1_to_stage2, 2),
        'compression_ratio_stage2_to_stage4': round(compression_stage2_to_stage4, 2),
        'compression_ratio_end_to_end': round(compression_end_to_end, 2),
        'space_saved_percent_stage1_to_stage2': round((1 - 1/compression_stage1_to_stage2) * 100, 1),
        'space_saved_percent_stage2_to_stage4': round((1 - 1/compression_stage2_to_stage4) * 100, 1),
        'space_saved_percent_end_to_end': round((1 - 1/compression_end_to_end) * 100, 1),
        'error_message': None if success else f"Sample error in run {run_id}"
    }
    
    return run_data


def create_sample_benchmark_dataset(output_dir: Path, num_runs: int = 10) -> None:
    """Create a complete sample benchmark dataset."""
    
    # Create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = output_dir / f"sample_benchmark_{timestamp}"
    summary_dir = benchmark_dir / "summary"
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(exist_ok=True)
    
    # Generate sample runs
    all_runs = []
    
    for run_id in range(1, num_runs + 1):
        run_data = create_sample_run_data(run_id)
        all_runs.append(run_data)
        
        # Create individual run directory and files
        run_dir = benchmark_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Save individual run metrics
        with open(run_dir / "run_metrics.json", 'w') as f:
            json.dump(run_data, f, indent=2)
    
    # Create summary benchmark file
    successful_runs = [r for r in all_runs if r['success']]
    execution_times = [r['execution_time_seconds'] for r in successful_runs]
    
    summary_stats = {
        'total_runs': len(all_runs),
        'successful_runs': len(successful_runs),
        'failed_runs': len(all_runs) - len(successful_runs),
        'success_rate': len(successful_runs) / len(all_runs),
        'execution_time_stats': {
            'mean_seconds': sum(execution_times) / len(execution_times) if execution_times else 0,
            'min_seconds': min(execution_times) if execution_times else 0,
            'max_seconds': max(execution_times) if execution_times else 0
        },
        'system_info': {
            'python_version': '3.13.2',
            'platform': 'macOS-14.0-arm64-arm-64bit',
            'processor': 'arm',
            'cpu_count': 8,
            'cpu_count_logical': 8,
            'memory_total_gb': 32.0,
            'apple_silicon': True,
            'architecture': 'ARM64',
            'timestamp': datetime.now().isoformat()
        },
        'benchmark_timestamp': datetime.now().isoformat()
    }
    
    # Save comprehensive summary
    summary_data = {
        'summary_stats': summary_stats,
        'individual_runs': all_runs
    }
    
    with open(summary_dir / "benchmark_summary.json", 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✅ Sample benchmark dataset created:")
    print(f"   Directory: {benchmark_dir}")
    print(f"   Runs: {num_runs} ({len(successful_runs)} successful)")
    print(f"   Summary: {summary_dir / 'benchmark_summary.json'}")
    
    return benchmark_dir


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample benchmark data for testing")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/benchmarks"), 
                       help="Output directory for sample data")
    parser.add_argument("--runs", type=int, default=10, help="Number of sample runs to create")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    benchmark_dir = create_sample_benchmark_dataset(args.output_dir, args.runs)
    
    print(f"\nTest aggregate statistics with:")
    print(f"python scripts/utils/aggregate_statistics.py {benchmark_dir}")


if __name__ == "__main__":
    main()
