#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script for All Performance Levels

This script tests all implementation levels (0-3) with the same dataset
and generates comparative performance reports for academic research.
"""

import subprocess
import time
import os
import pandas as pd
import json
import psutil
from datetime import datetime
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Performance level configurations
PERFORMANCE_LEVELS = {
    0: {
        'script': 'scripts/01_data_ingestion.py',
        'name': 'Level 0: Baseline',
        'description': 'Original implementation without optimizations'
    },
    1: {
        'script': 'scripts/01_data_ingestion_optimized.py',
        'name': 'Level 1: Conservative',
        'description': 'Low-risk optimizations with proven benefits'
    },
    2: {
        'script': 'scripts/01_data_ingestion_moderate.py',
        'name': 'Level 2: Moderate',
        'description': 'Additional optimizations with balanced risk/reward'
    },
    3: {
        'script': 'scripts/01_data_ingestion_aggressive.py',
        'name': 'Level 3: Aggressive',
        'description': 'High-performance optimizations with complexity'
    }
}

class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.baseline_cpu = psutil.cpu_percent(interval=1)
        self.baseline_memory = psutil.virtual_memory().percent
        self.monitoring = False
        self.samples = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.samples = []
        self.start_time = time.time()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return summary."""
        self.monitoring = False
        if not self.samples:
            return None
        
        df = pd.DataFrame(self.samples)
        return {
            'duration': time.time() - self.start_time,
            'cpu_mean': df['cpu_percent'].mean(),
            'cpu_max': df['cpu_percent'].max(),
            'memory_mean': df['memory_percent'].mean(),
            'memory_max': df['memory_percent'].max(),
            'samples_count': len(self.samples)
        }
    
    def sample_resources(self):
        """Sample current resource usage."""
        if self.monitoring:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3)
            }
            self.samples.append(sample)
            return sample
        return None

def run_benchmark(level, input_path, output_dir="benchmark_outputs"):
    """Run benchmark for a specific performance level."""
    config = PERFORMANCE_LEVELS[level]
    script_path = config['script']
    
    print(f"\n{'='*60}")
    print(f"Running {config['name']}")
    print(f"Script: {script_path}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Warning: Script {script_path} not found. Skipping level {level}.")
        return None
    
    # Prepare command
    cmd = ['uv', 'run', 'python', script_path, input_path]
    
    # Add level-specific arguments
    if level >= 1:  # Conservative and above
        cmd.extend(['--output_summary'])
    
    if level >= 2:  # Moderate and above
        cmd.extend(['--no_batch_processing'])  # Disable for fair comparison
    
    if level >= 3:  # Aggressive
        cmd.extend(['--no_async_processing'])  # Disable for fair comparison
    
    # Initialize monitoring
    monitor = SystemMonitor()
    
    # Get initial system state
    initial_state = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    # Start monitoring
    monitor.start_monitoring()
    start_time = time.perf_counter()
    
    try:
        # Run the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Stop monitoring
        resource_stats = monitor.stop_monitoring()
        
        # Parse output for performance metrics
        output_lines = result.stdout.split('\n')
        
        # Extract key metrics from output
        metrics = {
            'level': level,
            'name': config['name'],
            'script': script_path,
            'execution_time': execution_time,
            'return_code': result.returncode,
            'success': result.returncode == 0,
            'initial_state': initial_state,
            'resource_stats': resource_stats
        }
        
        # Try to extract specific metrics from output
        for line in output_lines:
            if 'filas totales:' in line.lower() or 'total rows:' in line.lower():
                try:
                    # Extract number from line
                    import re
                    numbers = re.findall(r'[\d,]+', line)
                    if numbers:
                        metrics['total_rows'] = int(numbers[-1].replace(',', ''))
                except:
                    pass
            
            elif 'filas/segundo' in line.lower() or 'rows/second' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'[\d,]+', line)
                    if numbers:
                        metrics['rows_per_second'] = int(numbers[-1].replace(',', ''))
                except:
                    pass
            
            elif 'archivos procesados:' in line.lower() or 'files processed:' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        metrics['files_processed'] = int(numbers[-1])
                except:
                    pass
        
        # Calculate derived metrics
        if 'total_rows' in metrics and execution_time > 0:
            metrics['calculated_rows_per_second'] = metrics['total_rows'] / execution_time
        
        # Store stdout/stderr for debugging
        metrics['stdout'] = result.stdout
        metrics['stderr'] = result.stderr
        
        print(f"\nBenchmark Results for {config['name']}:")
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Return code: {result.returncode}")
        print(f"  Success: {metrics['success']}")
        
        if 'total_rows' in metrics:
            print(f"  Total rows: {metrics['total_rows']:,}")
        if 'rows_per_second' in metrics:
            print(f"  Throughput: {metrics['rows_per_second']:,} rows/second")
        
        if resource_stats:
            print(f"  Average CPU: {resource_stats['cpu_mean']:.1f}%")
            print(f"  Peak CPU: {resource_stats['cpu_max']:.1f}%")
            print(f"  Average Memory: {resource_stats['memory_mean']:.1f}%")
            print(f"  Peak Memory: {resource_stats['memory_max']:.1f}%")
        
        if not metrics['success']:
            print(f"  Error output: {result.stderr[:200]}...")
        
        return metrics
    
    except subprocess.TimeoutExpired:
        print(f"Timeout: {config['name']} exceeded 10 minutes")
        monitor.stop_monitoring()
        return {
            'level': level,
            'name': config['name'],
            'script': script_path,
            'execution_time': 600,
            'return_code': -1,
            'success': False,
            'error': 'Timeout',
            'initial_state': initial_state
        }
    
    except Exception as e:
        print(f"Error running {config['name']}: {e}")
        monitor.stop_monitoring()
        return {
            'level': level,
            'name': config['name'],
            'script': script_path,
            'execution_time': -1,
            'return_code': -1,
            'success': False,
            'error': str(e),
            'initial_state': initial_state
        }

def generate_comparison_report(results, output_dir):
    """Generate comprehensive comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful benchmarks to compare.")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for result in successful_results:
        row = {
            'Level': result['level'],
            'Name': result['name'],
            'Execution_Time': result['execution_time'],
            'Total_Rows': result.get('total_rows', 0),
            'Rows_Per_Second': result.get('calculated_rows_per_second', 0),
            'Files_Processed': result.get('files_processed', 0)
        }
        
        if result.get('resource_stats'):
            stats = result['resource_stats']
            row.update({
                'Avg_CPU': stats['cpu_mean'],
                'Peak_CPU': stats['cpu_max'],
                'Avg_Memory': stats['memory_mean'],
                'Peak_Memory': stats['memory_max']
            })
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'benchmark_comparison_{timestamp}.txt')
    
    with open(report_file, 'w') as f:
        f.write("=== Performance Level Comparison Report ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total/(1024**3):.1f} GB RAM\n\n")
        
        # Performance summary
        f.write("Performance Summary:\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Name']}:\n")
            f.write(f"  Execution Time: {row['Execution_Time']:.4f} seconds\n")
            f.write(f"  Throughput: {row['Rows_Per_Second']:,.0f} rows/second\n")
            f.write(f"  Total Rows: {row['Total_Rows']:,}\n")
            if 'Avg_CPU' in row:
                f.write(f"  CPU Usage: {row['Avg_CPU']:.1f}% avg, {row['Peak_CPU']:.1f}% peak\n")
                f.write(f"  Memory Usage: {row['Avg_Memory']:.1f}% avg, {row['Peak_Memory']:.1f}% peak\n")
            f.write("\n")
        
        # Relative performance
        if len(df) > 1:
            baseline = df.iloc[0]  # Assume Level 0 is baseline
            f.write("Relative Performance (vs Level 0):\n")
            f.write("-" * 80 + "\n")
            
            for _, row in df.iterrows():
                if row['Level'] == 0:
                    continue
                
                speedup = baseline['Execution_Time'] / row['Execution_Time']
                throughput_improvement = (row['Rows_Per_Second'] / baseline['Rows_Per_Second'] - 1) * 100
                
                f.write(f"{row['Name']}:\n")
                f.write(f"  Speedup: {speedup:.2f}x faster\n")
                f.write(f"  Throughput improvement: {throughput_improvement:.1f}%\n")
                f.write("\n")
    
    # Save detailed results as JSON
    json_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save comparison CSV
    csv_file = os.path.join(output_dir, f'benchmark_comparison_{timestamp}.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"\nBenchmark reports generated:")
    print(f"  Text report: {report_file}")
    print(f"  JSON data: {json_file}")
    print(f"  CSV data: {csv_file}")
    
    return df

def create_visualizations(df, output_dir):
    """Create performance visualization charts."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Level Comparison', fontsize=16)
        
        # Execution Time
        axes[0, 0].bar(df['Name'], df['Execution_Time'])
        axes[0, 0].set_title('Execution Time (seconds)')
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput
        axes[0, 1].bar(df['Name'], df['Rows_Per_Second'])
        axes[0, 1].set_title('Throughput (rows/second)')
        axes[0, 1].set_ylabel('Rows/Second')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # CPU Usage (if available)
        if 'Avg_CPU' in df.columns:
            axes[1, 0].bar(df['Name'], df['Avg_CPU'])
            axes[1, 0].set_title('Average CPU Usage (%)')
            axes[1, 0].set_ylabel('CPU %')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory Usage (if available)
        if 'Avg_Memory' in df.columns:
            axes[1, 1].bar(df['Name'], df['Avg_Memory'])
            axes[1, 1].set_title('Average Memory Usage (%)')
            axes[1, 1].set_ylabel('Memory %')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_file = os.path.join(output_dir, f'performance_comparison_{timestamp}.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization: {chart_file}")
        
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmarking of all performance levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_path",
        help="Path to test dataset directory"
    )
    
    parser.add_argument(
        "--levels",
        nargs='+',
        type=int,
        default=[0, 1, 2, 3],
        choices=[0, 1, 2, 3],
        help="Performance levels to benchmark"
    )
    
    parser.add_argument(
        "--output_dir",
        default="benchmark_outputs",
        help="Output directory for benchmark results"
    )
    
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Don't clean up temporary files"
    )
    
    args = parser.parse_args()
    
    print("=== Comprehensive Performance Level Benchmarking ===")
    print(f"Input dataset: {args.input_path}")
    print(f"Levels to test: {args.levels}")
    print(f"Output directory: {args.output_dir}")
    
    # Verify input path
    if not os.path.exists(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist.")
        return 1
    
    # System information
    print(f"\nSystem Information:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  Memory: {psutil.virtual_memory().total/(1024**3):.1f} GB")
    print(f"  Python: uv managed environment")
    
    # Run benchmarks
    results = []
    total_start_time = time.time()
    
    for level in sorted(args.levels):
        result = run_benchmark(level, args.input_path, args.output_dir)
        if result:
            results.append(result)
        
        # Brief pause between benchmarks
        time.sleep(2)
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print(f"Benchmarking completed in {total_time:.2f} seconds")
    print(f"Successful benchmarks: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    # Generate comparison report
    if results:
        df = generate_comparison_report(results, args.output_dir)
        if df is not None and len(df) > 1:
            create_visualizations(df, args.output_dir)
    
    print(f"\nAll benchmark results saved to: {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    exit(main())

