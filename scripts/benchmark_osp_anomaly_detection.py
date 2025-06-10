#!/usr/bin/env python3
"""
Benchmarking Framework for OSP Anomaly Detection

This script provides comprehensive benchmarking capabilities for the OSP anomaly detection
implementation (Stage 04). It evaluates performance across different parameter configurations
and provides detailed analysis of throughput, accuracy, and resource utilization.

Key Features:
- Parameter sweep across multiple OSP configurations
- Performance profiling with memory and CPU monitoring
- Comparative analysis across different cell subsets
- Automated report generation with visualizations
- Apple Silicon optimization testing

Author: Jose Miguel Franco
Date: June 2025
"""

import argparse
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import sys

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class OSPBenchmarkRunner:
    """
    Comprehensive benchmarking framework for OSP anomaly detection.
    """
    
    def __init__(self, 
                 data_path: Path,
                 reference_weeks_path: Path,
                 output_dir: Path,
                 script_path: Path = None):
        """
        Initialize benchmark runner.
        
        Args:
            data_path: Path to consolidated telecom data
            reference_weeks_path: Path to reference weeks data
            output_dir: Output directory for benchmark results
            script_path: Path to OSP anomaly detection script
        """
        self.data_path = data_path
        self.reference_weeks_path = reference_weeks_path
        self.output_dir = output_dir
        self.script_path = script_path or Path("scripts/04_anomaly_detection_osp.py")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark results storage
        self.results = []
        
    def run_single_benchmark(self, 
                           config: Dict,
                           run_id: str) -> Dict:
        """
        Run a single benchmark configuration.
        
        Args:
            config: Configuration dictionary with OSP parameters
            run_id: Unique identifier for this run
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nRunning benchmark: {run_id}")
        print(f"Configuration: {config}")
        
        # Create run-specific output directory
        run_output_dir = self.output_dir / f"run_{run_id}"
        run_output_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            "uv", "run", "python", str(self.script_path),
            str(self.data_path),
            str(self.reference_weeks_path),
            "--output_dir", str(run_output_dir),
            "--output_format", "parquet"
        ]
        
        # Add configuration parameters
        for key, value in config.items():
            if key == 'standardize' and not value:
                cmd.append('--no_standardize')
            elif key != 'standardize':
                cmd.extend([f"--{key}", str(value)])
        
        # Monitor system resources
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        
        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            execution_time = time.time() - start_time
            
            # Monitor final resources
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            final_cpu_percent = psutil.cpu_percent(interval=1)
            
            if result.returncode == 0:
                # Parse performance report
                report_files = list(run_output_dir.glob("osp_anomaly_detection_report_*.json"))
                if report_files:
                    with open(report_files[0], 'r') as f:
                        performance_report = json.load(f)
                else:
                    performance_report = {}
                
                # Load summary results
                summary_files = list(run_output_dir.glob("osp_anomaly_summary_*.parquet"))
                if summary_files:
                    summary_df = pd.read_parquet(summary_files[0])
                    
                    summary_stats = {
                        "total_cells": len(summary_df),
                        "avg_anomaly_rate": summary_df['anomaly_rate'].mean(),
                        "std_anomaly_rate": summary_df['anomaly_rate'].std(),
                        "avg_processing_time": summary_df['processing_time'].mean(),
                        "total_samples": summary_df['total_samples'].sum(),
                        "total_anomalies": summary_df['anomalies_detected'].sum()
                    }
                else:
                    summary_stats = {}
                
                return {
                    "run_id": run_id,
                    "config": config,
                    "status": "success",
                    "execution_time": execution_time,
                    "resource_usage": {
                        "memory_increase_mb": final_memory - initial_memory,
                        "initial_cpu_percent": initial_cpu_percent,
                        "final_cpu_percent": final_cpu_percent
                    },
                    "performance_report": performance_report,
                    "summary_stats": summary_stats,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "run_id": run_id,
                    "config": config,
                    "status": "failed",
                    "execution_time": execution_time,
                    "error_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "run_id": run_id,
                "config": config,
                "status": "timeout",
                "execution_time": 3600,
                "error_message": "Execution timed out after 1 hour"
            }
        except Exception as e:
            return {
                "run_id": run_id,
                "config": config,
                "status": "error",
                "execution_time": time.time() - start_time,
                "error_message": str(e)
            }
    
    def generate_parameter_configurations(self, 
                                        config_type: str = "standard") -> List[Dict]:
        """
        Generate parameter configurations for benchmarking.
        
        Args:
            config_type: Type of configuration set ('standard', 'extensive', 'quick')
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        if config_type == "quick":
            # Quick test configurations
            base_config = {
                "max_cells": 10,
                "standardize": True,
                "random_state": 42,
                "max_workers": 4
            }
            
            # Vary key parameters
            for n_components in [2, 3, 5]:
                for anomaly_threshold in [1.5, 2.0, 2.5]:
                    config = base_config.copy()
                    config.update({
                        "n_components": n_components,
                        "anomaly_threshold": anomaly_threshold
                    })
                    configs.append(config)
                    
        elif config_type == "standard":
            # Standard benchmarking configurations
            base_config = {
                "max_cells": 50,
                "standardize": True,
                "random_state": 42
            }
            
            # Parameter sweep
            for n_components in [2, 3, 5, 8, 10]:
                for anomaly_threshold in [1.5, 2.0, 2.5, 3.0]:
                    for max_workers in [1, 4, 8]:
                        config = base_config.copy()
                        config.update({
                            "n_components": n_components,
                            "anomaly_threshold": anomaly_threshold,
                            "max_workers": max_workers
                        })
                        configs.append(config)
                        
        elif config_type == "extensive":
            # Extensive parameter exploration
            base_config = {
                "max_cells": 200,
                "standardize": True,
                "random_state": 42
            }
            
            # Comprehensive parameter sweep
            for n_components in [2, 3, 5, 8, 10, 15, 20]:
                for anomaly_threshold in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                    for max_workers in [1, 2, 4, 8, 12, 16]:
                        for standardize in [True, False]:
                            config = base_config.copy()
                            config.update({
                                "n_components": n_components,
                                "anomaly_threshold": anomaly_threshold,
                                "max_workers": max_workers,
                                "standardize": standardize
                            })
                            configs.append(config)
        
        return configs
    
    def run_benchmark_suite(self, config_type: str = "standard") -> List[Dict]:
        """
        Run complete benchmark suite.
        
        Args:
            config_type: Type of configuration set to run
            
        Returns:
            List of benchmark results
        """
        print(f"\n{'='*80}")
        print(f"OSP ANOMALY DETECTION BENCHMARK SUITE - {config_type.upper()}")
        print(f"{'='*80}")
        
        configs = self.generate_parameter_configurations(config_type)
        print(f"Generated {len(configs)} benchmark configurations")
        
        results = []
        
        for i, config in enumerate(configs):
            run_id = f"{config_type}_{i+1:03d}"
            result = self.run_single_benchmark(config, run_id)
            results.append(result)
            
            # Progress update
            if (i + 1) % 5 == 0 or i == len(configs) - 1:
                successful = sum(1 for r in results if r['status'] == 'success')
                print(f"\nProgress: {i+1}/{len(configs)} benchmarks completed")
                print(f"Success rate: {successful}/{i+1} ({successful/(i+1)*100:.1f}%)")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """
        Analyze benchmark results and generate insights.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            return {"error": "No successful benchmark runs"}
        
        # Extract performance metrics
        performance_data = []
        for result in successful_results:
            config = result['config']
            perf_report = result.get('performance_report', {})
            summary_stats = result.get('summary_stats', {})
            
            perf_metrics = perf_report.get('performance_metrics', {})
            exec_summary = perf_report.get('execution_summary', {})
            
            performance_data.append({
                'run_id': result['run_id'],
                'n_components': config.get('n_components'),
                'anomaly_threshold': config.get('anomaly_threshold'),
                'max_workers': config.get('max_workers'),
                'standardize': config.get('standardize'),
                'execution_time': result['execution_time'],
                'throughput_samples_per_second': perf_metrics.get('throughput_samples_per_second', 0),
                'throughput_cells_per_second': perf_metrics.get('throughput_cells_per_second', 0),
                'success_rate': exec_summary.get('success_rate', 0),
                'avg_anomaly_rate': summary_stats.get('avg_anomaly_rate', 0),
                'total_samples': summary_stats.get('total_samples', 0),
                'memory_increase_mb': result.get('resource_usage', {}).get('memory_increase_mb', 0)
            })
        
        df = pd.DataFrame(performance_data)
        
        # Analysis insights
        analysis = {
            "summary": {
                "total_runs": len(self.results),
                "successful_runs": len(successful_results),
                "success_rate": len(successful_results) / len(self.results),
                "best_throughput_samples": df['throughput_samples_per_second'].max(),
                "best_throughput_cells": df['throughput_cells_per_second'].max(),
                "avg_execution_time": df['execution_time'].mean(),
                "avg_memory_usage": df['memory_increase_mb'].mean()
            },
            "optimal_configurations": {
                "fastest_samples": df.loc[df['throughput_samples_per_second'].idxmax()].to_dict(),
                "fastest_cells": df.loc[df['throughput_cells_per_second'].idxmax()].to_dict(),
                "lowest_memory": df.loc[df['memory_increase_mb'].idxmin()].to_dict()
            },
            "parameter_analysis": {
                "n_components_impact": df.groupby('n_components')['throughput_samples_per_second'].agg(['mean', 'std']).to_dict(),
                "threshold_impact": df.groupby('anomaly_threshold')['avg_anomaly_rate'].agg(['mean', 'std']).to_dict(),
                "workers_impact": df.groupby('max_workers')['throughput_cells_per_second'].agg(['mean', 'std']).to_dict()
            },
            "data_frame": df
        }
        
        return analysis
    
    def generate_visualizations(self, analysis: Dict) -> List[Path]:
        """
        Generate visualization plots for benchmark results.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of paths to generated plots
        """
        df = analysis['data_frame']
        plot_files = []
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Throughput vs Parameters
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OSP Anomaly Detection Performance Analysis', fontsize=16)
        
        # Throughput vs n_components
        sns.boxplot(data=df, x='n_components', y='throughput_samples_per_second', ax=axes[0,0])
        axes[0,0].set_title('Throughput vs SVD Components')
        axes[0,0].set_ylabel('Samples/second')
        
        # Throughput vs workers
        sns.boxplot(data=df, x='max_workers', y='throughput_cells_per_second', ax=axes[0,1])
        axes[0,1].set_title('Throughput vs Parallel Workers')
        axes[0,1].set_ylabel('Cells/second')
        
        # Memory usage vs parameters
        sns.scatterplot(data=df, x='n_components', y='memory_increase_mb', 
                       hue='max_workers', size='total_samples', ax=axes[1,0])
        axes[1,0].set_title('Memory Usage vs Components')
        axes[1,0].set_ylabel('Memory Increase (MB)')
        
        # Anomaly rate vs threshold
        sns.boxplot(data=df, x='anomaly_threshold', y='avg_anomaly_rate', ax=axes[1,1])
        axes[1,1].set_title('Anomaly Rate vs Threshold')
        axes[1,1].set_ylabel('Average Anomaly Rate')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'performance_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # 2. Execution Time Analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Execution time vs configuration complexity
        df['config_complexity'] = df['n_components'] * df['max_workers']
        sns.scatterplot(data=df, x='config_complexity', y='execution_time', 
                       hue='anomaly_threshold', size='total_samples')
        plt.title('Execution Time vs Configuration Complexity')
        plt.xlabel('Configuration Complexity (components × workers)')
        plt.ylabel('Execution Time (seconds)')
        
        plot_file = self.output_dir / 'execution_time_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
        
        # 3. Parameter Heatmap
        if len(df) > 10:  # Only create heatmap if we have enough data
            pivot_data = df.groupby(['n_components', 'anomaly_threshold'])['throughput_samples_per_second'].mean().unstack()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='viridis')
            plt.title('Throughput Heatmap: Components vs Threshold')
            plt.xlabel('Anomaly Threshold')
            plt.ylabel('SVD Components')
            
            plot_file = self.output_dir / 'parameter_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file)
        
        return plot_files
    
    def save_comprehensive_report(self, analysis: Dict) -> Path:
        """
        Save comprehensive benchmark report.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"osp_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            analysis_copy = analysis.copy()
            if 'data_frame' in analysis_copy:
                analysis_copy['data_frame'] = analysis_copy['data_frame'].to_dict('records')
            
            json.dump({
                "benchmark_metadata": {
                    "timestamp": timestamp,
                    "data_path": str(self.data_path),
                    "reference_weeks_path": str(self.reference_weeks_path),
                    "script_path": str(self.script_path)
                },
                "results": self.results,
                "analysis": analysis_copy
            }, f, indent=2, default=str)
        
        # Save performance DataFrame as CSV
        if 'data_frame' in analysis:
            csv_file = self.output_dir / f"osp_benchmark_performance_{timestamp}.csv"
            analysis['data_frame'].to_csv(csv_file, index=False)
        
        # Generate markdown report
        md_file = self.output_dir / f"osp_benchmark_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# OSP Anomaly Detection Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section
            summary = analysis['summary']
            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Total benchmark runs:** {summary['total_runs']}\n")
            f.write(f"- **Successful runs:** {summary['successful_runs']} ({summary['success_rate']*100:.1f}%)\n")
            f.write(f"- **Best sample throughput:** {summary['best_throughput_samples']:,.0f} samples/sec\n")
            f.write(f"- **Best cell throughput:** {summary['best_throughput_cells']:.1f} cells/sec\n")
            f.write(f"- **Average execution time:** {summary['avg_execution_time']:.1f} seconds\n")
            f.write(f"- **Average memory usage:** {summary['avg_memory_usage']:.1f} MB\n\n")
            
            # Optimal configurations
            f.write(f"## Optimal Configurations\n\n")
            optimal = analysis['optimal_configurations']
            
            f.write(f"### Fastest Sample Processing\n")
            fastest_samples = optimal['fastest_samples']
            f.write(f"- **Components:** {fastest_samples['n_components']}\n")
            f.write(f"- **Threshold:** {fastest_samples['anomaly_threshold']}\n")
            f.write(f"- **Workers:** {fastest_samples['max_workers']}\n")
            f.write(f"- **Throughput:** {fastest_samples['throughput_samples_per_second']:,.0f} samples/sec\n\n")
            
            f.write(f"### Fastest Cell Processing\n")
            fastest_cells = optimal['fastest_cells']
            f.write(f"- **Components:** {fastest_cells['n_components']}\n")
            f.write(f"- **Threshold:** {fastest_cells['anomaly_threshold']}\n")
            f.write(f"- **Workers:** {fastest_cells['max_workers']}\n")
            f.write(f"- **Throughput:** {fastest_cells['throughput_cells_per_second']:.1f} cells/sec\n\n")
            
            # Parameter analysis
            f.write(f"## Parameter Impact Analysis\n\n")
            param_analysis = analysis['parameter_analysis']
            
            f.write(f"### SVD Components Impact\n")
            for components, stats in param_analysis['n_components_impact']['mean'].items():
                std = param_analysis['n_components_impact']['std'][components]
                f.write(f"- **{components} components:** {stats:,.0f} ± {std:,.0f} samples/sec\n")
            
            f.write(f"\n### Anomaly Threshold Impact\n")
            for threshold, stats in param_analysis['threshold_impact']['mean'].items():
                std = param_analysis['threshold_impact']['std'][threshold]
                f.write(f"- **Threshold {threshold}:** {stats:.3f} ± {std:.3f} anomaly rate\n")
        
        print(f"\nComprehensive report saved to: {md_file}")
        return md_file

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive OSP Anomaly Detection Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark (small parameter set)
  uv run python scripts/benchmark_osp_anomaly_detection.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --config_type quick
  
  # Standard benchmark suite
  uv run python scripts/benchmark_osp_anomaly_detection.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --config_type standard
  
  # Extensive benchmarking (long-running)
  uv run python scripts/benchmark_osp_anomaly_detection.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --config_type extensive --output_dir extensive_benchmarks/
        """
    )
    
    # Required arguments
    parser.add_argument('data_path', type=str,
                       help='Path to consolidated telecom data (Parquet format)')
    parser.add_argument('reference_weeks_path', type=str,
                       help='Path to reference weeks data (Parquet format)')
    
    # Benchmark configuration
    parser.add_argument('--config_type', choices=['quick', 'standard', 'extensive'], 
                       default='standard',
                       help='Type of benchmark configuration set (default: standard)')
    parser.add_argument('--output_dir', type=str, default='reports/benchmarks',
                       help='Output directory for benchmark results (default: reports/benchmarks)')
    parser.add_argument('--script_path', type=str, default='scripts/04_anomaly_detection_osp.py',
                       help='Path to OSP anomaly detection script (default: scripts/04_anomaly_detection_osp.py)')
    
    # Analysis options
    parser.add_argument('--skip_visualizations', action='store_true',
                       help='Skip generation of visualization plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate paths
    data_path = Path(args.data_path)
    reference_weeks_path = Path(args.reference_weeks_path)
    output_dir = Path(args.output_dir)
    script_path = Path(args.script_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not reference_weeks_path.exists():
        raise FileNotFoundError(f"Reference weeks file not found: {reference_weeks_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"OSP script not found: {script_path}")
    
    print(f"\n{'='*80}")
    print(f"OSP ANOMALY DETECTION COMPREHENSIVE BENCHMARK")
    print(f"{'='*80}")
    print(f"Data file: {data_path}")
    print(f"Reference weeks: {reference_weeks_path}")
    print(f"Output directory: {output_dir}")
    print(f"Configuration type: {args.config_type}")
    print(f"Script path: {script_path}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Initialize benchmark runner
        runner = OSPBenchmarkRunner(
            data_path=data_path,
            reference_weeks_path=reference_weeks_path,
            output_dir=output_dir,
            script_path=script_path
        )
        
        # Run benchmark suite
        print(f"\nStarting {args.config_type} benchmark suite...")
        results = runner.run_benchmark_suite(args.config_type)
        
        # Analyze results
        print(f"\nAnalyzing benchmark results...")
        analysis = runner.analyze_results()
        
        if 'error' in analysis:
            print(f"Error in analysis: {analysis['error']}")
            return
        
        # Generate visualizations
        if not args.skip_visualizations:
            print(f"\nGenerating visualizations...")
            plot_files = runner.generate_visualizations(analysis)
            print(f"Generated {len(plot_files)} visualization plots")
        
        # Save comprehensive report
        print(f"\nSaving comprehensive report...")
        report_file = runner.save_comprehensive_report(analysis)
        
        # Final summary
        total_time = time.time() - start_time
        summary = analysis['summary']
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETED")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Benchmark runs: {summary['successful_runs']}/{summary['total_runs']} successful")
        print(f"Best sample throughput: {summary['best_throughput_samples']:,.0f} samples/sec")
        print(f"Best cell throughput: {summary['best_throughput_cells']:.1f} cells/sec")
        print(f"Report saved to: {report_file}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

