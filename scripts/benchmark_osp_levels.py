#!/usr/bin/env python3
"""
Benchmarking Framework for OSP Anomaly Detection Performance Levels

This script provides comprehensive benchmarking across different OSP optimization
levels (0-2) to evaluate the impact of various optimization strategies.

Performance Levels:
- Level 0: Baseline implementation
- Level 1: Conservative optimizations (memory-focused)
- Level 2: Moderate optimizations (computational-focused)

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

class OSPLevelBenchmarkRunner:
    """
    Comprehensive benchmarking framework for OSP optimization levels.
    """
    
    def __init__(self, 
                 data_path: Path,
                 reference_weeks_path: Path,
                 output_dir: Path):
        """
        Initialize benchmark runner.
        
        Args:
            data_path: Path to consolidated telecom data
            reference_weeks_path: Path to reference weeks data
            output_dir: Output directory for benchmark results
        """
        self.data_path = data_path
        self.reference_weeks_path = reference_weeks_path
        self.output_dir = output_dir
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark results storage
        self.results = []
        
        # Level configurations
        self.levels = {
            0: {
                "script": "scripts/04_anomaly_detection_osp.py",
                "name": "Level 0 (Baseline)",
                "description": "Original implementation",
                "optimizations": []
            },
            1: {
                "script": "scripts/04_anomaly_detection_osp_optimized.py",
                "name": "Level 1 (Conservative)", 
                "description": "Memory-focused optimizations",
                "optimizations": [
                    "Reduced precision (float32)",
                    "Memory-efficient streaming",
                    "Batch processing",
                    "Adaptive workers",
                    "Resource monitoring"
                ]
            },
            2: {
                "script": "scripts/04_anomaly_detection_osp_moderate.py",
                "name": "Level 2 (Moderate)",
                "description": "Computational optimizations",
                "optimizations": [
                    "Fast randomized SVD",
                    "Vectorized operations",
                    "Contiguous arrays",
                    "Optimized linear algebra",
                    "High-precision timers",
                    "imap_unordered processing"
                ]
            }
        }
        
    def run_single_level_benchmark(self, 
                                  level: int,
                                  test_config: Dict) -> Dict:
        """
        Run benchmark for a single optimization level.
        
        Args:
            level: Optimization level (0-2)
            test_config: Test configuration parameters
            
        Returns:
            Dictionary with benchmark results
        """
        if level not in self.levels:
            raise ValueError(f"Invalid level {level}. Must be 0, 1, or 2.")
            
        level_info = self.levels[level]
        script_path = Path(level_info["script"])
        
        if not script_path.exists():
            return {
                "level": level,
                "status": "script_not_found",
                "error_message": f"Script not found: {script_path}"
            }
        
        print(f"\nRunning {level_info['name']} benchmark...")
        print(f"Script: {script_path}")
        print(f"Configuration: {test_config}")
        
        # Create level-specific output directory
        level_output_dir = self.output_dir / f"level_{level}"
        level_output_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            "uv", "run", "python", str(script_path),
            str(self.data_path),
            str(self.reference_weeks_path),
            "--output_dir", str(level_output_dir),
            "--output_format", "parquet"
        ]
        
        # Add test configuration parameters
        for key, value in test_config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        # Monitor system resources
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        
        start_time = time.perf_counter()
        
        try:
            # Run the benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            execution_time = time.perf_counter() - start_time
            
            # Monitor final resources
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            final_cpu_percent = psutil.cpu_percent(interval=1)
            
            if result.returncode == 0:
                # Extract performance metrics from output
                output_lines = result.stdout.split('\n')
                
                # Parse key metrics from output
                metrics = self._parse_performance_metrics(output_lines)
                
                return {
                    "level": level,
                    "level_info": level_info,
                    "status": "success",
                    "execution_time": execution_time,
                    "test_config": test_config,
                    "metrics": metrics,
                    "resource_usage": {
                        "memory_increase_mb": final_memory - initial_memory,
                        "initial_cpu_percent": initial_cpu_percent,
                        "final_cpu_percent": final_cpu_percent
                    },
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "level": level,
                    "level_info": level_info,
                    "status": "failed",
                    "execution_time": execution_time,
                    "test_config": test_config,
                    "error_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "level": level,
                "level_info": level_info,
                "status": "timeout",
                "execution_time": 1800,
                "test_config": test_config,
                "error_message": "Execution timed out after 30 minutes"
            }
        except Exception as e:
            return {
                "level": level,
                "level_info": level_info,
                "status": "error",
                "execution_time": time.perf_counter() - start_time,
                "test_config": test_config,
                "error_message": str(e)
            }
    
    def _parse_performance_metrics(self, output_lines: List[str]) -> Dict:
        """
        Parse performance metrics from script output.
        
        Args:
            output_lines: Lines from script stdout
            
        Returns:
            Dictionary with parsed metrics
        """
        metrics = {}
        
        for line in output_lines:
            # Parse execution time
            if "Total execution time:" in line:
                try:
                    time_str = line.split(":")
                    if len(time_str) > 1:
                        time_value = float(time_str[-1].split()[0])
                        metrics["execution_time_parsed"] = time_value
                except:
                    pass
            
            # Parse throughput
            if "Throughput:" in line and "samples/sec" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "samples/sec" in part:
                            throughput_str = parts[i-1].replace(",", "")
                            metrics["throughput_samples_per_second"] = float(throughput_str)
                            break
                except:
                    pass
            
            # Parse cells processed
            if "Cells processed:" in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if "/" in part and part.replace("/", "").replace(",", "").isdigit():
                            successful, total = part.split("/")
                            metrics["cells_successful"] = int(successful.replace(",", ""))
                            metrics["cells_total"] = int(total.replace(",", ""))
                            break
                except:
                    pass
            
            # Parse samples processed
            if "Samples processed:" in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if part.replace(",", "").isdigit():
                            metrics["samples_processed"] = int(part.replace(",", ""))
                            break
                except:
                    pass
            
            # Parse memory usage
            if "Memory usage:" in line and "MB" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "MB" in part:
                            memory_str = parts[i-1].replace(",", "")
                            metrics["memory_increase_mb"] = float(memory_str)
                            break
                except:
                    pass
        
        return metrics
    
    def run_level_comparison_benchmark(self, 
                                     levels: List[int] = [0, 1, 2],
                                     test_configs: List[Dict] = None) -> List[Dict]:
        """
        Run comparative benchmark across multiple levels.
        
        Args:
            levels: List of optimization levels to test
            test_configs: List of test configurations
            
        Returns:
            List of benchmark results
        """
        if test_configs is None:
            test_configs = [
                {
                    "max_cells": 50,
                    "n_components": 3,
                    "anomaly_threshold": 2.0,
                    "max_workers": 8
                },
                {
                    "max_cells": 100,
                    "n_components": 3,
                    "anomaly_threshold": 2.0,
                    "max_workers": 8
                }
            ]
        
        print(f"\n{'='*80}")
        print(f"OSP OPTIMIZATION LEVELS BENCHMARK")
        print(f"{'='*80}")
        print(f"Levels to test: {levels}")
        print(f"Test configurations: {len(test_configs)}")
        print(f"Total benchmarks: {len(levels) * len(test_configs)}")
        print(f"{'='*80}")
        
        results = []
        
        for config_idx, test_config in enumerate(test_configs):
            print(f"\n🔧 Test Configuration {config_idx + 1}/{len(test_configs)}: {test_config}")
            
            for level in levels:
                result = self.run_single_level_benchmark(level, test_config)
                result["config_index"] = config_idx
                results.append(result)
                
                # Brief status update
                if result["status"] == "success":
                    metrics = result.get("metrics", {})
                    throughput = metrics.get("throughput_samples_per_second", 0)
                    exec_time = metrics.get("execution_time_parsed", result["execution_time"])
                    print(f"  ✅ {result['level_info']['name']}: {exec_time:.2f}s, {throughput:,.0f} samples/sec")
                else:
                    print(f"  ❌ {result['level_info']['name']}: {result['status']}")
        
        self.results = results
        return results
    
    def analyze_benchmark_results(self) -> Dict:
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
        
        # Group results by configuration
        config_groups = {}
        for result in successful_results:
            config_idx = result.get('config_index', 0)
            if config_idx not in config_groups:
                config_groups[config_idx] = []
            config_groups[config_idx].append(result)
        
        analysis = {
            "summary": {
                "total_runs": len(self.results),
                "successful_runs": len(successful_results),
                "success_rate": len(successful_results) / len(self.results),
                "configurations_tested": len(config_groups)
            },
            "level_comparison": {},
            "optimization_impact": {},
            "recommendations": []
        }
        
        # Analyze each configuration
        for config_idx, config_results in config_groups.items():
            config_analysis = self._analyze_config_results(config_results)
            analysis["level_comparison"][f"config_{config_idx}"] = config_analysis
        
        # Generate overall recommendations
        analysis["recommendations"] = self._generate_recommendations(config_groups)
        
        return analysis
    
    def _analyze_config_results(self, config_results: List[Dict]) -> Dict:
        """
        Analyze results for a single configuration.
        
        Args:
            config_results: Results for one test configuration
            
        Returns:
            Analysis dictionary
        """
        if not config_results:
            return {}
        
        # Sort by level
        config_results.sort(key=lambda x: x['level'])
        
        # Extract metrics
        performance_data = []
        for result in config_results:
            metrics = result.get('metrics', {})
            resource_usage = result.get('resource_usage', {})
            
            performance_data.append({
                'level': result['level'],
                'level_name': result['level_info']['name'],
                'execution_time': metrics.get('execution_time_parsed', result['execution_time']),
                'throughput': metrics.get('throughput_samples_per_second', 0),
                'memory_increase': metrics.get('memory_increase_mb', resource_usage.get('memory_increase_mb', 0)),
                'samples_processed': metrics.get('samples_processed', 0),
                'cells_processed': metrics.get('cells_successful', 0)
            })
        
        df = pd.DataFrame(performance_data)
        
        if len(df) == 0:
            return {"error": "No valid performance data"}
        
        # Calculate improvements relative to Level 0
        baseline = df[df['level'] == 0].iloc[0] if len(df[df['level'] == 0]) > 0 else df.iloc[0]
        
        improvements = {}
        for _, row in df.iterrows():
            level_name = row['level_name']
            if row['level'] != baseline['level']:
                time_improvement = (baseline['execution_time'] - row['execution_time']) / baseline['execution_time'] * 100
                throughput_improvement = (row['throughput'] - baseline['throughput']) / baseline['throughput'] * 100
                
                improvements[level_name] = {
                    'time_improvement_percent': time_improvement,
                    'throughput_improvement_percent': throughput_improvement,
                    'is_faster': time_improvement > 0,
                    'is_more_efficient': throughput_improvement > 0
                }
        
        return {
            'performance_data': performance_data,
            'baseline': baseline.to_dict(),
            'improvements': improvements,
            'best_time': df.loc[df['execution_time'].idxmin()].to_dict(),
            'best_throughput': df.loc[df['throughput'].idxmax()].to_dict(),
            'test_config': config_results[0]['test_config']
        }
    
    def _generate_recommendations(self, config_groups: Dict) -> List[str]:
        """
        Generate recommendations based on benchmark results.
        
        Args:
            config_groups: Grouped configuration results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze across all configurations
        all_improvements = {}
        for config_results in config_groups.values():
            analysis = self._analyze_config_results(config_results)
            improvements = analysis.get('improvements', {})
            
            for level_name, improvement in improvements.items():
                if level_name not in all_improvements:
                    all_improvements[level_name] = []
                all_improvements[level_name].append(improvement)
        
        # Generate recommendations based on consistent patterns
        for level_name, improvement_list in all_improvements.items():
            avg_time_improvement = np.mean([imp['time_improvement_percent'] for imp in improvement_list])
            avg_throughput_improvement = np.mean([imp['throughput_improvement_percent'] for imp in improvement_list])
            
            if avg_time_improvement > 20 and avg_throughput_improvement > 20:
                recommendations.append(f"✅ {level_name}: Highly recommended - significant performance gains")
            elif avg_time_improvement > 10 or avg_throughput_improvement > 10:
                recommendations.append(f"✨ {level_name}: Recommended - moderate performance gains")
            elif avg_time_improvement < -10 or avg_throughput_improvement < -10:
                recommendations.append(f"❌ {level_name}: Not recommended - performance regression")
            else:
                recommendations.append(f"⚠️ {level_name}: Neutral - minimal impact")
        
        return recommendations
    
    def generate_visualizations(self, analysis: Dict) -> List[Path]:
        """
        Generate visualization plots for benchmark results.
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            List of paths to generated plots
        """
        plot_files = []
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Prepare data for plotting
        plot_data = []
        for config_name, config_analysis in analysis.get('level_comparison', {}).items():
            performance_data = config_analysis.get('performance_data', [])
            for perf in performance_data:
                plot_data.append({
                    'configuration': config_name,
                    'level': perf['level'],
                    'level_name': perf['level_name'],
                    'execution_time': perf['execution_time'],
                    'throughput': perf['throughput'],
                    'memory_increase': perf['memory_increase']
                })
        
        if not plot_data:
            return plot_files
        
        plot_df = pd.DataFrame(plot_data)
        
        # 1. Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OSP Optimization Levels Performance Comparison', fontsize=16)
        
        # Execution time comparison
        sns.barplot(data=plot_df, x='level_name', y='execution_time', ax=axes[0,0])
        axes[0,0].set_title('Execution Time by Optimization Level')
        axes[0,0].set_ylabel('Time (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        sns.barplot(data=plot_df, x='level_name', y='throughput', ax=axes[0,1])
        axes[0,1].set_title('Throughput by Optimization Level')
        axes[0,1].set_ylabel('Samples/second')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        sns.barplot(data=plot_df, x='level_name', y='memory_increase', ax=axes[1,0])
        axes[1,0].set_title('Memory Usage by Optimization Level')
        axes[1,0].set_ylabel('Memory Increase (MB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Performance vs Memory trade-off
        sns.scatterplot(data=plot_df, x='memory_increase', y='throughput', 
                       hue='level_name', size='execution_time', ax=axes[1,1])
        axes[1,1].set_title('Performance vs Memory Trade-off')
        axes[1,1].set_xlabel('Memory Increase (MB)')
        axes[1,1].set_ylabel('Throughput (samples/sec)')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'osp_levels_comparison.png'
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
        results_file = self.output_dir / f"osp_levels_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "benchmark_metadata": {
                    "timestamp": timestamp,
                    "data_path": str(self.data_path),
                    "reference_weeks_path": str(self.reference_weeks_path),
                    "levels_tested": list(self.levels.keys())
                },
                "results": self.results,
                "analysis": analysis
            }, f, indent=2, default=str)
        
        # Generate markdown report
        md_file = self.output_dir / f"osp_levels_benchmark_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# OSP Optimization Levels Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section
            summary = analysis['summary']
            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Total benchmark runs:** {summary['total_runs']}\n")
            f.write(f"- **Successful runs:** {summary['successful_runs']} ({summary['success_rate']*100:.1f}%)\n")
            f.write(f"- **Configurations tested:** {summary['configurations_tested']}\n\n")
            
            # Optimization levels overview
            f.write(f"## Optimization Levels\n\n")
            for level, info in self.levels.items():
                f.write(f"### {info['name']}\n")
                f.write(f"- **Description:** {info['description']}\n")
                f.write(f"- **Script:** `{info['script']}`\n")
                if info['optimizations']:
                    f.write(f"- **Optimizations:**\n")
                    for opt in info['optimizations']:
                        f.write(f"  - {opt}\n")
                f.write(f"\n")
            
            # Recommendations
            f.write(f"## Recommendations\n\n")
            recommendations = analysis.get('recommendations', [])
            for rec in recommendations:
                f.write(f"- {rec}\n")
            f.write(f"\n")
            
            # Performance comparison
            f.write(f"## Performance Comparison\n\n")
            for config_name, config_analysis in analysis.get('level_comparison', {}).items():
                test_config = config_analysis.get('test_config', {})
                f.write(f"### {config_name.replace('_', ' ').title()}\n")
                f.write(f"**Configuration:** {test_config}\n\n")
                
                performance_data = config_analysis.get('performance_data', [])
                if performance_data:
                    f.write(f"| Level | Execution Time | Throughput | Memory Usage |\n")
                    f.write(f"|-------|----------------|------------|--------------|\n")
                    for perf in performance_data:
                        f.write(f"| {perf['level_name']} | {perf['execution_time']:.2f}s | {perf['throughput']:,.0f} samples/sec | {perf['memory_increase']:.1f} MB |\n")
                    f.write(f"\n")
        
        print(f"\nComprehensive report saved to: {md_file}")
        return md_file

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive OSP Optimization Levels Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark of all levels
  uv run python scripts/benchmark_osp_levels.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet
  
  # Test specific levels
  uv run python scripts/benchmark_osp_levels.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --levels 0 2
  
  # Custom test configurations
  uv run python scripts/benchmark_osp_levels.py data/processed/consolidated_milan_telecom_merged.parquet reports/reference_weeks_20250607_135521.parquet --max_cells 200 --output_dir level_benchmarks/
        """
    )
    
    # Required arguments
    parser.add_argument('data_path', type=str,
                       help='Path to consolidated telecom data (Parquet format)')
    parser.add_argument('reference_weeks_path', type=str,
                       help='Path to reference weeks data (Parquet format)')
    
    # Benchmark configuration
    parser.add_argument('--levels', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2],
                       help='Optimization levels to test (default: 0 1 2)')
    parser.add_argument('--output_dir', type=str, default='reports/level_benchmarks',
                       help='Output directory for benchmark results (default: reports/level_benchmarks)')
    
    # Test parameters
    parser.add_argument('--max_cells', type=int, default=100,
                       help='Maximum number of cells to test (default: 100)')
    parser.add_argument('--n_components', type=int, default=3,
                       help='Number of SVD components (default: 3)')
    parser.add_argument('--anomaly_threshold', type=float, default=2.0,
                       help='Anomaly threshold (default: 2.0)')
    parser.add_argument('--max_workers', type=int, default=8,
                       help='Maximum workers (default: 8)')
    
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
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not reference_weeks_path.exists():
        raise FileNotFoundError(f"Reference weeks file not found: {reference_weeks_path}")
    
    print(f"\n{'='*80}")
    print(f"OSP OPTIMIZATION LEVELS COMPREHENSIVE BENCHMARK")
    print(f"{'='*80}")
    print(f"Data file: {data_path}")
    print(f"Reference weeks: {reference_weeks_path}")
    print(f"Output directory: {output_dir}")
    print(f"Levels to test: {args.levels}")
    print(f"Max cells: {args.max_cells}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Initialize benchmark runner
        runner = OSPLevelBenchmarkRunner(
            data_path=data_path,
            reference_weeks_path=reference_weeks_path,
            output_dir=output_dir
        )
        
        # Define test configurations
        test_configs = [
            {
                "max_cells": args.max_cells,
                "n_components": args.n_components,
                "anomaly_threshold": args.anomaly_threshold,
                "max_workers": args.max_workers
            }
        ]
        
        # Run benchmark suite
        print(f"\nStarting optimization levels benchmark...")
        results = runner.run_level_comparison_benchmark(
            levels=args.levels,
            test_configs=test_configs
        )
        
        # Analyze results
        print(f"\nAnalyzing benchmark results...")
        analysis = runner.analyze_benchmark_results()
        
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
        print(f"\nRecommendations:")
        for rec in analysis.get('recommendations', []):
            print(f"  {rec}")
        print(f"\nReport saved to: {report_file}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

