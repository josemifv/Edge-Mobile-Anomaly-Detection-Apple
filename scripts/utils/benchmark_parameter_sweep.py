#!/usr/bin/env python3
"""
benchmark_parameter_sweep.py

CMMSE 2025: Comprehensive Parameter Sweep Benchmark
Systematic testing of configurable parameters across all pipeline stages

This script performs automated parameter sweeps without modifying existing code,
running multiple configurations and collecting performance metrics.

Usage:
    python scripts/benchmark_parameter_sweep.py [--mode quick|standard|extensive]

Example:
    python scripts/benchmark_parameter_sweep.py --mode standard --output_dir results/benchmarks/
"""

import subprocess
import json
import csv
import time
import argparse
import psutil
import os
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd

class ParameterSweepBenchmark:
    """Comprehensive parameter sweep benchmark for CMMSE 2025 pipeline."""
    
    def __init__(self, output_dir: str = "results/benchmarks/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directories
        self.data_dir = Path("results/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results storage
        self.results = []
        
    def get_parameter_configurations(self, mode: str = "standard"):
        """Define parameter configurations for different benchmark modes."""
        
        configs = {
            "quick": {
                "max_workers": [4, 8, 10],
                "n_components": [2, 3],
                "anomaly_threshold": [1.5, 2.0],
                "num_weeks": [3, 4],
                "mad_threshold": [1.0, 1.5]
            },
            "standard": {
                "max_workers": [4, 6, 8, 10, 12],
                "n_components": [2, 3, 4, 5],
                "anomaly_threshold": [1.5, 2.0, 2.5],
                "num_weeks": [3, 4, 5],
                "mad_threshold": [1.0, 1.5, 2.0]
            },
            "extensive": {
                "max_workers": [2, 4, 6, 8, 10, 12, 14],
                "n_components": [2, 3, 4, 5, 6],
                "anomaly_threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
                "num_weeks": [2, 3, 4, 5, 6],
                "mad_threshold": [0.5, 1.0, 1.5, 2.0, 2.5]
            }
        }
        
        return configs.get(mode, configs["standard"])
    
    def run_single_configuration(self, config_id: int, params: dict):
        """Run pipeline with a specific parameter configuration."""
        print(f"\n{'='*60}")
        print(f"Configuration {config_id}: {params}")
        print('='*60)
        
        # Create unique output directory for this configuration
        config_output = self.data_dir / f"config_{config_id:03d}_{self.timestamp}"
        config_output.mkdir(exist_ok=True)
        
        # Start timing and resource monitoring
        start_time = time.perf_counter()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Stage 1: Data Ingestion
            stage1_start = time.perf_counter()
            cmd1 = [
                "uv", "run", "scripts/01_data_ingestion.py",
                "data/raw/",
                "--output_path", str(config_output / "ingested_data.parquet"),
                "--max_workers", str(params["max_workers"])
            ]
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
            stage1_time = time.perf_counter() - stage1_start
            
            # Stage 2: Data Preprocessing
            stage2_start = time.perf_counter()
            cmd2 = [
                "uv", "run", "scripts/02_data_preprocessing.py",
                str(config_output / "ingested_data.parquet"),
                "--output_path", str(config_output / "preprocessed_data.parquet")
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
            stage2_time = time.perf_counter() - stage2_start
            
            # Stage 3: Week Selection
            stage3_start = time.perf_counter()
            cmd3 = [
                "uv", "run", "scripts/03_week_selection.py",
                str(config_output / "preprocessed_data.parquet"),
                "--output_path", str(config_output / "reference_weeks.parquet"),
                "--num_weeks", str(params["num_weeks"]),
                "--mad_threshold", str(params["mad_threshold"])
            ]
            
            result3 = subprocess.run(cmd3, capture_output=True, text=True, check=True)
            stage3_time = time.perf_counter() - stage3_start
            
            # Stage 4: Individual Anomaly Detection
            stage4_start = time.perf_counter()
            cmd4 = [
                "uv", "run", "scripts/04_anomaly_detection_individual.py",
                str(config_output / "preprocessed_data.parquet"),
                str(config_output / "reference_weeks.parquet"),
                "--output_path", str(config_output / "individual_anomalies.parquet"),
                "--n_components", str(params["n_components"]),
                "--anomaly_threshold", str(params["anomaly_threshold"]),
                "--max_workers", str(params["max_workers"])
            ]
            
            result4 = subprocess.run(cmd4, capture_output=True, text=True, check=True)
            stage4_time = time.perf_counter() - stage4_start
            
            # Calculate total time and memory usage
            total_time = time.perf_counter() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            # Extract performance metrics from outputs
            metrics = self.extract_metrics_from_outputs(
                result1.stdout, result2.stdout, result3.stdout, result4.stdout
            )
            
            # Calculate file sizes
            file_sizes = {
                "ingested_mb": self.get_file_size_mb(config_output / "ingested_data.parquet"),
                "preprocessed_mb": self.get_file_size_mb(config_output / "preprocessed_data.parquet"),
                "reference_weeks_mb": self.get_file_size_mb(config_output / "reference_weeks.parquet"),
                "individual_anomalies_mb": self.get_file_size_mb(config_output / "individual_anomalies.parquet")
            }
            
            # Compile results
            result = {
                "config_id": config_id,
                "timestamp": self.timestamp,
                "status": "success",
                "parameters": params,
                "timing": {
                    "stage1_time": stage1_time,
                    "stage2_time": stage2_time,
                    "stage3_time": stage3_time,
                    "stage4_time": stage4_time,
                    "total_time": total_time
                },
                "memory": {
                    "initial_mb": initial_memory,
                    "final_mb": final_memory,
                    "delta_mb": memory_delta
                },
                "file_sizes": file_sizes,
                "metrics": metrics,
                "output_dir": str(config_output)
            }
            
            print(f"✅ Configuration {config_id} completed successfully")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Memory delta: {memory_delta:.1f}MB")
            
            return result
            
        except subprocess.CalledProcessError as e:
            error_time = time.perf_counter() - start_time
            error_result = {
                "config_id": config_id,
                "timestamp": self.timestamp,
                "status": "failed",
                "parameters": params,
                "error": str(e),
                "partial_time": error_time,
                "output_dir": str(config_output)
            }
            
            print(f"❌ Configuration {config_id} failed: {e}")
            return error_result
    
    def extract_metrics_from_outputs(self, out1: str, out2: str, out3: str, out4: str):
        """Extract performance metrics from script outputs."""
        metrics = {}
        
        try:
            # Extract from Stage 1 output
            for line in out1.split('\n'):
                if "Total rows:" in line:
                    metrics["total_rows"] = int(line.split(':')[1].strip().replace(',', ''))
                elif "Processing rate:" in line:
                    rate_str = line.split(':')[1].strip().split()[0].replace(',', '')
                    metrics["stage1_rate"] = int(rate_str)
                elif "Files processed:" in line:
                    files_info = line.split(':')[1].strip()
                    metrics["files_processed"] = files_info
            
            # Extract from Stage 2 output
            for line in out2.split('\n'):
                if "Output rows:" in line:
                    metrics["output_rows"] = int(line.split(':')[1].strip().replace(',', ''))
                elif "Compression:" in line:
                    comp_str = line.split(':')[1].strip().rstrip('%')
                    metrics["compression_pct"] = float(comp_str)
            
            # Extract from Stage 3 output
            for line in out3.split('\n'):
                if "Reference weeks selected:" in line:
                    metrics["reference_weeks"] = int(line.split(':')[1].strip().replace(',', ''))
                elif "Cells with reference weeks:" in line:
                    metrics["cells_with_refs"] = int(line.split(':')[1].strip().replace(',', ''))
            
            # Extract from Stage 4 output
            for line in out4.split('\n'):
                if "Total anomalies detected:" in line:
                    metrics["anomalies_detected"] = int(line.split(':')[1].strip().replace(',', ''))
                elif "Overall anomaly rate:" in line:
                    rate_str = line.split(':')[1].strip().rstrip('%')
                    metrics["anomaly_rate"] = float(rate_str)
                elif "Successful cells:" in line:
                    metrics["successful_cells"] = int(line.split(':')[1].strip())
                elif "Processing rate:" in line and "samples/second" in line:
                    rate_str = line.split(':')[1].strip().split()[0].replace(',', '')
                    metrics["stage4_rate"] = int(rate_str)
        
        except Exception as e:
            print(f"Warning: Error extracting metrics: {e}")
        
        return metrics
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / 1024 / 1024
        except (FileNotFoundError, OSError):
            return 0.0
    
    def run_parameter_sweep(self, mode: str = "standard"):
        """Run complete parameter sweep benchmark."""
        print(f"🚀 Starting CMMSE 2025 Parameter Sweep Benchmark")
        print(f"Mode: {mode.upper()}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Output directory: {self.output_dir}")
        
        # Get parameter configurations
        param_space = self.get_parameter_configurations(mode)
        
        # Generate all parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        combinations = list(product(*param_values))
        
        print(f"\nParameter space:")
        for name, values in param_space.items():
            print(f"  {name}: {values}")
        
        print(f"\nTotal configurations to test: {len(combinations)}")
        estimated_time = len(combinations) * 15  # Rough estimate: 15 min per config
        print(f"Estimated total time: {estimated_time:.1f} minutes")
        
        # Confirm execution
        response = input(f"\nProceed with {len(combinations)} configurations? (y/N): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return
        
        # Run benchmark
        benchmark_start = time.perf_counter()
        
        for i, combination in enumerate(combinations, 1):
            params = dict(zip(param_names, combination))
            result = self.run_single_configuration(i, params)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results_incremental(i)
            
            # Print progress
            elapsed = time.perf_counter() - benchmark_start
            avg_time = elapsed / i
            remaining = (len(combinations) - i) * avg_time
            
            print(f"\nProgress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
            print(f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")
        
        # Save final results
        total_benchmark_time = time.perf_counter() - benchmark_start
        self.save_final_results(mode, total_benchmark_time)
        
        print(f"\n🎉 Parameter sweep completed!")
        print(f"Total benchmark time: {total_benchmark_time/60:.1f} minutes")
        print(f"Results saved in: {self.output_dir}")
    
    def save_results_incremental(self, current_config: int):
        """Save incremental results after each configuration."""
        # Save as JSON
        json_file = self.output_dir / f"sweep_results_{self.timestamp}_incremental.json"
        with open(json_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": self.timestamp,
                    "configs_completed": current_config,
                    "last_update": datetime.now().isoformat()
                },
                "results": self.results
            }, f, indent=2)
    
    def save_final_results(self, mode: str, total_time: float):
        """Save final comprehensive results."""
        # Metadata
        metadata = {
            "timestamp": self.timestamp,
            "mode": mode,
            "total_configurations": len(self.results),
            "successful_configs": len([r for r in self.results if r["status"] == "success"]),
            "failed_configs": len([r for r in self.results if r["status"] == "failed"]),
            "total_benchmark_time": total_time,
            "system_info": {
                "python_version": "3.13.2",
                "platform": "macOS",
                "cpu_cores": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024**3
            }
        }
        
        # Save JSON (complete data)
        json_file = self.output_dir / f"parameter_sweep_{mode}_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "metadata": metadata,
                "results": self.results
            }, f, indent=2)
        
        # Save CSV (flattened for analysis)
        csv_file = self.output_dir / f"parameter_sweep_{mode}_{self.timestamp}.csv"
        self.save_results_csv(csv_file)
        
        # Save summary report
        summary_file = self.output_dir / f"parameter_sweep_summary_{mode}_{self.timestamp}.md"
        self.save_summary_report(summary_file, metadata)
        
        print(f"\nFiles created:")
        print(f"  📄 {json_file}")
        print(f"  📊 {csv_file}")
        print(f"  📋 {summary_file}")
    
    def save_results_csv(self, csv_file: Path):
        """Save results in CSV format for easy analysis."""
        flattened_results = []
        
        for result in self.results:
            if result["status"] == "success":
                flat_result = {
                    "config_id": result["config_id"],
                    "status": result["status"],
                    # Parameters
                    "max_workers": result["parameters"]["max_workers"],
                    "n_components": result["parameters"]["n_components"],
                    "anomaly_threshold": result["parameters"]["anomaly_threshold"],
                    "num_weeks": result["parameters"]["num_weeks"],
                    "mad_threshold": result["parameters"]["mad_threshold"],
                    # Timing
                    "stage1_time": result["timing"]["stage1_time"],
                    "stage2_time": result["timing"]["stage2_time"],
                    "stage3_time": result["timing"]["stage3_time"],
                    "stage4_time": result["timing"]["stage4_time"],
                    "total_time": result["timing"]["total_time"],
                    # Memory
                    "memory_delta_mb": result["memory"]["delta_mb"],
                    # Metrics
                    **result.get("metrics", {})
                }
                flattened_results.append(flat_result)
        
        # Save to CSV
        if flattened_results:
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_file, index=False)
    
    def save_summary_report(self, summary_file: Path, metadata: dict):
        """Save markdown summary report."""
        successful_results = [r for r in self.results if r["status"] == "success"]
        
        with open(summary_file, 'w') as f:
            f.write(f"# Parameter Sweep Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode:** {metadata['mode']}\n")
            f.write(f"**Duration:** {metadata['total_benchmark_time']/60:.1f} minutes\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total configurations: {metadata['total_configurations']}\n")
            f.write(f"- Successful: {metadata['successful_configs']}\n")
            f.write(f"- Failed: {metadata['failed_configs']}\n")
            f.write(f"- Success rate: {metadata['successful_configs']/metadata['total_configurations']*100:.1f}%\n\n")
            
            if successful_results:
                # Performance statistics
                times = [r["timing"]["total_time"] for r in successful_results]
                f.write(f"## Performance Statistics\n\n")
                f.write(f"- Fastest configuration: {min(times):.1f}s\n")
                f.write(f"- Slowest configuration: {max(times):.1f}s\n")
                f.write(f"- Average time: {sum(times)/len(times):.1f}s\n")
                f.write(f"- Standard deviation: {pd.Series(times).std():.1f}s\n\n")
                
                # Best performing configurations
                f.write(f"## Top 5 Fastest Configurations\n\n")
                sorted_results = sorted(successful_results, key=lambda x: x["timing"]["total_time"])
                for i, result in enumerate(sorted_results[:5], 1):
                    params = result["parameters"]
                    time_val = result["timing"]["total_time"]
                    f.write(f"{i}. **{time_val:.1f}s** - ")
                    f.write(f"workers={params['max_workers']}, ")
                    f.write(f"components={params['n_components']}, ")
                    f.write(f"threshold={params['anomaly_threshold']}, ")
                    f.write(f"weeks={params['num_weeks']}, ")
                    f.write(f"mad={params['mad_threshold']}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Parameter Sweep Benchmark")
    parser.add_argument("--mode", choices=["quick", "standard", "extensive"], 
                       default="standard", help="Benchmark mode")
    parser.add_argument("--output_dir", default="results/benchmarks/",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run benchmark
    benchmark = ParameterSweepBenchmark(args.output_dir)
    benchmark.run_parameter_sweep(args.mode)


if __name__ == "__main__":
    main()
