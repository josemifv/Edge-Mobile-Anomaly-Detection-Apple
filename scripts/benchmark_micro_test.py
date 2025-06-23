#!/usr/bin/env python3
"""
benchmark_micro_test.py

CMMSE 2025: Micro Parameter Sweep Test
Ultra-small benchmark for system validation (4 configurations only)

This script tests just 4 carefully selected parameter combinations
to validate the benchmark system quickly.
"""

import subprocess
import json
import time
import psutil
from pathlib import Path
from datetime import datetime

def run_micro_benchmark():
    """Run micro benchmark with just 4 configurations."""
    
    # Micro test configurations (carefully selected)
    configs = [
        {"max_workers": 4, "n_components": 2, "anomaly_threshold": 1.5, "num_weeks": 3, "mad_threshold": 1.0},
        {"max_workers": 8, "n_components": 3, "anomaly_threshold": 2.0, "num_weeks": 4, "mad_threshold": 1.5},
        {"max_workers": 10, "n_components": 2, "anomaly_threshold": 2.0, "num_weeks": 3, "mad_threshold": 1.5},
        {"max_workers": 8, "n_components": 3, "anomaly_threshold": 1.5, "num_weeks": 4, "mad_threshold": 1.0}
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/benchmarks/micro_test_{timestamp}")
    data_dir = Path("results/data")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧪 MICRO BENCHMARK TEST")
    print(f"Configurations: {len(configs)}")
    print(f"Estimated time: {len(configs) * 15:.1f} minutes")
    print(f"Timestamp: {timestamp}")
    
    results = []
    
    for i, params in enumerate(configs, 1):
        print(f"\n{'='*50}")
        print(f"Micro Config {i}/{len(configs)}: {params}")
        print('='*50)
        
        config_output = data_dir / f"micro_{i:02d}_{timestamp}"
        config_output.mkdir(exist_ok=True)
        
        start_time = time.perf_counter()
        
        try:
            # Stage 1: Data Ingestion
            print("🔄 Running Stage 1: Data Ingestion...")
            stage1_start = time.perf_counter()
            cmd1 = [
                "uv", "run", "scripts/01_data_ingestion.py",
                "data/raw/",
                "--output_path", str(config_output / "ingested_data.parquet"),
                "--max_workers", str(params["max_workers"])
            ]
            subprocess.run(cmd1, check=True, capture_output=True)
            stage1_time = time.perf_counter() - stage1_start
            print(f"   ✅ Stage 1 completed in {stage1_time:.1f}s")
            
            # Stage 2: Data Preprocessing  
            print("🔄 Running Stage 2: Data Preprocessing...")
            stage2_start = time.perf_counter()
            cmd2 = [
                "uv", "run", "scripts/02_data_preprocessing.py",
                str(config_output / "ingested_data.parquet"),
                "--output_path", str(config_output / "preprocessed_data.parquet")
            ]
            subprocess.run(cmd2, check=True, capture_output=True)
            stage2_time = time.perf_counter() - stage2_start
            print(f"   ✅ Stage 2 completed in {stage2_time:.1f}s")
            
            # Stage 3: Week Selection
            print("🔄 Running Stage 3: Week Selection...")
            stage3_start = time.perf_counter()
            cmd3 = [
                "uv", "run", "scripts/03_week_selection.py",
                str(config_output / "preprocessed_data.parquet"),
                "--output_path", str(config_output / "reference_weeks.parquet"),
                "--num_weeks", str(params["num_weeks"]),
                "--mad_threshold", str(params["mad_threshold"])
            ]
            subprocess.run(cmd3, check=True, capture_output=True)
            stage3_time = time.perf_counter() - stage3_start
            print(f"   ✅ Stage 3 completed in {stage3_time:.1f}s")
            
            # Stage 4: Individual Anomaly Detection
            print("🔄 Running Stage 4: Individual Anomaly Detection...")
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
            subprocess.run(cmd4, check=True, capture_output=True)
            stage4_time = time.perf_counter() - stage4_start
            print(f"   ✅ Stage 4 completed in {stage4_time:.1f}s")
            
            total_time = time.perf_counter() - start_time
            
            # Calculate file sizes
            file_sizes = {
                "ingested_mb": (config_output / "ingested_data.parquet").stat().st_size / 1024 / 1024,
                "preprocessed_mb": (config_output / "preprocessed_data.parquet").stat().st_size / 1024 / 1024,
                "reference_weeks_mb": (config_output / "reference_weeks.parquet").stat().st_size / 1024 / 1024,
                "individual_anomalies_mb": (config_output / "individual_anomalies.parquet").stat().st_size / 1024 / 1024
            }
            
            result = {
                "config_id": i,
                "status": "success",
                "parameters": params,
                "timing": {
                    "stage1_time": stage1_time,
                    "stage2_time": stage2_time,
                    "stage3_time": stage3_time,
                    "stage4_time": stage4_time,
                    "total_time": total_time
                },
                "file_sizes": file_sizes,
                "output_dir": str(config_output)
            }
            
            results.append(result)
            
            print(f"✅ Configuration {i} completed successfully!")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Throughput: {319896289/total_time:,.0f} rows/second")
            
        except Exception as e:
            print(f"❌ Configuration {i} failed: {e}")
            results.append({
                "config_id": i,
                "status": "failed",
                "parameters": params,
                "error": str(e)
            })
    
    # Save results
    results_file = output_dir / "micro_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_configs": len(configs),
                "successful_configs": len([r for r in results if r["status"] == "success"]),
                "system_info": {
                    "cpu_cores": psutil.cpu_count(),
                    "memory_gb": psutil.virtual_memory().total / 1024**3
                }
            },
            "results": results
        }, f, indent=2)
    
    # Print summary
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        times = [r["timing"]["total_time"] for r in successful]
        print(f"\n🎉 MICRO BENCHMARK COMPLETED!")
        print(f"Successful configurations: {len(successful)}/{len(configs)}")
        print(f"Fastest: {min(times):.1f}s")
        print(f"Slowest: {max(times):.1f}s") 
        print(f"Average: {sum(times)/len(times):.1f}s")
        print(f"Results saved: {results_file}")
        
        # Show parameter effects
        print(f"\n📊 PARAMETER EFFECTS:")
        for result in sorted(successful, key=lambda x: x["timing"]["total_time"]):
            params = result["parameters"]
            time_val = result["timing"]["total_time"]
            print(f"  {time_val:6.1f}s - workers={params['max_workers']}, components={params['n_components']}, threshold={params['anomaly_threshold']}")


if __name__ == "__main__":
    run_micro_benchmark()
