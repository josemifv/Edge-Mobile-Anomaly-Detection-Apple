#!/usr/bin/env python3
"""
Simple Pipeline Benchmark - Run pipeline 10 times and get basic results
"""

import subprocess
import time
import re
import json
import statistics
from pathlib import Path
import argparse
import os
import datetime

def run_pipeline_once(run_id, data_path="data/raw/", output_dir="."):
    """Run the pipeline once and extract timing info"""
    print(f"🚀 Running pipeline iteration {run_id}/10...")
    
    start_time = time.time()
    
    # Prepare output directory for this run
    run_output_dir = Path(output_dir) / f"results_run_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Run the pipeline
    result = subprocess.run(
        ["uv", "run", "scripts/run_pipeline.py", data_path, "--output_dir", str(run_output_dir)],
        capture_output=True,
        text=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if result.returncode != 0:
        print(f"❌ Run {run_id} failed!")
        print(result.stderr)
        return None
    
    # Extract stage timings from output
    stage_times = {}
    stage_pattern = r"✅ Stage (\d+) completed successfully in ([\d.]+) seconds"
    
    for match in re.finditer(stage_pattern, result.stdout):
        stage_num = int(match.group(1))
        stage_time = float(match.group(2))
        stage_times[f"stage_{stage_num}"] = stage_time
    
    # Extract total pipeline time if available
    total_pattern = r"Total pipeline execution time: ([\d.]+) seconds"
    total_match = re.search(total_pattern, result.stdout)
    pipeline_time = float(total_match.group(1)) if total_match else sum(stage_times.values())
    
    run_data = {
        "run_id": run_id,
        "success": True,
        "total_time": total_time,
        "pipeline_time": pipeline_time,
        **stage_times
    }
    
    print(f"✅ Run {run_id} completed in {total_time:.1f}s (pipeline: {pipeline_time:.1f}s)")
    return run_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Simple Pipeline Benchmark")
    parser.add_argument("--data_path", default="data/raw/", help="Path to input data")
    parser.add_argument("--output_dir", default=".", help="Base directory for outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    # Create timestamped benchmark directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"🔖 Storing all outputs in {base_output_dir}")
    print("🔬 Simple Pipeline Benchmark - 10 runs")
    print("=" * 50)
    
    results = []
    
    # Run pipeline 10 times
    for i in range(1, 11):
        result = run_pipeline_once(i, args.data_path, base_output_dir)
        if result:
            results.append(result)
        time.sleep(1)  # Brief pause between runs
    
    if not results:
        print("❌ No successful runs!")
        return
    
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    # Calculate basic statistics
    total_times = [r["total_time"] for r in results]
    pipeline_times = [r["pipeline_time"] for r in results]
    
    print(f"Successful runs: {len(results)}/10")
    print(f"Total execution time:")
    print(f"  Mean: {statistics.mean(total_times):.1f}s")
    print(f"  Min:  {min(total_times):.1f}s")
    print(f"  Max:  {max(total_times):.1f}s")
    print(f"  Std:  {statistics.stdev(total_times) if len(total_times) > 1 else 0:.1f}s")
    
    print(f"\nPipeline execution time:")
    print(f"  Mean: {statistics.mean(pipeline_times):.1f}s")
    print(f"  Min:  {min(pipeline_times):.1f}s")
    print(f"  Max:  {max(pipeline_times):.1f}s")
    print(f"  Std:  {statistics.stdev(pipeline_times) if len(pipeline_times) > 1 else 0:.1f}s")
    
    # Stage-wise analysis if available
    stage_keys = [k for k in results[0].keys() if k.startswith("stage_")]
    if stage_keys:
        print(f"\nStage timings (mean ± std):")
        for stage in sorted(stage_keys):
            stage_times = [r.get(stage, 0) for r in results if stage in r]
            if stage_times:
                mean_time = statistics.mean(stage_times)
                std_time = statistics.stdev(stage_times) if len(stage_times) > 1 else 0
                print(f"  {stage}: {mean_time:.1f} ± {std_time:.1f}s")
    
    # Save results
    output_file = os.path.join(base_output_dir, "benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "successful_runs": len(results),
                "total_runs": 10,
                "mean_total_time": statistics.mean(total_times),
                "mean_pipeline_time": statistics.mean(pipeline_times),
                "std_total_time": statistics.stdev(total_times) if len(total_times) > 1 else 0,
                "std_pipeline_time": statistics.stdev(pipeline_times) if len(pipeline_times) > 1 else 0
            },
            "individual_runs": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"\n🎯 Average pipeline time: {statistics.mean(pipeline_times):.1f}s")

if __name__ == "__main__":
    main()
