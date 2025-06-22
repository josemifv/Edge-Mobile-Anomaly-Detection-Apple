#!/usr/bin/env python3
"""
run_pipeline.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Complete 5-Stage Pipeline Runner

Executes the complete refactored 5-stage pipeline for mobile network anomaly detection.
Optimized for Apple Silicon with configurable parameters.

Usage:
    python scripts/run_pipeline.py <input_data_dir> [--output_dir <dir>]

Example:
    python scripts/run_pipeline.py data/raw/ --output_dir results/
"""

import subprocess
import argparse
import time
from pathlib import Path
import sys


def run_stage(stage_num: int, script_name: str, args: list, description: str) -> float:
    """Run a single pipeline stage and return execution time."""
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {description}")
    print('='*60)
    
    start_time = time.perf_counter()
    
    try:
        # Run the script
        cmd = [sys.executable, f"scripts/{script_name}"] + args
        print(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        duration = time.perf_counter() - start_time
        print(f"\n✅ Stage {stage_num} completed successfully in {duration:.2f} seconds")
        return duration
        
    except subprocess.CalledProcessError as e:
        duration = time.perf_counter() - start_time
        print(f"\n❌ Stage {stage_num} failed after {duration:.2f} seconds")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Complete 5-Stage Pipeline Runner")
    parser.add_argument("input_data_dir", help="Directory containing raw .txt data files")
    parser.add_argument("--output_dir", default="results/", help="Output directory for results")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes")
    parser.add_argument("--n_components", type=int, default=3, help="OSP SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0, help="OSP anomaly threshold")
    parser.add_argument("--sample_cells", type=int, help="Process only N cells for testing")
    parser.add_argument("--preview", action="store_true", help="Show data previews")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CMMSE 2025: MOBILE NETWORK ANOMALY DETECTION PIPELINE")
    print("Complete 5-Stage Execution (Refactored)")
    print("="*80)
    print(f"Input data: {args.input_data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directories
    output_path = Path(args.output_dir)
    data_path = output_path / "data"
    analysis_path = output_path / "analysis"
    data_path.mkdir(parents=True, exist_ok=True)
    analysis_path.mkdir(parents=True, exist_ok=True)
    
    # Track execution times
    stage_times = {}
    pipeline_start = time.perf_counter()
    
    # Stage 1: Data Ingestion
    stage1_args = [
        args.input_data_dir,
        "--output_path", str(data_path / "ingested_data.parquet")
    ]
    if args.max_workers:
        stage1_args.extend(["--max_workers", str(args.max_workers)])
    
    stage_times['stage1'] = run_stage(
        1, "01_data_ingestion.py", stage1_args, 
        "Data Ingestion and Initial Processing"
    )
    
    # Stage 2: Data Preprocessing  
    stage2_args = [
        str(data_path / "ingested_data.parquet"),
        "--output_path", str(data_path / "preprocessed_data.parquet")
    ]
    if args.preview:
        stage2_args.append("--preview")
    
    stage_times['stage2'] = run_stage(
        2, "02_data_preprocessing.py", stage2_args,
        "Data Preprocessing and Aggregation"
    )
    
    # Stage 3: Reference Week Selection
    stage3_args = [
        str(data_path / "preprocessed_data.parquet"),
        "--output_path", str(data_path / "reference_weeks.parquet")
    ]
    if args.preview:
        stage3_args.append("--preview")
    
    stage_times['stage3'] = run_stage(
        3, "03_week_selection.py", stage3_args,
        "Reference Week Selection"
    )
    
    # Stage 4: Individual OSP Anomaly Detection
    stage4_args = [
        str(data_path / "preprocessed_data.parquet"),
        str(data_path / "reference_weeks.parquet"),
        "--output_path", str(data_path / "individual_anomalies.parquet"),
        "--n_components", str(args.n_components),
        "--anomaly_threshold", str(args.anomaly_threshold)
    ]
    if args.max_workers:
        stage4_args.extend(["--max_workers", str(args.max_workers)])
    if args.sample_cells:
        stage4_args.extend(["--sample_cells", str(args.sample_cells)])
    if args.preview:
        stage4_args.append("--preview")
    
    stage_times['stage4'] = run_stage(
        4, "04_anomaly_detection_individual.py", stage4_args,
        "Individual OSP Anomaly Detection"
    )
    
    # Stage 5: Comprehensive Anomaly Analysis
    stage5_args = [
        str(data_path / "individual_anomalies.parquet"),
        "--output_dir", str(analysis_path),
        "--top_n_severe", "20"
    ]
    if args.preview:
        stage5_args.append("--preview")
    
    stage_times['stage5'] = run_stage(
        5, "05_analyze_anomalies.py", stage5_args,
        "Comprehensive Anomaly Analysis"
    )
    
    # Pipeline Summary
    total_time = time.perf_counter() - pipeline_start
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Stage 1 (Data Ingestion):        {stage_times['stage1']:8.2f} seconds")
    print(f"Stage 2 (Data Preprocessing):    {stage_times['stage2']:8.2f} seconds") 
    print(f"Stage 3 (Week Selection):        {stage_times['stage3']:8.2f} seconds")
    print(f"Stage 4 (Individual Detection):  {stage_times['stage4']:8.2f} seconds")
    print(f"Stage 5 (Anomaly Analysis):      {stage_times['stage5']:8.2f} seconds")
    print("-" * 50)
    print(f"Total Pipeline Time:             {total_time:8.2f} seconds")
    print(f"Total Pipeline Time:             {total_time/60:8.2f} minutes")
    
    print(f"\nOutput files created in: {args.output_dir}")
    print("✅ Complete 5-stage pipeline executed successfully!")
    
    # Save execution summary
    summary_file = output_path / "pipeline_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CMMSE 2025: Mobile Network Anomaly Detection Pipeline\n")
        f.write("="*60 + "\n\n")
        f.write(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {args.input_data_dir}\n")
        f.write(f"Output Directory: {args.output_dir}\n\n")
        f.write("Stage Execution Times:\n")
        f.write(f"  Stage 1 (Data Ingestion):        {stage_times['stage1']:8.2f} seconds\n")
        f.write(f"  Stage 2 (Data Preprocessing):    {stage_times['stage2']:8.2f} seconds\n")
        f.write(f"  Stage 3 (Week Selection):        {stage_times['stage3']:8.2f} seconds\n")
        f.write(f"  Stage 4 (Individual Detection):  {stage_times['stage4']:8.2f} seconds\n")
        f.write(f"  Stage 5 (Anomaly Analysis):      {stage_times['stage5']:8.2f} seconds\n")
        f.write(f"  Total Pipeline Time:             {total_time:8.2f} seconds\n")
        f.write(f"  Total Pipeline Time:             {total_time/60:8.2f} minutes\n\n")
        f.write("OSP Configuration:\n")
        f.write(f"  SVD Components: {args.n_components}\n")
        f.write(f"  Anomaly Threshold: {args.anomaly_threshold}\n")
        if args.sample_cells:
            f.write(f"  Sample Cells: {args.sample_cells}\n")
        f.write("\nPipeline Architecture:\n")
        f.write("  Stage 1: Raw data ingestion and preprocessing\n")
        f.write("  Stage 2: Data aggregation and validation\n")
        f.write("  Stage 3: Reference week selection (MAD analysis)\n")
        f.write("  Stage 4: Individual anomaly detection (OSP)\n")
        f.write("  Stage 5: Comprehensive anomaly analysis\n")
    
    print(f"Execution summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
