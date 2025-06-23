#!/usr/bin/env python3
"""
run_pipeline.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Complete 5-Stage Pipeline Runner

Executes the complete 5-stage pipeline for mobile network anomaly detection.

Usage:
    python run_pipeline.py <input_data_dir> [--output_dir <dir>]

Example:
    python run_pipeline.py inputs/raw/ --output_dir outputs/
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
        cmd = [sys.executable, f"scripts/{script_name}"] + args
        print(f"Executing: {' '.join(str(c) for c in cmd)}")
        
        # Using check=True will raise CalledProcessError on non-zero exit codes
        subprocess.run(cmd, check=True, text=True)
        
        duration = time.perf_counter() - start_time
        print(f"\n✅ Stage {stage_num} completed successfully in {duration:.2f} seconds.")
        return duration
        
    except subprocess.CalledProcessError as e:
        duration = time.perf_counter() - start_time
        print(f"\n❌ Stage {stage_num} FAILED after {duration:.2f} seconds.")
        print(f"Error: {e}")
        # The CalledProcessError might not capture stdout/stderr well unless redirected.
        # The failing script's own output should indicate the error.
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Stage {stage_num} FAILED. Script 'scripts/{script_name}' not found.")
        sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Complete 5-Stage Pipeline Runner")
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .txt data files (e.g., inputs/raw)")
    parser.add_argument("--output_dir", type=Path, default="outputs/", help="Base directory for all generated outputs")
    parser.add_argument("--reports_dir", type=Path, default="reports/", help="Directory for final analysis reports")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes for applicable stages")
    parser.add_argument("--n_components", type=int, default=3, help="OSP SVD components")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0, help="OSP anomaly threshold")
    parser.add_argument("--preview", action="store_true", help="Show data previews after each stage")
    
    args = parser.parse_args()
    
    print("="*80); print("CMMSE 2025: ANOMALY DETECTION PIPELINE - START"); print("="*80)
    print(f"Input data: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reports directory: {args.reports_dir}")

    # Define output paths for each stage
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    
    path_stage1_out = args.output_dir / "01_ingested_data.parquet"
    path_stage2_out = args.output_dir / "02_preprocessed_data.parquet"
    path_stage3_out = args.output_dir / "03_reference_weeks.parquet"
    path_stage4_out = args.output_dir / "04_individual_anomalies.parquet"

    # Track execution times
    stage_times = {}
    pipeline_start = time.perf_counter()
    
    common_args = ["--preview"] if args.preview else []
    
    # Stage 1: Data Ingestion
    stage1_args = [str(args.input_dir), "--output_path", str(path_stage1_out)] + common_args
    stage_times['stage1'] = run_stage(1, "01_data_ingestion.py", stage1_args, "Data Ingestion")
    
    # Stage 2: Data Preprocessing  
    stage2_args = [str(path_stage1_out), "--output_path", str(path_stage2_out)] + common_args
    stage_times['stage2'] = run_stage(2, "02_data_preprocessing.py", stage2_args, "Data Preprocessing & Aggregation")
    
    # Stage 3: Reference Week Selection
    stage3_args = [str(path_stage2_out), "--output_path", str(path_stage3_out)] + common_args
    stage_times['stage3'] = run_stage(3, "03_week_selection.py", stage3_args, "Reference Week Selection")
    
    # Stage 4: OSP Anomaly Detection
    stage4_args = [
        str(path_stage2_out),
        str(path_stage3_out),
        "--output_path", str(path_stage4_out),
        "--n_components", str(args.n_components),
        "--anomaly_threshold", str(args.anomaly_threshold)
    ] + common_args
    if args.max_workers:
        stage4_args.extend(["--max_workers", str(args.max_workers)])
    stage_times['stage4'] = run_stage(4, "04_anomaly_detection_individual.py", stage4_args, "OSP Anomaly Detection")
    
    # Stage 5: Anomaly Analysis
    stage5_args = [str(path_stage4_out), "--output_dir", str(args.reports_dir)] + common_args
    stage_times['stage5'] = run_stage(5, "05_analyze_anomalies.py", stage5_args, "Comprehensive Anomaly Analysis")
    
    total_time = time.perf_counter() - pipeline_start
    
    print("\n" + "="*80); print("PIPELINE EXECUTION SUMMARY"); print("="*80)
    for i, desc in enumerate(["Data Ingestion", "Data Preprocessing", "Week Selection", "Anomaly Detection", "Anomaly Analysis"], 1):
        print(f"Stage {i} ({desc:<22}): {stage_times[f'stage{i}']:8.2f} seconds")
    print("-" * 50)
    print(f"Total Pipeline Time:             {total_time:8.2f} seconds ({total_time/60:.2f} minutes)")
    print("✅ Complete 5-stage pipeline executed successfully!")

if __name__ == "__main__":
    main()