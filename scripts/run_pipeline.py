#!/usr/bin/env python3
"""
run_pipeline.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Complete 5-Stage Pipeline Runner with Robust Error Handling

Executes the complete 5-stage pipeline for mobile network anomaly detection with:
• Comprehensive try/except error handling for each stage
• Detailed error logging and status tracking
• Temporary file cleanup management
• Graceful failure recovery and reporting

Usage:
    python run_pipeline.py <input_data_dir> [--output_dir <dir>] [--keep_tmp]

Example:
    python run_pipeline.py inputs/raw/ --output_dir outputs/ --keep_tmp
"""

import subprocess
import argparse
import time
import logging
import logging.handlers
import sys
import traceback
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def setup_robust_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """
    Set up rotating file handler for comprehensive pipeline logging.
    
    Args:
        log_file: Path to the log file
        verbose: Enable verbose logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('pipeline_runner')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Rotating file handler (50MB max, keep 3 backup files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class PipelineStatus:
    """Track pipeline execution status and metrics."""
    
    def __init__(self):
        self.stage_results = {}
        self.total_start_time = None
        self.total_end_time = None
        self.failed_stage = None
        self.error_message = None
        self.error_traceback = None
    
    def start_pipeline(self):
        """Mark pipeline start time."""
        self.total_start_time = time.perf_counter()
    
    def record_stage_success(self, stage_num: int, duration: float, description: str):
        """Record successful stage completion."""
        self.stage_results[stage_num] = {
            'status': 'success',
            'duration': duration,
            'description': description,
            'error': None
        }
    
    def record_stage_failure(self, stage_num: int, duration: float, description: str, error: str, traceback_str: str = None):
        """Record stage failure."""
        self.stage_results[stage_num] = {
            'status': 'failed',
            'duration': duration,
            'description': description,
            'error': error
        }
        self.failed_stage = stage_num
        self.error_message = error
        self.error_traceback = traceback_str
    
    def complete_pipeline(self):
        """Mark pipeline completion time."""
        self.total_end_time = time.perf_counter()
    
    def get_total_duration(self) -> float:
        """Get total pipeline duration."""
        if self.total_start_time and self.total_end_time:
            return self.total_end_time - self.total_start_time
        return 0.0
    
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.failed_stage is None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        return {
            'total_duration': self.get_total_duration(),
            'successful': self.is_successful(),
            'failed_stage': self.failed_stage,
            'error_message': self.error_message,
            'stage_results': self.stage_results,
            'timestamp': datetime.now().isoformat()
        }


def run_stage_robust(stage_num: int, script_name: str, args: list, description: str, 
                    logger: logging.Logger, status: PipelineStatus) -> float:
    """
    Run a single pipeline stage with comprehensive error handling.
    
    Args:
        stage_num: Stage number (1-5)
        script_name: Script filename to execute
        args: Arguments to pass to the script
        description: Human-readable stage description
        logger: Logger instance for recording events
        status: PipelineStatus instance for tracking
        
    Returns:
        Stage execution duration in seconds
        
    Raises:
        SystemExit: On stage failure (after logging and status recording)
    """
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num}: {description}")
    print('='*60)
    
    logger.info(f"Starting Stage {stage_num}: {description}")
    start_time = time.perf_counter()
    
    try:
        cmd = [sys.executable, f"scripts/{script_name}"] + args
        logger.debug(f"Executing command: {' '.join(str(c) for c in cmd)}")
        print(f"Executing: {' '.join(str(c) for c in cmd)}")
        
        # Execute with comprehensive output capture
        result = subprocess.run(
            cmd, 
            check=True, 
            text=True, 
            capture_output=True,
            timeout=3600  # 1 hour timeout per stage
        )
        
        duration = time.perf_counter() - start_time
        
        # Log successful completion
        logger.info(f"Stage {stage_num} completed successfully in {duration:.2f} seconds")
        
        # Log any stdout/stderr for debugging
        if result.stdout.strip():
            logger.debug(f"Stage {stage_num} stdout: {result.stdout[:1000]}...")  # Truncate long output
        if result.stderr.strip():
            logger.debug(f"Stage {stage_num} stderr: {result.stderr[:1000]}...")
        
        print(f"\n✅ Stage {stage_num} completed successfully in {duration:.2f} seconds.")
        status.record_stage_success(stage_num, duration, description)
        
        return duration
        
    except subprocess.CalledProcessError as e:
        duration = time.perf_counter() - start_time
        error_msg = f"Stage {stage_num} subprocess failed with exit code {e.returncode}"
        
        logger.error(error_msg)
        logger.error(f"Command that failed: {' '.join(str(c) for c in cmd)}")
        
        # Log captured output
        if e.stdout:
            logger.error(f"Subprocess stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Subprocess stderr: {e.stderr}")
        
        status.record_stage_failure(stage_num, duration, description, error_msg, traceback.format_exc())
        
        print(f"\n❌ Stage {stage_num} FAILED after {duration:.2f} seconds.")
        print(f"Error: {error_msg}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        
        # Exit with stage-specific error code
        sys.exit(stage_num)
        
    except subprocess.TimeoutExpired as e:
        duration = time.perf_counter() - start_time
        error_msg = f"Stage {stage_num} timed out after {e.timeout} seconds"
        
        logger.error(error_msg)
        logger.error(f"Command that timed out: {' '.join(str(c) for c in cmd)}")
        
        status.record_stage_failure(stage_num, duration, description, error_msg, traceback.format_exc())
        
        print(f"\n❌ Stage {stage_num} TIMED OUT after {duration:.2f} seconds.")
        print(f"Error: {error_msg}")
        
        sys.exit(stage_num + 10)  # Timeout errors get +10
        
    except FileNotFoundError as e:
        duration = time.perf_counter() - start_time
        error_msg = f"Stage {stage_num} script not found: scripts/{script_name}"
        
        logger.error(error_msg)
        logger.error(f"FileNotFoundError: {e}")
        
        status.record_stage_failure(stage_num, duration, description, error_msg, traceback.format_exc())
        
        print(f"\n❌ Stage {stage_num} FAILED: Script not found.")
        print(f"Error: {error_msg}")
        
        sys.exit(stage_num + 20)  # File not found errors get +20
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        error_msg = f"Stage {stage_num} unexpected error: {str(e)}"
        
        logger.error(error_msg)
        logger.error(f"Unexpected exception: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        status.record_stage_failure(stage_num, duration, description, error_msg, traceback.format_exc())
        
        print(f"\n❌ Stage {stage_num} FAILED with unexpected error.")
        print(f"Error: {error_msg}")
        
        sys.exit(stage_num + 30)  # Unexpected errors get +30


def cleanup_temporary_files(output_dir: Path, keep_tmp: bool, logger: logging.Logger):
    """
    Clean up bulky parquet artifacts if --keep_tmp not set.
    
    Removes large intermediate files while preserving:
    - Small metadata files (reference weeks)
    - Final analysis results
    - Log files
    
    Args:
        output_dir: Directory containing pipeline outputs
        keep_tmp: If True, skip cleanup
        logger: Logger for recording cleanup activities
    """
    if keep_tmp:
        logger.info("Keeping temporary files as requested (--keep_tmp flag)")
        return
    
    logger.info("Starting cleanup of temporary parquet artifacts...")
    
    # Files to remove (bulky artifacts)
    files_to_remove = [
        '01_ingested_data.parquet',      # Stage 1 output (typically 4-5 GB)
        '02_preprocessed_data.parquet',  # Stage 2 output (typically 2-3 GB)
        # Keep: 03_reference_weeks.parquet (small, ~500KB)
        # Keep: 04_individual_anomalies.parquet (needed for analysis)
    ]
    
    total_bytes_cleaned = 0
    files_removed = 0
    
    try:
        for filename in files_to_remove:
            file_path = output_dir / filename
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_bytes_cleaned += file_size
                    files_removed += 1
                    
                    mb_size = file_size / (1024 * 1024)
                    logger.info(f"Removed {filename} ({mb_size:.1f} MB)")
                    
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {e}")
        
        if files_removed > 0:
            mb_cleaned = total_bytes_cleaned / (1024 * 1024)
            gb_cleaned = total_bytes_cleaned / (1024 * 1024 * 1024)
            
            logger.info(f"Cleanup completed: {files_removed} files removed")
            logger.info(f"Space freed: {mb_cleaned:.1f} MB ({gb_cleaned:.2f} GB)")
            print(f"\n🧹 Cleanup: Removed {files_removed} temporary files ({gb_cleaned:.2f} GB freed)")
        else:
            logger.info("No temporary files found to clean up")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
        print(f"⚠️  Warning: Cleanup failed: {e}")


def main():
    """Main execution function with robust error handling and cleanup."""
    parser = argparse.ArgumentParser(
        description="CMMSE 2025: Complete 5-Stage Pipeline Runner with Robust Error Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pipeline execution
  python run_pipeline.py data/raw/ --output_dir outputs/

  # With custom parameters and cleanup disabled
  python run_pipeline.py data/raw/ --output_dir outputs/ \
      --n_components 5 --anomaly_threshold 2.5 --keep_tmp --verbose

  # Quick preview run
  python run_pipeline.py data/raw/ --preview --verbose
        """
    )
    
    parser.add_argument("input_dir", type=Path, help="Directory containing raw .txt data files")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Base directory for all generated outputs")
    parser.add_argument("--reports_dir", type=Path, default=None, help="Directory for final analysis reports (default: output_dir/reports)")
    parser.add_argument("--max_workers", type=int, help="Max parallel processes for applicable stages")
    parser.add_argument("--n_components", type=int, default=3, help="OSP SVD components (default: 3)")
    parser.add_argument("--anomaly_threshold", type=float, default=2.0, help="OSP anomaly threshold (default: 2.0)")
    parser.add_argument("--preview", action="store_true", help="Show data previews after each stage")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary parquet files (do not clean up bulky artifacts)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    # Set up reports directory
    if args.reports_dir is None:
        args.reports_dir = args.output_dir / "reports"
    
    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up robust logging
    log_file = args.output_dir / "pipeline_execution.log"
    logger = setup_robust_logging(log_file, args.verbose)
    
    # Initialize pipeline status tracking
    status = PipelineStatus()
    
    print("="*80)
    print("CMMSE 2025: ROBUST ANOMALY DETECTION PIPELINE - START")
    print("="*80)
    print(f"Input data: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Keep temporary files: {args.keep_tmp}")
    print(f"Log file: {log_file}")
    
    logger.info("Starting robust pipeline execution")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Reports directory: {args.reports_dir}")
    logger.info(f"Pipeline parameters: n_components={args.n_components}, threshold={args.anomaly_threshold}")
    
    # Define output paths for each stage
    path_stage1_out = args.output_dir / "01_ingested_data.parquet"
    path_stage2_out = args.output_dir / "02_preprocessed_data.parquet"
    path_stage3_out = args.output_dir / "03_reference_weeks.parquet"
    path_stage4_out = args.output_dir / "04_individual_anomalies.parquet"
    
    status.start_pipeline()
    
    try:
        common_args = ["--preview"] if args.preview else []
        
        # Stage 1: Data Ingestion
        logger.info("Preparing Stage 1: Data Ingestion")
        stage1_args = [str(args.input_dir), "--output_path", str(path_stage1_out)] + common_args
        run_stage_robust(1, "01_data_ingestion.py", stage1_args, "Data Ingestion", logger, status)
        
        # Stage 2: Data Preprocessing  
        logger.info("Preparing Stage 2: Data Preprocessing")
        stage2_args = [str(path_stage1_out), "--output_path", str(path_stage2_out)] + common_args
        run_stage_robust(2, "02_data_preprocessing.py", stage2_args, "Data Preprocessing & Aggregation", logger, status)
        
        # Stage 3: Reference Week Selection
        logger.info("Preparing Stage 3: Reference Week Selection")
        stage3_args = [str(path_stage2_out), "--output_path", str(path_stage3_out)] + common_args
        run_stage_robust(3, "03_week_selection.py", stage3_args, "Reference Week Selection", logger, status)
        
        # Stage 4: OSP Anomaly Detection
        logger.info("Preparing Stage 4: OSP Anomaly Detection")
        stage4_args = [
            str(path_stage2_out),
            str(path_stage3_out),
            "--output_path", str(path_stage4_out),
            "--n_components", str(args.n_components),
            "--anomaly_threshold", str(args.anomaly_threshold)
        ] + common_args
        if args.max_workers:
            stage4_args.extend(["--max_workers", str(args.max_workers)])
        run_stage_robust(4, "04_anomaly_detection_individual.py", stage4_args, "OSP Anomaly Detection", logger, status)
        
        # Stage 5: Anomaly Analysis
        logger.info("Preparing Stage 5: Comprehensive Anomaly Analysis")
        stage5_args = [str(path_stage4_out), "--output_dir", str(args.reports_dir)] + common_args
        run_stage_robust(5, "05_analyze_anomalies.py", stage5_args, "Comprehensive Anomaly Analysis", logger, status)
        
        # Mark pipeline completion
        status.complete_pipeline()
        
        # Generate summary
        total_time = status.get_total_duration()
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        for stage_num, stage_result in status.stage_results.items():
            desc = stage_result['description']
            duration = stage_result['duration']
            print(f"Stage {stage_num} ({desc:<25}): {duration:8.2f} seconds")
        
        print("-" * 50)
        print(f"Total Pipeline Time:             {total_time:8.2f} seconds ({total_time/60:.2f} minutes)")
        print("✅ Complete 5-stage pipeline executed successfully!")
        
    except SystemExit as e:
        # Handle stage failures (already logged by run_stage_robust)
        status.complete_pipeline()
        logger.error(f"Pipeline failed with exit code {e.code}")
        
        # Save pipeline status for debugging
        try:
            import json
            status_file = args.output_dir / "pipeline_status.json"
            with open(status_file, 'w') as f:
                json.dump(status.get_summary(), f, indent=2, default=str)
            logger.info(f"Pipeline status saved to {status_file}")
        except Exception as status_error:
            logger.error(f"Failed to save pipeline status: {status_error}")
        
        # Re-raise to maintain exit code
        raise
        
    except Exception as e:
        # Handle unexpected errors
        status.complete_pipeline()
        logger.error(f"Unexpected pipeline error: {e}")
        logger.debug(f"Unexpected error traceback: {traceback.format_exc()}")
        
        print(f"\n❌ Pipeline failed with unexpected error: {e}")
        sys.exit(99)  # Unexpected error code
        
    finally:
        # Always attempt cleanup (regardless of success/failure)
        try:
            logger.info("Starting cleanup phase...")
            cleanup_temporary_files(args.output_dir, args.keep_tmp, logger)
            
            # Save final pipeline status
            import json
            status_file = args.output_dir / "pipeline_status.json"
            with open(status_file, 'w') as f:
                json.dump(status.get_summary(), f, indent=2, default=str)
            logger.info(f"Pipeline status saved to {status_file}")
            
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup phase: {cleanup_error}")
            logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
            print(f"⚠️  Warning: Cleanup phase failed: {cleanup_error}")
        
        finally:
            logger.info("Pipeline execution completed (with or without errors)")

if __name__ == "__main__":
    main()