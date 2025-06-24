#!/usr/bin/env python3
"""
test_throughput_metrics.py

CMMSE 2025: Test Script for Throughput Metrics Computation
==========================================================

Tests the throughput and compression metrics computation system with sample data
to validate functionality before running actual benchmarks.

This script creates sample benchmark run data and tests:
• Row throughput calculations per stage
• Compression ratio computations
• CPU efficiency metrics analysis
• Thermal headroom calculations
• JSON persistence functionality

Usage:
    python scripts/utils/test_throughput_metrics.py
    
Example:
    python scripts/utils/test_throughput_metrics.py --verbose

Author: José Miguel Franco-Valiente
Created: December 2024
"""

import argparse
import json
import logging
import tempfile
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the metrics calculator
try:
    from compute_throughput_metrics import ThroughputMetricsCalculator, STAGE_ROW_COUNTS, STAGE_NAMES
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from compute_throughput_metrics import ThroughputMetricsCalculator, STAGE_ROW_COUNTS, STAGE_NAMES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThroughputMetricsTestSuite:
    """
    Test suite for validating throughput metrics computation functionality.
    
    Creates sample benchmark data and validates all metrics computation components.
    """
    
    def __init__(self, test_dir: Path):
        """
        Initialize the test suite.
        
        Args:
            test_dir: Directory to create test data in
        """
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized test suite in: {self.test_dir}")
    
    def create_sample_run_metrics(self, run_id: int, success: bool = True) -> Dict:
        """
        Create sample run metrics data.
        
        Args:
            run_id: Run identifier
            success: Whether the run was successful
            
        Returns:
            Sample run metrics dictionary
        """
        base_time = 100  # Base execution time
        
        # Simulate realistic stage timings
        stage_timings = {
            'stage1_time': base_time + np.random.normal(0, 10),  # ~100s
            'stage2_time': base_time * 0.8 + np.random.normal(0, 8),  # ~80s
            'stage3_time': base_time * 0.2 + np.random.normal(0, 5),  # ~20s
            'stage4_time': base_time * 2.0 + np.random.normal(0, 20),  # ~200s
            'stage5_time': base_time * 0.1 + np.random.normal(0, 2),   # ~10s
        }
        
        # Ensure positive timings
        for key in stage_timings:
            stage_timings[key] = max(stage_timings[key], 1.0)
        
        total_time = sum(stage_timings.values())
        
        return {
            'run_id': run_id,
            'success': success,
            'execution_time_seconds': total_time,
            'execution_time_minutes': total_time / 60,
            'start_timestamp': datetime.now().isoformat(),
            'end_timestamp': (datetime.now() + timedelta(seconds=total_time)).isoformat(),
            'parameters': {
                'n_components': 3,
                'anomaly_threshold': 2.0,
                'max_workers': 8,
                'input_dir': 'data/raw/'
            },
            'stage_timings': stage_timings,
            'stage_artifacts': self._create_sample_artifacts(),
            'system_metrics': {
                'initial_memory_mb': 1000 + np.random.normal(0, 100),
                'final_memory_mb': 1500 + np.random.normal(0, 150),
                'memory_delta_mb': 500 + np.random.normal(0, 50),
                'initial_cpu_percent': 5 + np.random.normal(0, 2),
                'final_cpu_percent': 10 + np.random.normal(0, 3)
            },
            'error_message': None if success else 'Simulated test error'
        }
    
    def _create_sample_artifacts(self) -> Dict:
        """Create sample stage artifacts information."""
        # Realistic file sizes based on actual pipeline data
        base_sizes = {
            'stage1': 4_400_000_000,  # ~4.4GB ingested data
            'stage2': 2_000_000_000,  # ~2.0GB preprocessed data
            'stage3': 479_000,        # ~479KB reference weeks
            'stage4': 148_000_000     # ~148MB individual anomalies
        }
        
        artifacts = {}
        for stage, base_size in base_sizes.items():
            # Add some variation
            size_bytes = int(base_size * (1 + np.random.normal(0, 0.1)))
            artifacts[stage] = {
                'filename': f'{stage.replace("stage", "0")}_test_data.parquet',
                'size_bytes': size_bytes,
                'size_mb': round(size_bytes / (1024**2), 2),
                'size_gb': round(size_bytes / (1024**3), 3)
            }
        
        return artifacts
    
    def create_sample_monitoring_data(self, duration_seconds: float) -> pd.DataFrame:
        """
        Create sample resource monitoring data.
        
        Args:
            duration_seconds: Duration of monitoring period
            
        Returns:
            DataFrame with monitoring samples
        """
        # Sample every second
        num_samples = int(duration_seconds)
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=num_samples,
            freq='1S'
        )
        
        # Generate realistic CPU and memory data
        np.random.seed(42)  # For reproducible test data
        
        # CPU usage patterns (higher during processing)
        base_cpu = 50
        cpu_noise = np.random.normal(0, 10, num_samples)
        system_cpu = np.clip(base_cpu + cpu_noise, 0, 100)
        
        process_cpu = system_cpu * 0.8 + np.random.normal(0, 5, num_samples)
        process_cpu = np.clip(process_cpu, 0, 100)
        
        # Per-core CPU data (8 cores)
        core_data = {}
        for i in range(8):
            core_usage = system_cpu + np.random.normal(0, 15, num_samples)
            core_data[f'cpu_core_{i}'] = np.clip(core_usage, 0, 100)
        
        # Memory usage (gradually increasing)
        base_memory = 2000  # MB
        memory_trend = np.linspace(0, 500, num_samples)  # Gradual increase
        memory_noise = np.random.normal(0, 50, num_samples)
        memory_rss = base_memory + memory_trend + memory_noise
        
        # Temperature data (Apple Silicon)
        base_temp = 35  # Celsius
        temp_variation = np.random.normal(0, 5, num_samples)
        temp_data = base_temp + temp_variation + (system_cpu * 0.3)  # Temperature correlates with CPU
        temp_data = np.clip(temp_data, 20, 85)  # Realistic temperature range
        
        # Create monitoring DataFrame
        monitoring_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_percent': process_cpu,
            'system_cpu_percent': system_cpu,
            'cpu_cores_avg': system_cpu,
            'memory_rss': memory_rss * 1024 * 1024,  # Convert to bytes
            'memory_vms': memory_rss * 1.5 * 1024 * 1024,  # VMS usually higher
            'memory_percent': np.clip(memory_rss / 16000 * 100, 0, 100),  # Assume 16GB total
            'system_memory_percent': np.clip(memory_rss / 16000 * 100, 0, 100),
            'peak_memory_tracemalloc': memory_rss * 1024 * 1024,
            'peak_memory_rusage': memory_rss * 1024 * 1024,
            'cpu_die_temperature': temp_data,
            **core_data
        })
        
        return monitoring_data
    
    def create_sample_benchmark_data(self, num_runs: int = 3) -> None:
        """
        Create complete sample benchmark data structure.
        
        Args:
            num_runs: Number of sample runs to create
        """
        logger.info(f"Creating sample benchmark data with {num_runs} runs...")
        
        for run_id in range(1, num_runs + 1):
            run_dir = self.test_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Create run subdirectories
            (run_dir / "data").mkdir(exist_ok=True)
            (run_dir / "logs").mkdir(exist_ok=True)
            (run_dir / "reports").mkdir(exist_ok=True)
            
            # Create run metrics
            success = np.random.random() > 0.1  # 90% success rate
            run_metrics = self.create_sample_run_metrics(run_id, success)
            
            with open(run_dir / "run_metrics.json", 'w') as f:
                json.dump(run_metrics, f, indent=2)
            
            # Create monitoring data
            if success:
                execution_time = run_metrics['execution_time_seconds']
                monitoring_data = self.create_sample_monitoring_data(execution_time)
                monitoring_data.to_csv(run_dir / "resource_monitor.csv", index=False)
            
            # Create sample artifact files (empty files with correct names)
            data_dir = run_dir / "data"
            for stage, artifact_info in run_metrics['stage_artifacts'].items():
                filename = artifact_info['filename']
                artifact_file = data_dir / filename
                artifact_file.touch()  # Create empty file
            
            logger.debug(f"Created sample data for run {run_id}")
        
        logger.info(f"Sample benchmark data created in: {self.test_dir}")
    
    def test_metrics_computation(self) -> bool:
        """
        Test the complete metrics computation pipeline.
        
        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("Testing throughput metrics computation...")
        
        try:
            # Initialize calculator
            calculator = ThroughputMetricsCalculator(self.test_dir)
            
            # Load run data
            logger.info("Loading sample run data...")
            calculator.load_run_data()
            
            if not calculator.runs_data:
                logger.error("Failed to load sample run data!")
                return False
            
            logger.info(f"Loaded {len(calculator.runs_data)} runs")
            
            # Compute all metrics
            logger.info("Computing all metrics...")
            all_metrics = calculator.compute_all_metrics()
            
            if not all_metrics:
                logger.error("No metrics computed!")
                return False
            
            # Save metrics
            logger.info("Saving computed metrics...")
            output_file = calculator.save_metrics_to_json(all_metrics)
            
            # Validate results
            return self._validate_metrics_results(all_metrics, output_file)
            
        except Exception as e:
            logger.error(f"Error during metrics computation test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _validate_metrics_results(self, metrics: List[Dict], output_file: Path) -> bool:
        """
        Validate the computed metrics results.
        
        Args:
            metrics: List of computed metrics
            output_file: Path to saved metrics file
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating computed metrics...")
        
        # Check basic structure
        if not metrics:
            logger.error("No metrics to validate!")
            return False
        
        successful_runs = [m for m in metrics if m.get('success', False)]
        logger.info(f"Successful runs: {len(successful_runs)}/{len(metrics)}")
        
        # Validate each successful run
        for run_metrics in successful_runs:
            run_id = run_metrics.get('run_id', 'unknown')
            
            # Check required top-level keys
            required_keys = ['throughput_metrics', 'compression_metrics', 
                           'cpu_efficiency_metrics', 'thermal_metrics']
            
            for key in required_keys:
                if key not in run_metrics:
                    logger.error(f"Run {run_id} missing required key: {key}")
                    return False
            
            # Validate throughput metrics
            if not self._validate_throughput_metrics(run_metrics['throughput_metrics'], run_id):
                return False
            
            # Validate compression metrics
            if not self._validate_compression_metrics(run_metrics['compression_metrics'], run_id):
                return False
            
            # Validate CPU efficiency metrics
            if not self._validate_cpu_metrics(run_metrics['cpu_efficiency_metrics'], run_id):
                return False
            
            # Validate thermal metrics
            if not self._validate_thermal_metrics(run_metrics['thermal_metrics'], run_id):
                return False
        
        # Check that output files exist
        if not output_file.exists():
            logger.error(f"Output file not created: {output_file}")
            return False
        
        # Check summary file
        summary_file = output_file.parent / "throughput_metrics_summary.json"
        if not summary_file.exists():
            logger.error(f"Summary file not created: {summary_file}")
            return False
        
        logger.info("✅ All metrics validation tests passed!")
        return True
    
    def _validate_throughput_metrics(self, throughput_metrics: Dict, run_id: str) -> bool:
        """Validate throughput metrics structure and values."""
        if 'throughput_per_stage' not in throughput_metrics:
            logger.error(f"Run {run_id}: No throughput_per_stage data")
            return False
        
        stage_data = throughput_metrics['throughput_per_stage']
        
        for stage, row_count in STAGE_ROW_COUNTS.items():
            if stage in stage_data:
                stage_metrics = stage_data[stage]
                
                # Check required fields
                required_fields = ['rows_per_second', 'execution_time_seconds', 'total_rows']
                for field in required_fields:
                    if field not in stage_metrics:
                        logger.error(f"Run {run_id} stage {stage}: Missing field {field}")
                        return False
                
                # Check values make sense
                rows_per_sec = stage_metrics['rows_per_second']
                if rows_per_sec <= 0:
                    logger.error(f"Run {run_id} stage {stage}: Invalid rows_per_second: {rows_per_sec}")
                    return False
                
                # Check calculation
                expected_rows_per_sec = row_count / stage_metrics['execution_time_seconds']
                if abs(rows_per_sec - expected_rows_per_sec) > 1:  # Allow small rounding error
                    logger.error(f"Run {run_id} stage {stage}: Incorrect calculation")
                    return False
        
        return True
    
    def _validate_compression_metrics(self, compression_metrics: Dict, run_id: str) -> bool:
        """Validate compression metrics structure and values."""
        if 'compression_ratios' not in compression_metrics:
            logger.error(f"Run {run_id}: No compression_ratios data")
            return False
        
        ratios = compression_metrics['compression_ratios']
        
        # Check for expected compression ratios
        expected_ratios = ['ingested_to_preprocessed', 'preprocessed_to_anomalies', 'end_to_end']
        
        for ratio_name in expected_ratios:
            if ratio_name in ratios:
                ratio_data = ratios[ratio_name]
                
                # Check required fields
                required_fields = ['compression_ratio', 'space_saved_percent']
                for field in required_fields:
                    if field not in ratio_data:
                        logger.error(f"Run {run_id} {ratio_name}: Missing field {field}")
                        return False
                
                # Check values are reasonable
                compression_ratio = ratio_data['compression_ratio']
                if compression_ratio <= 0:
                    logger.error(f"Run {run_id} {ratio_name}: Invalid compression ratio: {compression_ratio}")
                    return False
                
                space_saved = ratio_data['space_saved_percent']
                if not 0 <= space_saved <= 100:
                    logger.error(f"Run {run_id} {ratio_name}: Invalid space saved: {space_saved}")
                    return False
        
        return True
    
    def _validate_cpu_metrics(self, cpu_metrics: Dict, run_id: str) -> bool:
        """Validate CPU efficiency metrics structure and values."""
        if cpu_metrics == 'monitoring_data_unavailable':
            logger.warning(f"Run {run_id}: No CPU monitoring data (expected for test)")
            return True
        
        # Check for CPU data
        if 'system_cpu' in cpu_metrics:
            system_cpu = cpu_metrics['system_cpu']
            
            required_fields = ['mean_percent', 'peak_percent']
            for field in required_fields:
                if field not in system_cpu:
                    logger.error(f"Run {run_id} system_cpu: Missing field {field}")
                    return False
                
                value = system_cpu[field]
                if not 0 <= value <= 100:
                    logger.error(f"Run {run_id} system_cpu {field}: Invalid value: {value}")
                    return False
        
        if 'cpu_efficiency' in cpu_metrics and isinstance(cpu_metrics['cpu_efficiency'], dict):
            efficiency = cpu_metrics['cpu_efficiency']
            
            if 'efficiency_score' in efficiency:
                score = efficiency['efficiency_score']
                if score <= 0:
                    logger.error(f"Run {run_id}: Invalid efficiency score: {score}")
                    return False
        
        return True
    
    def _validate_thermal_metrics(self, thermal_metrics: Dict, run_id: str) -> bool:
        """Validate thermal metrics structure and values."""
        if thermal_metrics.get('thermal_headroom') == 'monitoring_data_unavailable':
            logger.warning(f"Run {run_id}: No thermal monitoring data (may be expected)")
            return True
        
        if thermal_metrics.get('thermal_headroom') == 'temperature_data_unavailable':
            logger.warning(f"Run {run_id}: No temperature data (may be expected)")
            return True
        
        if 'thermal_headroom' in thermal_metrics and isinstance(thermal_metrics['thermal_headroom'], dict):
            headroom = thermal_metrics['thermal_headroom']
            
            if 'max_temperature_reached_celsius' in headroom:
                max_temp = headroom['max_temperature_reached_celsius']
                if not 20 <= max_temp <= 120:  # Reasonable temperature range
                    logger.error(f"Run {run_id}: Unrealistic temperature: {max_temp}°C")
                    return False
        
        return True
    
    def run_all_tests(self) -> bool:
        """
        Run complete test suite.
        
        Returns:
            True if all tests pass, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING THROUGHPUT METRICS TEST SUITE")
        logger.info("=" * 60)
        
        try:
            # Create sample data
            self.create_sample_benchmark_data(num_runs=3)
            
            # Test metrics computation
            success = self.test_metrics_computation()
            
            if success:
                logger.info("=" * 60)
                logger.info("✅ ALL TESTS PASSED SUCCESSFULLY")
                logger.info("=" * 60)
            else:
                logger.error("=" * 60)
                logger.error("❌ SOME TESTS FAILED")
                logger.error("=" * 60)
            
            return success
            
        except Exception as e:
            logger.error(f"Critical error during test execution: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Test throughput metrics computation with sample data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run basic test suite
    python scripts/utils/test_throughput_metrics.py
    
    # Run with verbose output
    python scripts/utils/test_throughput_metrics.py --verbose
    
    # Use custom test directory
    python scripts/utils/test_throughput_metrics.py --test_dir /tmp/test_metrics/
        """
    )
    
    parser.add_argument(
        '--test_dir',
        type=Path,
        help='Custom test directory (default: temporary directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--keep_test_data',
        action='store_true',
        help='Keep test data after completion'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test directory
    if args.test_dir:
        test_dir = args.test_dir
        cleanup = False
    else:
        import tempfile
        test_dir = Path(tempfile.mkdtemp(prefix='throughput_metrics_test_'))
        cleanup = not args.keep_test_data
    
    try:
        # Initialize and run test suite
        test_suite = ThroughputMetricsTestSuite(test_dir)
        success = test_suite.run_all_tests()
        
        if success:
            print("\n✅ All throughput metrics tests passed successfully!")
            print(f"Test data location: {test_dir}")
            
            if cleanup:
                print("Test data will be cleaned up automatically.")
            else:
                print("Test data preserved for inspection.")
            
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            print(f"Test data preserved for debugging: {test_dir}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical test error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup if requested
        if cleanup:
            try:
                import shutil
                shutil.rmtree(test_dir)
                logger.debug(f"Cleaned up test directory: {test_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up test directory: {e}")


if __name__ == "__main__":
    main()
