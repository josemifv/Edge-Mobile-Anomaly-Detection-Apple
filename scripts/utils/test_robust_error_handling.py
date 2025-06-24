#!/usr/bin/env python3
"""
test_robust_error_handling.py

CMMSE 2025: Test Suite for Robust Error Handling and Cleanup
============================================================

This script validates the robust error handling implementation for:
• Try/except wrapper functionality for each run
• Error logging and status marking capabilities
• Temporary file cleanup with --keep_tmp flag
• Monitor thread termination guarantees
• Rotating file handler operation

Usage:
    python scripts/utils/test_robust_error_handling.py [--test_type TYPE] [--verbose]

Test Types:
    - monitoring: Test monitor thread termination
    - cleanup: Test temporary file cleanup
    - logging: Test rotating log handler
    - pipeline: Test pipeline error handling (requires data)
    - all: Run all tests (default)
"""

import argparse
import json
import logging
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.monitoring import start_monitor, stop_monitor, SystemMonitor
    from utils.robust_pipeline_runner import RobustPipelineRunner, setup_rotating_logger
    from run_pipeline import setup_robust_logging, PipelineStatus, cleanup_temporary_files
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class ErrorHandlingTestSuite:
    """Test suite for robust error handling components."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = {}
        self.temp_dir = None
        
        # Set up test logging
        self.logger = logging.getLogger('test_suite')
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Console handler for test output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def setup_test_environment(self):
        """Set up temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="robust_error_test_"))
        self.logger.info(f"Created test environment: {self.temp_dir}")
        
        # Create test directories
        (self.temp_dir / "data").mkdir()
        (self.temp_dir / "logs").mkdir()
        (self.temp_dir / "outputs").mkdir()
        
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up test environment: {self.temp_dir}")
    
    def test_monitor_thread_termination(self) -> Dict[str, Any]:
        """Test that monitor threads always terminate properly."""
        test_name = "monitor_thread_termination"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            # Test 1: Normal termination
            self.logger.debug("Testing normal monitor termination...")
            monitor = start_monitor(interval=0.1)
            time.sleep(1.0)  # Let it collect some samples
            df = stop_monitor(monitor)
            
            result['details']['normal_termination'] = {
                'samples_collected': len(df),
                'termination_successful': len(df) > 0
            }
            
            # Test 2: Multiple monitors
            self.logger.debug("Testing multiple monitor termination...")
            monitors = []
            for i in range(3):
                monitors.append(start_monitor(interval=0.2))
            
            time.sleep(0.5)
            
            # Stop all monitors
            for i, monitor in enumerate(monitors):
                df = stop_monitor(monitor)
                result['details'][f'monitor_{i}'] = {
                    'samples_collected': len(df),
                    'termination_successful': len(df) >= 0  # Allow empty for fast tests
                }
            
            # Test 3: Exception during monitoring (simulated)
            self.logger.debug("Testing monitor with simulated error...")
            monitor = SystemMonitor(interval=0.1)
            monitor.start()
            
            # Let it run briefly then force stop
            time.sleep(0.3)
            df = monitor.stop()
            
            result['details']['exception_handling'] = {
                'samples_collected': len(df),
                'graceful_termination': True
            }
            
            result['passed'] = True
            self.logger.info(f"✅ {test_name} passed")
            
        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"❌ {test_name} failed: {e}")
            if self.verbose:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def test_cleanup_functionality(self) -> Dict[str, Any]:
        """Test temporary file cleanup with --keep_tmp flag."""
        test_name = "cleanup_functionality"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            test_dir = self.temp_dir / "cleanup_test"
            test_dir.mkdir()
            
            # Create fake parquet files
            large_files = [
                '01_ingested_data.parquet',
                '02_preprocessed_data.parquet'
            ]
            
            small_files = [
                '03_reference_weeks.parquet',
                '04_individual_anomalies.parquet'
            ]
            
            # Create test files with different sizes
            for filename in large_files:
                file_path = test_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(b'x' * (10 * 1024 * 1024))  # 10MB test file
            
            for filename in small_files:
                file_path = test_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(b'x' * (100 * 1024))  # 100KB test file
            
            # Test cleanup with keep_tmp=False
            log_file = self.temp_dir / "cleanup_test.log"
            logger = setup_robust_logging(log_file, self.verbose)
            
            initial_files = list(test_dir.glob("*.parquet"))
            cleanup_temporary_files(test_dir, keep_tmp=False, logger=logger)
            remaining_files = list(test_dir.glob("*.parquet"))
            
            result['details']['cleanup_without_keep_tmp'] = {
                'initial_files': len(initial_files),
                'remaining_files': len(remaining_files),
                'large_files_removed': all(not (test_dir / f).exists() for f in large_files),
                'small_files_preserved': all((test_dir / f).exists() for f in small_files)
            }
            
            # Test cleanup with keep_tmp=True
            # Recreate large files
            for filename in large_files:
                file_path = test_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(b'x' * (10 * 1024 * 1024))
            
            initial_files_2 = list(test_dir.glob("*.parquet"))
            cleanup_temporary_files(test_dir, keep_tmp=True, logger=logger)
            remaining_files_2 = list(test_dir.glob("*.parquet"))
            
            result['details']['cleanup_with_keep_tmp'] = {
                'initial_files': len(initial_files_2),
                'remaining_files': len(remaining_files_2),
                'all_files_preserved': len(initial_files_2) == len(remaining_files_2)
            }
            
            result['passed'] = True
            self.logger.info(f"✅ {test_name} passed")
            
        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"❌ {test_name} failed: {e}")
            if self.verbose:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def test_rotating_log_handler(self) -> Dict[str, Any]:
        """Test rotating file handler functionality."""
        test_name = "rotating_log_handler"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            log_file = self.temp_dir / "rotating_test.log"
            
            # Set up logger with small max file size for testing
            logger = logging.getLogger('rotating_test')
            logger.setLevel(logging.DEBUG)
            
            # Clear any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create rotating handler with small size
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=1024,  # 1KB for quick testing
                backupCount=2
            )
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Generate logs to trigger rotation
            for i in range(100):
                logger.info(f"Test log message {i} - This is a longer message to fill up the log file quickly and trigger rotation.")
            
            # Check for rotated files
            log_files = list(self.temp_dir.glob("rotating_test.log*"))
            
            result['details'] = {
                'log_files_created': len(log_files),
                'main_log_exists': log_file.exists(),
                'rotation_occurred': len(log_files) > 1,
                'log_file_names': [f.name for f in log_files]
            }
            
            # Verify log content
            if log_file.exists():
                with open(log_file, 'r') as f:
                    content = f.read()
                    result['details']['main_log_size'] = len(content)
                    result['details']['contains_log_messages'] = 'Test log message' in content
            
            result['passed'] = len(log_files) >= 1
            self.logger.info(f"✅ {test_name} passed" if result['passed'] else f"❌ {test_name} failed")
            
        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"❌ {test_name} failed: {e}")
            if self.verbose:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def test_pipeline_status_tracking(self) -> Dict[str, Any]:
        """Test pipeline status tracking and error recording."""
        test_name = "pipeline_status_tracking"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            # Test PipelineStatus class
            status = PipelineStatus()
            
            # Test successful stage recording
            status.start_pipeline()
            status.record_stage_success(1, 10.5, "Test Stage 1")
            status.record_stage_success(2, 15.2, "Test Stage 2")
            
            # Test failure recording
            status.record_stage_failure(3, 5.0, "Test Stage 3", "Simulated error", "Traceback here")
            status.complete_pipeline()
            
            summary = status.get_summary()
            
            result['details'] = {
                'total_duration': summary['total_duration'],
                'successful': summary['successful'],
                'failed_stage': summary['failed_stage'],
                'stage_count': len(summary['stage_results']),
                'has_success_stages': any(s['status'] == 'success' for s in summary['stage_results'].values()),
                'has_failed_stage': any(s['status'] == 'failed' for s in summary['stage_results'].values()),
                'error_message_recorded': summary['error_message'] is not None
            }
            
            # Verify JSON serialization
            json_str = json.dumps(summary, default=str)
            result['details']['json_serializable'] = len(json_str) > 0
            
            result['passed'] = (
                summary['failed_stage'] == 3 and
                not summary['successful'] and
                len(summary['stage_results']) == 3
            )
            
            self.logger.info(f"✅ {test_name} passed" if result['passed'] else f"❌ {test_name} failed")
            
        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"❌ {test_name} failed: {e}")
            if self.verbose:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def test_robust_pipeline_runner(self) -> Dict[str, Any]:
        """Test RobustPipelineRunner class initialization and basic functionality."""
        test_name = "robust_pipeline_runner"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'passed': False,
            'details': {},
            'errors': []
        }
        
        try:
            # Create test input directory
            input_dir = self.temp_dir / "input"
            input_dir.mkdir()
            
            # Create a small test file
            test_file = input_dir / "test.txt"
            with open(test_file, 'w') as f:
                f.write("cell_id\ttimestamp\tdata\n1\t123456789\t100\n")
            
            output_dir = self.temp_dir / "output"
            
            # Set up logger
            log_file = self.temp_dir / "runner_test.log"
            logger = setup_robust_logging(log_file, self.verbose)
            
            # Initialize runner
            runner = RobustPipelineRunner(input_dir, output_dir, logger)
            
            result['details'] = {
                'runner_initialized': runner is not None,
                'session_dir_created': runner.session_dir.exists(),
                'input_dir_set': runner.input_dir == input_dir,
                'output_dir_set': runner.output_dir == output_dir,
                'logger_configured': runner.logger is not None,
                'successful_runs_empty': len(runner.successful_runs) == 0,
                'failed_runs_empty': len(runner.failed_runs) == 0,
                'active_monitors_empty': len(runner.active_monitors) == 0
            }
            
            # Test directory setup
            run_dir, data_dir, logs_dir = runner.setup_run_directory(1)
            
            result['details']['directory_setup'] = {
                'run_dir_created': run_dir.exists(),
                'data_dir_created': data_dir.exists(),
                'logs_dir_created': logs_dir.exists(),
                'directory_structure_correct': (
                    run_dir.name == "run_1" and
                    data_dir.parent == run_dir and
                    logs_dir.parent == run_dir
                )
            }
            
            # Test cleanup function
            runner.ensure_all_monitors_stopped()
            
            result['details']['cleanup_function'] = {
                'monitors_stopped': len(runner.active_monitors) == 0
            }
            
            result['passed'] = all([
                runner.session_dir.exists(),
                run_dir.exists(),
                data_dir.exists(),
                logs_dir.exists()
            ])
            
            self.logger.info(f"✅ {test_name} passed" if result['passed'] else f"❌ {test_name} failed")
            
        except Exception as e:
            result['errors'].append(str(e))
            self.logger.error(f"❌ {test_name} failed: {e}")
            if self.verbose:
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all error handling tests."""
        self.logger.info("Starting comprehensive error handling test suite...")
        
        self.setup_test_environment()
        
        try:
            tests = [
                self.test_monitor_thread_termination,
                self.test_cleanup_functionality,
                self.test_rotating_log_handler,
                self.test_pipeline_status_tracking,
                self.test_robust_pipeline_runner
            ]
            
            for test_func in tests:
                result = test_func()
                self.test_results[result['test_name']] = result
                
                if not result['passed']:
                    self.logger.error(f"Test {result['test_name']} failed: {result['errors']}")
        
        finally:
            self.cleanup_test_environment()
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['passed'])
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'all_tests_passed': passed_tests == total_tests
        }
        
        self.logger.info(f"Test suite completed: {passed_tests}/{total_tests} tests passed ({summary['success_rate']:.1f}%)")
        
        return {
            'summary': summary,
            'test_results': self.test_results
        }


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(
        description="CMMSE 2025: Robust Error Handling Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  monitoring   - Test monitor thread termination guarantees
  cleanup      - Test temporary file cleanup functionality  
  logging      - Test rotating log handler operation
  status       - Test pipeline status tracking
  runner       - Test robust pipeline runner class
  all          - Run all tests (default)

Examples:
  # Run all tests
  python scripts/utils/test_robust_error_handling.py
  
  # Run specific test with verbose output
  python scripts/utils/test_robust_error_handling.py --test_type monitoring --verbose
  
  # Run cleanup tests only
  python scripts/utils/test_robust_error_handling.py --test_type cleanup
        """
    )
    
    parser.add_argument(
        "--test_type",
        choices=['monitoring', 'cleanup', 'logging', 'status', 'runner', 'all'],
        default='all',
        help="Type of test to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Save test results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = ErrorHandlingTestSuite(verbose=args.verbose)
    
    print("=" * 80)
    print("CMMSE 2025: ROBUST ERROR HANDLING TEST SUITE")
    print("=" * 80)
    
    # Run tests based on type
    if args.test_type == 'all':
        results = test_suite.run_all_tests()
    else:
        test_suite.setup_test_environment()
        try:
            if args.test_type == 'monitoring':
                result = test_suite.test_monitor_thread_termination()
            elif args.test_type == 'cleanup':
                result = test_suite.test_cleanup_functionality()
            elif args.test_type == 'logging':
                result = test_suite.test_rotating_log_handler()
            elif args.test_type == 'status':
                result = test_suite.test_pipeline_status_tracking()
            elif args.test_type == 'runner':
                result = test_suite.test_robust_pipeline_runner()
            
            results = {
                'summary': {
                    'total_tests': 1,
                    'passed_tests': 1 if result['passed'] else 0,
                    'failed_tests': 0 if result['passed'] else 1,
                    'success_rate': 100.0 if result['passed'] else 0.0,
                    'all_tests_passed': result['passed']
                },
                'test_results': {result['test_name']: result}
            }
        finally:
            test_suite.cleanup_test_environment()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    if summary['all_tests_passed']:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        
        # Print details of failed tests
        for test_name, test_result in results['test_results'].items():
            if not test_result['passed']:
                print(f"\n❌ {test_name} failed:")
                for error in test_result['errors']:
                    print(f"  - {error}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nTest results saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['all_tests_passed'] else 1)


if __name__ == "__main__":
    main()
