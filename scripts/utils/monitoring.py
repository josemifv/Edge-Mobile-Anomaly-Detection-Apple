#!/usr/bin/env python3
"""
Low-Overhead System Monitoring Utilities

This module provides comprehensive system monitoring capabilities optimized for Apple Silicon
devices, including CPU usage, memory consumption, temperature, and frequency tracking.

Features:
- Background monitoring with configurable sampling rates (default 1 Hz)
- Peak memory tracking via tracemalloc and resource.getrusage
- Apple-specific temperature and frequency monitoring via powermetrics and sysctl
- Non-root compatible operation with graceful fallbacks
- Pandas DataFrame output for analysis and visualization
- Thread-safe operation with proper cleanup

Usage:
    from scripts.utils.monitoring import start_monitor, stop_monitor
    
    # Start monitoring
    monitor = start_monitor()
    
    # Your code here...
    
    # Stop monitoring and get results
    df = stop_monitor(monitor)
    print(df.describe())

Author: José Miguel Franco-Valiente
Created: December 2024
"""

import subprocess
import threading
import time
import tracemalloc
import resource
import platform
import psutil
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import warnings
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Low-overhead system monitoring class with Apple Silicon optimizations.
    
    This class provides comprehensive system monitoring including:
    - CPU percentage and core utilization
    - Memory usage (RSS, VMS, peak tracking)
    - Apple-specific temperature and frequency monitoring
    - Background thread-based sampling for minimal overhead
    """
    
    def __init__(self, interval: float = 1.0, enable_apple_metrics: bool = True):
        """
        Initialize the system monitor.
        
        Args:
            interval: Sampling interval in seconds (default: 1.0 Hz)
            enable_apple_metrics: Enable Apple-specific monitoring (temperature, frequency)
        """
        self.interval = interval
        self.enable_apple_metrics = enable_apple_metrics
        self.is_running = False
        self.monitor_thread = None
        self.data_lock = threading.Lock()
        self.samples = []
        
        # System information
        self.is_apple_silicon = self._detect_apple_silicon()
        self.process = psutil.Process()
        
        # Peak memory tracking
        self.peak_memory_tracemalloc = 0
        self.peak_memory_rusage = 0
        
        # Start tracemalloc for peak memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
        logger.info(f"SystemMonitor initialized - Apple Silicon: {self.is_apple_silicon}, Interval: {interval}s")
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon (M-series) processor."""
        try:
            return platform.machine() == 'arm64' and platform.system() == 'Darwin'
        except Exception:
            return False
    
    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU usage metrics."""
        try:
            # Get overall CPU percentage
            cpu_percent = self.process.cpu_percent(interval=None)
            
            # Get system-wide CPU usage
            system_cpu = psutil.cpu_percent(interval=None)
            
            # Get per-core usage
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            metrics = {
                'cpu_percent': cpu_percent,
                'system_cpu_percent': system_cpu,
                'cpu_cores_avg': sum(per_core) / len(per_core) if per_core else 0.0
            }
            
            # Add per-core metrics (up to 8 cores for common Apple Silicon)
            for i, core_usage in enumerate(per_core[:8]):
                metrics[f'cpu_core_{i}'] = core_usage
                
            return metrics
        except Exception as e:
            logger.warning(f"Error getting CPU metrics: {e}")
            return {'cpu_percent': 0.0, 'system_cpu_percent': 0.0, 'cpu_cores_avg': 0.0}
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics including RSS, VMS, and peak tracking."""
        try:
            # Process memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # System memory info
            system_memory = psutil.virtual_memory()
            
            # Peak memory from tracemalloc
            current_tracemalloc, peak_tracemalloc = tracemalloc.get_traced_memory()
            self.peak_memory_tracemalloc = max(self.peak_memory_tracemalloc, peak_tracemalloc)
            
            # Peak memory from resource.getrusage
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            peak_rusage = rusage.ru_maxrss
            # On macOS, ru_maxrss is in bytes, on Linux it's in KB
            if platform.system() == 'Darwin':
                peak_rusage_bytes = peak_rusage
            else:
                peak_rusage_bytes = peak_rusage * 1024
            
            self.peak_memory_rusage = max(self.peak_memory_rusage, peak_rusage_bytes)
            
            return {
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms,
                'memory_percent': memory_percent,
                'system_memory_percent': system_memory.percent,
                'system_memory_available': system_memory.available,
                'peak_memory_tracemalloc': self.peak_memory_tracemalloc,
                'peak_memory_rusage': self.peak_memory_rusage,
                'current_memory_tracemalloc': current_tracemalloc
            }
        except Exception as e:
            logger.warning(f"Error getting memory metrics: {e}")
            return {
                'memory_rss': 0, 'memory_vms': 0, 'memory_percent': 0.0,
                'system_memory_percent': 0.0, 'system_memory_available': 0,
                'peak_memory_tracemalloc': 0, 'peak_memory_rusage': 0,
                'current_memory_tracemalloc': 0
            }
    
    def _get_apple_metrics(self) -> Dict[str, Optional[float]]:
        """Get Apple-specific metrics (temperature, frequency) with non-root fallbacks."""
        if not self.enable_apple_metrics or not self.is_apple_silicon:
            return {}
        
        metrics = {}
        
        # Try to get temperature via powermetrics (requires root)
        try:
            result = subprocess.run(
                ['powermetrics', '--samplers', 'smc', '--sample-count', '1', '--format', 'plist'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse plist output for temperature data
                temp_data = self._parse_powermetrics_temperature(result.stdout)
                metrics.update(temp_data)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"powermetrics not available (normal for non-root): {e}")
        
        # Try to get temperature via sysctl (fallback)
        try:
            result = subprocess.run(
                ['sysctl', '-a'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                temp_data = self._parse_sysctl_temperature(result.stdout)
                metrics.update(temp_data)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"sysctl temperature parsing failed: {e}")
        
        # Try to get CPU frequency information
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.cpufrequency_max', 'hw.cpufrequency'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                freq_data = self._parse_frequency_data(result.stdout)
                metrics.update(freq_data)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"CPU frequency data not available: {e}")
        
        return metrics
    
    def _parse_powermetrics_temperature(self, output: str) -> Dict[str, float]:
        """Parse powermetrics plist output for temperature data."""
        try:
            # This is a simplified parser - in practice, you'd want to use plistlib
            # for proper plist parsing, but we'll do basic text parsing for now
            temps = {}
            lines = output.split('\n')
            for line in lines:
                if 'CPU die temperature' in line:
                    # Extract temperature value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            temps['cpu_die_temperature'] = float(part)
                            break
            return temps
        except Exception as e:
            logger.debug(f"Error parsing powermetrics temperature: {e}")
            return {}
    
    def _parse_sysctl_temperature(self, output: str) -> Dict[str, float]:
        """Parse sysctl output for temperature data."""
        try:
            temps = {}
            lines = output.split('\n')
            for line in lines:
                if 'temperature' in line.lower() and 'cpu' in line.lower():
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            # Extract numeric value
                            temp_str = parts[1].strip()
                            # Look for numeric value (handle various formats)
                            import re
                            match = re.search(r'(\d+\.?\d*)', temp_str)
                            if match:
                                temp_value = float(match.group(1))
                                temps['cpu_temperature_sysctl'] = temp_value
                                break
                        except ValueError:
                            continue
            return temps
        except Exception as e:
            logger.debug(f"Error parsing sysctl temperature: {e}")
            return {}
    
    def _parse_frequency_data(self, output: str) -> Dict[str, float]:
        """Parse CPU frequency data from sysctl output."""
        try:
            freqs = {}
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip().isdigit():
                    freq_hz = int(line.strip())
                    # Convert to MHz for readability
                    freq_mhz = freq_hz / 1_000_000
                    if 'cpu_frequency_max' not in freqs:
                        freqs['cpu_frequency_max_mhz'] = freq_mhz
                    else:
                        freqs['cpu_frequency_current_mhz'] = freq_mhz
            return freqs
        except Exception as e:
            logger.debug(f"Error parsing frequency data: {e}")
            return {}
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                timestamp = datetime.now()
                
                # Collect all metrics
                sample = {'timestamp': timestamp}
                
                # CPU metrics
                sample.update(self._get_cpu_metrics())
                
                # Memory metrics
                sample.update(self._get_memory_metrics())
                
                # Apple-specific metrics
                if self.enable_apple_metrics:
                    sample.update(self._get_apple_metrics())
                
                # Thread-safe sample storage
                with self.data_lock:
                    self.samples.append(sample)
                
                # Sleep until next sample
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Continue monitoring despite errors
                time.sleep(self.interval)
    
    def start(self):
        """Start background monitoring."""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("System monitoring started")
    
    def stop(self) -> pd.DataFrame:
        """
        Stop monitoring and return collected data as DataFrame.
        
        Returns:
            pd.DataFrame: Timestamped system metrics data
        """
        if not self.is_running:
            logger.warning("Monitor is not running")
            return pd.DataFrame()
        
        self.is_running = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Convert samples to DataFrame
        with self.data_lock:
            df = pd.DataFrame(self.samples)
        
        logger.info(f"System monitoring stopped. Collected {len(df)} samples.")
        
        return df
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics without starting continuous monitoring."""
        sample = {}
        sample.update(self._get_cpu_metrics())
        sample.update(self._get_memory_metrics())
        
        if self.enable_apple_metrics:
            sample.update(self._get_apple_metrics())
        
        return sample


# Convenience functions for easy usage
def start_monitor(interval: float = 1.0, enable_apple_metrics: bool = True) -> SystemMonitor:
    """
    Start system monitoring with specified parameters.
    
    Args:
        interval: Sampling interval in seconds (default: 1.0 Hz)
        enable_apple_metrics: Enable Apple-specific monitoring
    
    Returns:
        SystemMonitor: Running monitor instance
    """
    monitor = SystemMonitor(interval=interval, enable_apple_metrics=enable_apple_metrics)
    monitor.start()
    return monitor


def stop_monitor(monitor: SystemMonitor) -> pd.DataFrame:
    """
    Stop monitoring and return results.
    
    Args:
        monitor: SystemMonitor instance to stop
    
    Returns:
        pd.DataFrame: Collected monitoring data
    """
    return monitor.stop()


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict containing system specs, capabilities, and current metrics
    """
    monitor = SystemMonitor(enable_apple_metrics=True)
    
    info = {
        'system': {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'is_apple_silicon': monitor.is_apple_silicon,
        },
        'cpu': {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent,
        },
        'current_metrics': monitor.get_current_metrics()
    }
    
    return info


def benchmark_monitoring_overhead(duration: float = 10.0, intervals: List[float] = None) -> pd.DataFrame:
    """
    Benchmark the overhead of monitoring at different sampling rates.
    
    Args:
        duration: Test duration in seconds
        intervals: List of sampling intervals to test (default: [0.1, 0.5, 1.0, 2.0])
    
    Returns:
        pd.DataFrame: Benchmark results showing overhead at different sampling rates
    """
    if intervals is None:
        intervals = [0.1, 0.5, 1.0, 2.0]
    
    results = []
    
    for interval in intervals:
        logger.info(f"Testing monitoring overhead at {interval}s interval")
        
        # Measure baseline (no monitoring)
        start_time = time.perf_counter()
        time.sleep(duration)
        baseline_time = time.perf_counter() - start_time
        
        # Measure with monitoring
        monitor = start_monitor(interval=interval, enable_apple_metrics=True)
        start_time = time.perf_counter()
        time.sleep(duration)
        monitored_time = time.perf_counter() - start_time
        df = stop_monitor(monitor)
        
        overhead = monitored_time - baseline_time
        overhead_percent = (overhead / baseline_time) * 100
        
        results.append({
            'interval': interval,
            'samples_collected': len(df),
            'baseline_time': baseline_time,
            'monitored_time': monitored_time,
            'overhead_seconds': overhead,
            'overhead_percent': overhead_percent,
            'expected_samples': duration / interval,
            'sample_accuracy': len(df) / (duration / interval) * 100
        })
    
    return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System Monitoring Utilities")
    parser.add_argument('--test', action='store_true', help='Run test monitoring session')
    parser.add_argument('--duration', type=float, default=10.0, help='Test duration in seconds')
    parser.add_argument('--interval', type=float, default=1.0, help='Sampling interval in seconds')
    parser.add_argument('--benchmark', action='store_true', help='Run monitoring overhead benchmark')
    parser.add_argument('--system-info', action='store_true', help='Display system information')
    parser.add_argument('--output', type=str, help='Output file for results (CSV format)')
    
    args = parser.parse_args()
    
    if args.system_info:
        print("System Information:")
        print("==================")
        info = get_system_info()
        print(json.dumps(info, indent=2, default=str))
    
    if args.benchmark:
        print(f"Running monitoring overhead benchmark...")
        benchmark_df = benchmark_monitoring_overhead(duration=args.duration)
        print("\nBenchmark Results:")
        print(benchmark_df.to_string(index=False))
        
        if args.output:
            benchmark_df.to_csv(args.output.replace('.csv', '_benchmark.csv'), index=False)
            print(f"Benchmark results saved to {args.output.replace('.csv', '_benchmark.csv')}")
    
    if args.test:
        print(f"Starting {args.duration}s monitoring test with {args.interval}s interval...")
        
        # Start monitoring
        monitor = start_monitor(interval=args.interval)
        
        # Simulate some work
        time.sleep(args.duration)
        
        # Stop monitoring and get results
        df = stop_monitor(monitor)
        
        print(f"\nCollected {len(df)} samples")
        print("\nSample Statistics:")
        print(df.describe())
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        
        # Display recent samples
        print(f"\nLast 5 samples:")
        print(df.tail().to_string(index=False))
