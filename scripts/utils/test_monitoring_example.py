#!/usr/bin/env python3
"""
Example Usage of System Monitoring Utilities

This script demonstrates how to use the monitoring module for academic research,
specifically for the CMMSE 2025 conference paper on Apple Silicon optimization.
"""

import time
import numpy as np
from monitoring import start_monitor, stop_monitor, get_system_info


def simulate_cpu_intensive_task(duration=10):
    """Simulate a CPU-intensive task for monitoring demonstration."""
    print(f"Simulating CPU-intensive task for {duration} seconds...")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # CPU-intensive operation
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
        time.sleep(0.1)  # Small pause to avoid 100% CPU


def simulate_memory_intensive_task(duration=10):
    """Simulate a memory-intensive task for monitoring demonstration."""
    print(f"Simulating memory-intensive task for {duration} seconds...")
    
    large_arrays = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Memory-intensive operation
        large_arrays.append(np.random.rand(100000))
        if len(large_arrays) > 100:
            large_arrays.pop(0)  # Keep memory usage manageable
        time.sleep(0.5)


def main():
    """Main demonstration function."""
    print("System Monitoring Utilities Demo")
    print("================================")
    
    # Display system information
    print("\n1. System Information:")
    info = get_system_info()
    print(f"   Platform: {info['system']['platform']}")
    print(f"   Machine: {info['system']['machine']}")
    print(f"   Apple Silicon: {info['system']['is_apple_silicon']}")
    print(f"   CPU Cores: {info['cpu']['cpu_count']}")
    print(f"   Memory Total: {info['memory']['total'] / (1024**3):.1f} GB")
    
    # Demonstrate basic monitoring
    print("\n2. Basic Monitoring Demo (10 seconds, CPU-intensive):")
    monitor = start_monitor(interval=1.0)
    simulate_cpu_intensive_task(duration=10)
    df_cpu = stop_monitor(monitor)
    
    print(f"   Samples collected: {len(df_cpu)}")
    print(f"   Average CPU usage: {df_cpu['cpu_percent'].mean():.1f}%")
    print(f"   Peak memory RSS: {df_cpu['memory_rss'].max() / (1024**2):.1f} MB")
    
    # Demonstrate memory monitoring
    print("\n3. Memory Monitoring Demo (10 seconds, memory-intensive):")
    monitor = start_monitor(interval=1.0)
    simulate_memory_intensive_task(duration=10)
    df_memory = stop_monitor(monitor)
    
    print(f"   Samples collected: {len(df_memory)}")
    print(f"   Average memory usage: {df_memory['memory_percent'].mean():.2f}%")
    print(f"   Peak memory via tracemalloc: {df_memory['peak_memory_tracemalloc'].max() / (1024**2):.1f} MB")
    
    # Demonstrate high-frequency monitoring
    print("\n4. High-Frequency Monitoring Demo (5 seconds, 0.1s interval):")
    monitor = start_monitor(interval=0.1, enable_apple_metrics=True)
    time.sleep(5)
    df_hf = stop_monitor(monitor)
    
    print(f"   Samples collected: {len(df_hf)}")
    print(f"   Sampling accuracy: {len(df_hf) / 50 * 100:.1f}%")
    
    # Apple Silicon specific metrics (if available)
    if info['system']['is_apple_silicon']:
        print("\n5. Apple Silicon Specific Metrics:")
        apple_columns = [col for col in df_hf.columns if 'temperature' in col or 'frequency' in col]
        if apple_columns:
            for col in apple_columns:
                non_null_values = df_hf[col].dropna()
                if len(non_null_values) > 0:
                    print(f"   {col}: {non_null_values.mean():.1f} (avg)")
        else:
            print("   Apple-specific metrics require root access or may not be available")
    
    print("\n6. Summary Statistics:")
    print("   CPU Intensive Task:")
    print(f"     - Average CPU: {df_cpu['cpu_percent'].mean():.1f}%")
    print(f"     - Peak CPU: {df_cpu['cpu_percent'].max():.1f}%")
    print(f"     - Average system CPU: {df_cpu['system_cpu_percent'].mean():.1f}%")
    
    print("   Memory Intensive Task:")
    print(f"     - Average memory: {df_memory['memory_percent'].mean():.2f}%")
    print(f"     - Peak memory RSS: {df_memory['memory_rss'].max() / (1024**2):.1f} MB")
    print(f"     - Peak memory tracemalloc: {df_memory['peak_memory_tracemalloc'].max() / (1024**2):.1f} MB")
    
    print("\nDemo completed successfully!")
    print("This monitoring system is ready for integration with the anomaly detection pipeline.")


if __name__ == "__main__":
    main()
