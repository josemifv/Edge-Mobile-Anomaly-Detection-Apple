import pandas as pd
import os
import argparse
import time
import multiprocessing
import psutil
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import gc
import numpy as np
import mmap
import io
from functools import partial
import queue
from threading import Thread

# --- Level 3: Aggressive Performance Configuration ---
ULTRA_OPTIMIZED_DTYPES = {
    'square_id': 'uint16',      # 2 bytes instead of 8
    'timestamp_ms': 'int64',    # Keep precision
    'country_code': 'uint8',    # 1 byte instead of 8
    'sms_in': 'float32',        # 4 bytes instead of 8
    'sms_out': 'float32',
    'call_in': 'float32',
    'call_out': 'float32',
    'internet_activity': 'float32'
}

# Memory-optimized column selection
ESSENTIAL_COLUMNS = [
    'square_id', 'timestamp_ms', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_activity'
]

ALL_COLUMNS = [
    'square_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_activity'
]

# Advanced buffer configurations
ADVANCED_BUFFER_CONFIG = {
    'small': {'buffer_size': 128 * 1024, 'read_ahead': 2},
    'medium': {'buffer_size': 512 * 1024, 'read_ahead': 4},
    'large': {'buffer_size': 2048 * 1024, 'read_ahead': 8},
}

# --- Advanced Resource and Performance Manager ---
class AggressiveResourceManager:
    def __init__(self, target_memory_usage=0.75, enable_profiling=True):
        self.target_memory_usage = target_memory_usage
        self.total_memory = psutil.virtual_memory().total
        self.target_memory_bytes = int(self.total_memory * target_memory_usage)
        self.enable_profiling = enable_profiling
        self.performance_history = []
        
    def get_detailed_usage(self):
        cpu_times = psutil.cpu_times_percent(interval=0)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=0),
            'cpu_user': cpu_times.user,
            'cpu_system': cpu_times.system,
            'cpu_idle': cpu_times.idle,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_cached_gb': getattr(memory, 'cached', 0) / (1024**3),
            'disk_read_mb': disk_io.read_bytes / (1024**2),
            'disk_write_mb': disk_io.write_bytes / (1024**2),
            'timestamp': time.time()
        }
    
    def log_performance(self, operation, duration, rows_processed):
        if self.enable_profiling:
            metrics = self.get_detailed_usage()
            metrics.update({
                'operation': operation,
                'duration': duration,
                'rows_processed': rows_processed,
                'rows_per_second': rows_processed / duration if duration > 0 else 0
            })
            self.performance_history.append(metrics)
    
    def get_optimal_parallelism(self, num_files, file_sizes):
        """Dynamic parallelism based on system state and file characteristics."""
        current_usage = self.get_detailed_usage()
        cpu_count = os.cpu_count() or 1
        
        # Base parallelism on current system load
        if current_usage['cpu_percent'] > 90:
            base_workers = max(1, cpu_count // 4)
        elif current_usage['cpu_percent'] > 70:
            base_workers = max(1, cpu_count // 2)
        elif current_usage['cpu_percent'] > 40:
            base_workers = cpu_count - 1
        else:
            base_workers = cpu_count + 2  # I/O bound optimization
        
        # Adjust based on memory availability
        available_gb = current_usage['memory_available_gb']
        if available_gb < 4:
            memory_factor = 0.5
        elif available_gb < 8:
            memory_factor = 0.75
        else:
            memory_factor = 1.0
        
        # Consider file sizes for load balancing
        avg_file_size_mb = np.mean(file_sizes) / (1024**2)
        if avg_file_size_mb > 500:  # Large files
            size_factor = 0.8
        elif avg_file_size_mb > 200:  # Medium files
            size_factor = 1.0
        else:  # Small files
            size_factor = 1.2
        
        optimal_workers = int(base_workers * memory_factor * size_factor)
        return min(max(1, optimal_workers), num_files, cpu_count * 2)
    
    def get_adaptive_chunk_size(self, file_size_bytes, target_memory_mb=100):
        """Calculate optimal chunk size based on file size and available memory."""
        available_memory = psutil.virtual_memory().available
        target_memory_bytes = target_memory_mb * 1024 * 1024
        
        # Estimate bytes per row (optimized dtypes)
        estimated_bytes_per_row = 48  # Conservative estimate
        
        # Calculate chunk size to stay within memory target
        max_rows_per_chunk = min(
            target_memory_bytes // estimated_bytes_per_row,
            available_memory // (estimated_bytes_per_row * 8)  # Safety factor
        )
        
        # Adaptive sizing based on file size
        file_size_mb = file_size_bytes / (1024**2)
        if file_size_mb < 50:
            chunk_multiplier = 0.5
        elif file_size_mb < 200:
            chunk_multiplier = 1.0
        elif file_size_mb < 500:
            chunk_multiplier = 1.5
        else:
            chunk_multiplier = 2.0
        
        optimal_chunk = int(max_rows_per_chunk * chunk_multiplier)
        return max(10_000, min(optimal_chunk, 5_000_000))  # Reasonable bounds

# --- Memory-Mapped File Reader ---
class MemoryMappedReader:
    def __init__(self, file_path, buffer_config):
        self.file_path = file_path
        self.buffer_config = buffer_config
        self.file_size = os.path.getsize(file_path)
        
    def read_optimized(self, chunk_size=None):
        """Memory-mapped reading with optimal buffer management."""
        try:
            with open(self.file_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Read entire file into memory-mapped buffer
                    content = mmapped_file.read()
                    
                    # Convert to string buffer for pandas
                    text_buffer = io.StringIO(content.decode('utf-8'))
                    return text_buffer
        except Exception as e:
            print(f"Memory mapping failed for {self.file_path}: {e}")
            return None

# --- Async File Processing Pipeline ---
async def async_file_processor(
    file_paths,
    processing_function,
    max_concurrent=4,
    **kwargs
):
    """Asynchronous file processing pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_file(file_path):
        async with semaphore:
            loop = asyncio.get_event_loop()
            # Run CPU-intensive processing in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor,
                    processing_function,
                    file_path,
                    **kwargs
                )
                return result
    
    # Process all files concurrently
    tasks = [process_single_file(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return successful results
    successful_results = [
        result for result in results 
        if not isinstance(result, Exception) and result is not None
    ]
    
    return successful_results

# --- Ultra-Optimized File Loading ---
def load_file_aggressive_optimization(
    file_path,
    chunk_size=None,
    convert_timestamp=True,
    reorder_cols=True,
    skip_country_code=True,
    use_memory_mapping=True,
    resource_manager=None
):
    """
    Level 3: Aggressive optimization file loading.
    """
    print(f"\n--- [Level 3] Processing: {Path(file_path).name} ---")
    start_time = time.perf_counter()
    
    # File analysis
    file_stat = os.stat(file_path)
    file_size_mb = file_stat.st_size / (1024**2)
    
    # Determine file category and buffer configuration
    if file_size_mb < 100:
        category = 'small'
    elif file_size_mb < 400:
        category = 'medium'
    else:
        category = 'large'
    
    buffer_config = ADVANCED_BUFFER_CONFIG[category]
    print(f"File: {file_size_mb:.1f} MB ({category}), "
          f"Buffer: {buffer_config['buffer_size']//1024} KB")
    
    # Column and dtype optimization
    if skip_country_code:
        columns_to_use = ESSENTIAL_COLUMNS
        dtype_mapping = {k: v for k, v in ULTRA_OPTIMIZED_DTYPES.items() 
                        if k in ESSENTIAL_COLUMNS}
    else:
        columns_to_use = ALL_COLUMNS
        dtype_mapping = ULTRA_OPTIMIZED_DTYPES
    
    # Ultra-optimized read parameters
    read_params = {
        "sep": r'\s+',
        "header": None,
        "names": ALL_COLUMNS,
        "usecols": columns_to_use,
        "dtype": dtype_mapping,
        "engine": 'c',
        "memory_map": use_memory_mapping,
        "na_filter": True,  # Keep for data integrity
        "low_memory": False,
        "buffer_size": buffer_config['buffer_size'],
        "float_precision": 'round_trip',  # Faster parsing
    }
    
    try:
        # Adaptive chunk sizing
        if chunk_size is None and resource_manager:
            chunk_size = resource_manager.get_adaptive_chunk_size(file_stat.st_size)
            auto_chunk = True
        else:
            auto_chunk = False
        
        # Memory-mapped reading for large files
        if use_memory_mapping and file_size_mb > 200:
            print(f"Using memory-mapped reading")
            mm_reader = MemoryMappedReader(file_path, buffer_config)
            text_buffer = mm_reader.read_optimized()
            
            if text_buffer:
                if chunk_size and chunk_size > 0:
                    chunks = []
                    for chunk in pd.read_csv(text_buffer, chunksize=chunk_size, **read_params):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                    del chunks  # Immediate cleanup
                else:
                    df = pd.read_csv(text_buffer, **read_params)
            else:
                # Fallback to standard reading
                df = pd.read_csv(file_path, **read_params)
        else:
            # Standard optimized reading
            if chunk_size and chunk_size > 0:
                if auto_chunk:
                    print(f"Adaptive chunk: {chunk_size:,} rows")
                
                chunk_list = []
                chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, **read_params)
                
                for chunk_num, chunk in enumerate(chunk_reader):
                    chunk_list.append(chunk)
                    
                    # Aggressive memory management
                    if chunk_num % 10 == 0:  # Every 10 chunks
                        gc.collect()
                
                df = pd.concat(chunk_list, ignore_index=True)
                del chunk_list
                gc.collect()
            else:
                df = pd.read_csv(file_path, **read_params)
        
        # Immediate memory usage feedback
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"Loaded: {len(df):,} rows, {df.shape[1]} cols, "
              f"Memory: {memory_usage_mb:.1f} MB")
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Ultra-fast preprocessing
    if convert_timestamp and 'timestamp_ms' in df.columns:
        # Vectorized operations with error handling
        print("Converting timestamps (vectorized)...")
        start_ts = time.perf_counter()
        
        # Use pandas optimized datetime conversion
        df['timestamp'] = pd.to_datetime(
            df['timestamp_ms'], 
            unit='ms', 
            errors='coerce',
            cache=True  # Cache conversion for repeated values
        )
        
        null_count = df['timestamp'].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} timestamp failures ({null_count/len(df)*100:.2f}%)")
        
        df.drop(columns=['timestamp_ms'], inplace=True)
        
        ts_duration = time.perf_counter() - start_ts
        print(f"Timestamp conversion: {ts_duration:.3f}s")
    
    # Efficient column reordering
    if reorder_cols and 'timestamp' in df.columns:
        column_order = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df.reindex(columns=column_order)
    
    # Final optimization pass
    start_opt = time.perf_counter()
    
    # Memory optimization for object columns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Try to convert to more efficient types
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    opt_duration = time.perf_counter() - start_opt
    
    # Performance metrics
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    rows_per_second = len(df) / total_duration if total_duration > 0 else 0
    
    print(f"Optimization: {opt_duration:.3f}s")
    print(f"Total: {total_duration:.4f}s ({rows_per_second:,.0f} rows/sec)")
    
    # Log performance if resource manager available
    if resource_manager:
        resource_manager.log_performance(
            'file_load', total_duration, len(df)
        )
    
    return df

# --- Aggressive Pipeline Processing ---
def process_directory_aggressive(
    directory_path,
    chunk_size=None,
    convert_timestamp=True,
    reorder_cols=True,
    max_workers=None,
    skip_country_code=True,
    use_memory_mapping=True,
    enable_async_processing=True
):
    """
    Level 3: Aggressive optimization directory processing.
    """
    print("=== Level 3: Aggressive Processing Pipeline ===")
    
    # Initialize advanced resource manager
    resource_manager = AggressiveResourceManager(enable_profiling=True)
    
    # File discovery and analysis
    files_to_process = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))
    ]
    
    if not files_to_process:
        print(f"No .txt files found in {directory_path}")
        return [], 0
    
    # Advanced file analysis
    file_info = []
    for file_path in files_to_process:
        stat = os.stat(file_path)
        file_info.append({
            'path': file_path,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024**2)
        })
    
    # Sort by size for optimal load balancing (largest first)
    file_info.sort(key=lambda x: x['size_bytes'], reverse=True)
    files_to_process = [info['path'] for info in file_info]
    file_sizes = [info['size_bytes'] for info in file_info]
    
    total_size_gb = sum(file_sizes) / (1024**3)
    print(f"Found {len(files_to_process)} files, Total: {total_size_gb:.2f} GB")
    print(f"Size range: {min(file_sizes)/(1024**2):.1f} - {max(file_sizes)/(1024**2):.1f} MB")
    
    # System resource analysis
    initial_resources = resource_manager.get_detailed_usage()
    print(f"System state - CPU: {initial_resources['cpu_percent']:.1f}%, "
          f"Memory: {initial_resources['memory_percent']:.1f}%, "
          f"Available: {initial_resources['memory_available_gb']:.1f} GB")
    
    # Dynamic parallelism optimization
    if max_workers is None:
        optimal_workers = resource_manager.get_optimal_parallelism(
            len(files_to_process), file_sizes
        )
    else:
        optimal_workers = max_workers
    
    print(f"Optimal parallelism: {optimal_workers} workers")
    
    start_time = time.perf_counter()
    
    try:
        if enable_async_processing and len(files_to_process) > 2:
            print("Using asynchronous processing pipeline")
            
            # Async processing with controlled concurrency
            processing_func = partial(
                load_file_aggressive_optimization,
                chunk_size=chunk_size,
                convert_timestamp=convert_timestamp,
                reorder_cols=reorder_cols,
                skip_country_code=skip_country_code,
                use_memory_mapping=use_memory_mapping,
                resource_manager=resource_manager
            )
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    async_file_processor(
                        files_to_process,
                        processing_func,
                        max_concurrent=optimal_workers
                    )
                )
            finally:
                loop.close()
        else:
            print("Using standard parallel processing")
            
            # Standard multiprocessing with optimizations
            with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                tasks = [
                    executor.submit(
                        load_file_aggressive_optimization,
                        file_path,
                        chunk_size,
                        convert_timestamp,
                        reorder_cols,
                        skip_country_code,
                        use_memory_mapping,
                        resource_manager
                    )
                    for file_path in files_to_process
                ]
                
                results = []
                for future in tasks:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
        
        # Filter successful results
        successful_dataframes = [df for df in results if df is not None]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance analysis
        successful_count = len(successful_dataframes)
        failed_count = len(files_to_process) - successful_count
        total_rows = sum(len(df) for df in successful_dataframes)
        
        print(f"\n--- Level 3 Aggressive Processing Results ---")
        print(f"Success rate: {successful_count}/{len(files_to_process)} "
              f"({successful_count/len(files_to_process)*100:.1f}%)")
        print(f"Failed files: {failed_count}")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Processing time: {total_time:.4f} seconds")
        print(f"Throughput: {total_rows/total_time:,.0f} rows/second")
        print(f"Data throughput: {total_size_gb/total_time:.2f} GB/second")
        
        # Resource utilization summary
        final_resources = resource_manager.get_detailed_usage()
        print(f"Final system state - CPU: {final_resources['cpu_percent']:.1f}%, "
              f"Memory: {final_resources['memory_percent']:.1f}%")
        
        # Memory efficiency calculation
        total_memory_used = sum(
            df.memory_usage(deep=True).sum() for df in successful_dataframes
        ) / (1024**3)
        print(f"Total DataFrame memory: {total_memory_used:.2f} GB")
        print(f"Memory efficiency: {total_size_gb/total_memory_used:.1f}x compression")
        
        return successful_dataframes, optimal_workers
    
    except Exception as e:
        print(f"Error in aggressive processing: {e}")
        return [], 0

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Level 3: Aggressive performance data ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_path", help="Path to file or directory")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--no_timestamp_conversion", action="store_false", dest="convert_timestamp")
    parser.add_argument("--no_reorder_columns", action="store_false", dest="reorder_cols")
    parser.add_argument("--no_memory_mapping", action="store_false", dest="use_memory_mapping")
    parser.add_argument("--no_async_processing", action="store_false", dest="enable_async_processing")
    parser.add_argument("--include_country_code", action="store_false", dest="skip_country_code")
    parser.add_argument("--output_summary", action="store_true")
    
    args = parser.parse_args()
    
    print("=== Level 3: Aggressive Performance Data Ingestion ===")
    print("Advanced optimizations:")
    print(f"  - Ultra-optimized data types: True")
    print(f"  - Memory mapping: {args.use_memory_mapping}")
    print(f"  - Async processing: {args.enable_async_processing}")
    print(f"  - Column selection: {args.skip_country_code}")
    print(f"  - Adaptive chunking: True")
    print(f"  - Performance profiling: True")
    
    overall_start_time = time.perf_counter()
    
    final_dataframes = []
    num_processes_used = 1
    
    if os.path.isdir(args.input_path):
        print(f"\nProcessing directory: {args.input_path}")
        final_dataframes, num_processes_used = process_directory_aggressive(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            max_workers=args.max_workers,
            skip_country_code=args.skip_country_code,
            use_memory_mapping=args.use_memory_mapping,
            enable_async_processing=args.enable_async_processing
        )
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.txt'):
        print(f"\nProcessing single file: {args.input_path}")
        resource_manager = AggressiveResourceManager()
        df = load_file_aggressive_optimization(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            skip_country_code=args.skip_country_code,
            use_memory_mapping=args.use_memory_mapping,
            resource_manager=resource_manager
        )
        if df is not None:
            final_dataframes.append(df)
    else:
        print(f"Error: '{args.input_path}' is not valid")
        return
    
    overall_end_time = time.perf_counter()
    total_execution_time = overall_end_time - overall_start_time
    
    if not final_dataframes:
        print("No data processed successfully.")
        return
    
    # Final performance analysis
    total_rows = sum(len(df) for df in final_dataframes)
    num_files = len(final_dataframes)
    
    print(f"\n=== Level 3 Final Performance Summary ===")
    print(f"Files successfully processed: {num_files}")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Total execution time: {total_execution_time:.4f} seconds")
    print(f"Overall throughput: {total_rows/total_execution_time:,.0f} rows/second")
    
    # Calculate memory efficiency
    total_memory_mb = sum(
        df.memory_usage(deep=True).sum() for df in final_dataframes
    ) / (1024**2)
    print(f"Total memory usage: {total_memory_mb:.1f} MB")
    print(f"Memory per million rows: {total_memory_mb/(total_rows/1_000_000):.1f} MB")
    
    # Generate detailed performance report
    os.makedirs("outputs", exist_ok=True)
    report_filename = f"outputs/ingestion_aggressive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    report_content = (
        f"--- Level 3 Aggressive Performance Ingestion Report ---\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Input Path: {args.input_path}\n"
        f"Configuration:\n"
        f"  - Chunk Size: {args.chunk_size or 'Adaptive'}\n"
        f"  - Memory Mapping: {args.use_memory_mapping}\n"
        f"  - Async Processing: {args.enable_async_processing}\n"
        f"  - Column Selection: {args.skip_country_code}\n"
        f"Workers Used: {num_processes_used}\n"
        f"Files Processed: {num_files}\n"
        f"Total Rows: {total_rows}\n"
        f"Execution Time: {total_execution_time:.4f} seconds\n"
        f"Throughput: {total_rows/total_execution_time:.0f} rows/second\n"
        f"Memory Usage: {total_memory_mb:.1f} MB\n"
        f"Memory Efficiency: {total_memory_mb/(total_rows/1_000_000):.1f} MB/M rows\n"
    )
    
    try:
        with open(report_filename, 'w') as f:
            f.write(report_content)
        print(f"Detailed report saved: {report_filename}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    # Optional detailed summary
    if args.output_summary and final_dataframes:
        print(f"\n--- Detailed Data Summary ---")
        try:
            if len(final_dataframes) > 1:
                # Sample from each DataFrame to avoid memory issues
                sample_size = min(1000, len(final_dataframes[0]) // 10)
                samples = [df.sample(n=sample_size) for df in final_dataframes[:3]]
                combined_sample = pd.concat(samples, ignore_index=True)
                print(f"Sample from combined data ({len(combined_sample)} rows):")
                combined_sample.info(memory_usage="deep")
            else:
                final_dataframes[0].info(memory_usage="deep")
        except Exception as e:
            print(f"Error generating summary: {e}")

if __name__ == '__main__':
    main()

