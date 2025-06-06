import pandas as pd
import os
import argparse
import time
import multiprocessing
import psutil
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import gc
import numpy as np

# --- Level 2: Moderate Performance Configuration ---
OPTIMIZED_DTYPES = {
    'square_id': 'uint16',      # Optimized for values < 65,536
    'timestamp_ms': 'int64',    # Keep precision for timestamp
    'country_code': 'uint8',    # Optimized for limited country codes
    'sms_in': 'float32',        # Reduced precision
    'sms_out': 'float32',
    'call_in': 'float32', 
    'call_out': 'float32',
    'internet_activity': 'float32'
}

# Only load essential columns to reduce memory
ESSENTIAL_COLUMNS = [
    'square_id', 'timestamp_ms', 'sms_in', 'sms_out', 
    'call_in', 'call_out', 'internet_activity'
]

ALL_COLUMNS = [
    'square_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out',
    'call_in', 'call_out', 'internet_activity'
]

# Advanced buffer sizes for different file sizes
BUFFER_SIZE_MAP = {
    'small': 64 * 1024,      # 64KB for files < 100MB
    'medium': 256 * 1024,    # 256KB for files 100MB-500MB
    'large': 1024 * 1024,    # 1MB for files > 500MB
}

# --- Advanced Resource Management ---
class ResourceManager:
    def __init__(self, target_memory_usage=0.7):
        self.target_memory_usage = target_memory_usage
        self.total_memory = psutil.virtual_memory().total
        self.target_memory_bytes = int(self.total_memory * target_memory_usage)
        
    def get_current_usage(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=0),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024**2)
        }
    
    def should_reduce_parallelism(self):
        usage = self.get_current_usage()
        return usage['memory_percent'] > 85 or usage['cpu_percent'] > 95
    
    def get_optimal_chunk_size(self, file_size_bytes, base_chunk_rows=500_000):
        # Calculate chunk size based on available memory and file size
        available_memory = psutil.virtual_memory().available
        
        # Estimate memory per row (approximately 64 bytes with optimized dtypes)
        bytes_per_row = 64
        max_rows_in_memory = available_memory // (bytes_per_row * 4)  # 25% of available memory
        
        # Adaptive chunk size based on file size
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb < 100:
            return min(base_chunk_rows // 2, max_rows_in_memory)
        elif file_size_mb < 500:
            return min(base_chunk_rows, max_rows_in_memory)
        else:
            return min(base_chunk_rows * 2, max_rows_in_memory)

# --- Advanced File Processing ---
def get_file_info(file_path):
    """Get detailed file information for optimization decisions."""
    stat = os.stat(file_path)
    size_mb = stat.st_size / (1024 * 1024)
    
    # Classify file size
    if size_mb < 100:
        category = 'small'
    elif size_mb < 500:
        category = 'medium'
    else:
        category = 'large'
    
    return {
        'size_bytes': stat.st_size,
        'size_mb': size_mb,
        'category': category,
        'buffer_size': BUFFER_SIZE_MAP[category]
    }

def load_file_with_advanced_optimization(
    file_path,
    chunk_size=None,
    convert_timestamp=True,
    reorder_cols=True,
    skip_country_code=True,
    use_optimized_dtypes=True,
    enable_memory_mapping=True,
    resource_manager=None
):
    """
    Level 2: Advanced file loading with moderate optimizations.
    """
    print(f"\n--- [Level 2] Processing: {Path(file_path).name} ---")
    start_time = time.perf_counter()
    
    # Get file information for optimization
    file_info = get_file_info(file_path)
    print(f"File size: {file_info['size_mb']:.1f} MB ({file_info['category']})")
    
    # Select columns and dtypes based on configuration
    if skip_country_code:
        columns_to_load = ESSENTIAL_COLUMNS
        dtype_subset = {k: v for k, v in OPTIMIZED_DTYPES.items() if k in ESSENTIAL_COLUMNS}
    else:
        columns_to_load = ALL_COLUMNS
        dtype_subset = OPTIMIZED_DTYPES if use_optimized_dtypes else None
    
    # Advanced read parameters
    read_params = {
        "sep": r'\s+',
        "header": None,
        "names": ALL_COLUMNS,  # Always provide all names
        "usecols": columns_to_load,  # But only load essential columns
        "dtype": dtype_subset,
        "engine": 'c',
        "memory_map": enable_memory_mapping,
        "na_filter": True,
        "low_memory": False,
        "buffer_size": file_info['buffer_size'],  # Custom buffer size
    }
    
    try:
        # Determine optimal chunk size
        if chunk_size is None and resource_manager:
            chunk_size = resource_manager.get_optimal_chunk_size(
                file_info['size_bytes']
            )
            auto_chunk = True
        else:
            auto_chunk = False
        
        # Memory-conscious loading
        if chunk_size and chunk_size > 0:
            if auto_chunk:
                print(f"Auto chunk size: {chunk_size:,} rows")
            
            chunks = []
            total_chunks = 0
            
            # Read in chunks with memory monitoring
            for chunk_num, chunk in enumerate(
                pd.read_csv(file_path, chunksize=chunk_size, **read_params)
            ):
                chunks.append(chunk)
                total_chunks += 1
                
                # Memory pressure check
                if resource_manager and resource_manager.should_reduce_parallelism():
                    print(f"Memory pressure detected at chunk {chunk_num}")
                    gc.collect()  # Force garbage collection
            
            print(f"Processed {total_chunks} chunks")
            df = pd.concat(chunks, ignore_index=True)
            
            # Clear chunks from memory
            del chunks
            gc.collect()
        else:
            print("Loading entire file")
            df = pd.read_csv(file_path, **read_params)
        
        print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols, "
              f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Advanced preprocessing
    if convert_timestamp and 'timestamp_ms' in df.columns:
        # Vectorized timestamp conversion
        print("Converting timestamps...")
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
        
        # Check for conversion errors
        null_count = df['timestamp'].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} timestamp conversion failures")
        
        df = df.drop(columns=['timestamp_ms'])
    
    if reorder_cols and 'timestamp' in df.columns:
        # Efficient column reordering
        cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df[cols]
    
    # Final memory optimization
    if use_optimized_dtypes:
        # Convert any remaining object columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except:
                pass
    
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    rows_per_second = len(df) / processing_time if processing_time > 0 else 0
    
    print(f"Completed in {processing_time:.4f}s ({rows_per_second:,.0f} rows/sec)")
    
    return df

# --- Batch Processing with Advanced Memory Management ---
def process_files_in_batches(
    file_paths,
    batch_size,
    resource_manager,
    **processing_kwargs
):
    """
    Process files in batches with memory management.
    """
    print(f"Processing {len(file_paths)} files in batches of {batch_size}")
    
    all_results = []
    
    for batch_idx in range(0, len(file_paths), batch_size):
        batch = file_paths[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        total_batches = (len(file_paths) - 1) // batch_size + 1
        
        print(f"\n--- Batch {batch_num}/{total_batches} ---")
        
        # Monitor resources before batch
        pre_batch_usage = resource_manager.get_current_usage()
        print(f"Pre-batch: CPU {pre_batch_usage['cpu_percent']:.1f}%, "
              f"Memory {pre_batch_usage['memory_percent']:.1f}%")
        
        # Process batch
        with multiprocessing.Pool(processes=len(batch)) as pool:
            tasks = [
                (file_path, *processing_kwargs.values())
                for file_path in batch
            ]
            
            batch_results = pool.starmap(
                load_file_with_advanced_optimization,
                [(path, *args, resource_manager) for path, *args in tasks]
            )
        
        all_results.extend(batch_results)
        
        # Monitor resources after batch
        post_batch_usage = resource_manager.get_current_usage()
        print(f"Post-batch: CPU {post_batch_usage['cpu_percent']:.1f}%, "
              f"Memory {post_batch_usage['memory_percent']:.1f}%")
        
        # Force garbage collection between batches
        gc.collect()
        
        # Adaptive batch size based on memory usage
        if post_batch_usage['memory_percent'] > 80:
            batch_size = max(1, batch_size // 2)
            print(f"Reducing batch size to {batch_size} due to memory pressure")
    
    return all_results

# --- Advanced Directory Processing ---
def process_directory_moderate(
    directory_path,
    chunk_size=None,
    convert_timestamp=True,
    reorder_cols=True,
    max_workers=None,
    use_optimized_dtypes=True,
    enable_memory_mapping=True,
    enable_batch_processing=True,
    skip_country_code=True
):
    """
    Level 2: Moderate optimization directory processing.
    """
    # Initialize resource manager
    resource_manager = ResourceManager()
    
    # Find files and sort by size (process larger files first)
    files_to_process = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))
    ]
    
    if not files_to_process:
        print(f"No .txt files found in {directory_path}")
        return [], 0
    
    # Sort files by size (largest first for better load balancing)
    files_with_sizes = [(f, os.path.getsize(f)) for f in files_to_process]
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    files_to_process = [f[0] for f in files_with_sizes]
    
    print(f"Found {len(files_to_process)} files")
    print(f"Size range: {min(s[1] for s in files_with_sizes) / 1024**2:.1f} - "
          f"{max(s[1] for s in files_with_sizes) / 1024**2:.1f} MB")
    
    # Monitor initial resources
    initial_resources = resource_manager.get_current_usage()
    print(f"Initial resources - CPU: {initial_resources['cpu_percent']:.1f}%, "
          f"Memory: {initial_resources['memory_percent']:.1f}%, "
          f"Available: {initial_resources['memory_available_gb']:.1f} GB")
    
    # Determine optimal parallelism
    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        # Conservative approach for Level 2
        if initial_resources['memory_percent'] > 60:
            optimal_workers = max(1, cpu_count // 2)
        elif initial_resources['memory_percent'] > 40:
            optimal_workers = max(1, cpu_count - 2)
        else:
            optimal_workers = min(len(files_to_process), cpu_count)
    else:
        optimal_workers = max_workers
    
    print(f"Using {optimal_workers} parallel workers")
    
    start_time = time.perf_counter()
    
    try:
        # Decide on batch processing
        if enable_batch_processing and len(files_to_process) > optimal_workers * 2:
            batch_size = optimal_workers
            print(f"Using batch processing with batch size: {batch_size}")
            
            results = process_files_in_batches(
                files_to_process,
                batch_size,
                resource_manager,
                chunk_size=chunk_size,
                convert_timestamp=convert_timestamp,
                reorder_cols=reorder_cols,
                skip_country_code=skip_country_code,
                use_optimized_dtypes=use_optimized_dtypes,
                enable_memory_mapping=enable_memory_mapping
            )
        else:
            # Standard parallel processing
            print("Using standard parallel processing")
            
            with multiprocessing.Pool(processes=optimal_workers) as pool:
                tasks = [
                    (
                        file_path,
                        chunk_size,
                        convert_timestamp,
                        reorder_cols,
                        skip_country_code,
                        use_optimized_dtypes,
                        enable_memory_mapping,
                        resource_manager
                    )
                    for file_path in files_to_process
                ]
                
                results = pool.starmap(
                    load_file_with_advanced_optimization,
                    tasks
                )
        
        # Filter successful results
        successful_dataframes = [df for df in results if df is not None]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance statistics
        successful_count = len(successful_dataframes)
        failed_count = len(results) - successful_count
        total_rows = sum(len(df) for df in successful_dataframes)
        
        print(f"\n--- Level 2 Processing Summary ---")
        print(f"Successful files: {successful_count}/{len(files_to_process)}")
        print(f"Failed files: {failed_count}")
        print(f"Total rows: {total_rows:,}")
        print(f"Processing time: {total_time:.4f} seconds")
        print(f"Throughput: {total_rows/total_time:,.0f} rows/second")
        
        # Final resource usage
        final_resources = resource_manager.get_current_usage()
        print(f"Final resources - CPU: {final_resources['cpu_percent']:.1f}%, "
              f"Memory: {final_resources['memory_percent']:.1f}%")
        
        return successful_dataframes, optimal_workers
    
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return [], 0

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Level 2: Moderate performance data ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_path", help="Path to file or directory")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (auto if not specified)")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel workers")
    parser.add_argument("--no_timestamp_conversion", action="store_false", dest="convert_timestamp")
    parser.add_argument("--no_reorder_columns", action="store_false", dest="reorder_cols")
    parser.add_argument("--no_dtype_optimization", action="store_false", dest="use_optimized_dtypes")
    parser.add_argument("--no_memory_mapping", action="store_false", dest="enable_memory_mapping")
    parser.add_argument("--no_batch_processing", action="store_false", dest="enable_batch_processing")
    parser.add_argument("--include_country_code", action="store_false", dest="skip_country_code")
    parser.add_argument("--output_summary", action="store_true")
    
    args = parser.parse_args()
    
    print("=== Level 2: Moderate Performance Data Ingestion ===")
    print("Optimizations enabled:")
    print(f"  - Advanced resource monitoring: True")
    print(f"  - Optimized data types: {args.use_optimized_dtypes}")
    print(f"  - Memory mapping: {args.enable_memory_mapping}")
    print(f"  - Batch processing: {args.enable_batch_processing}")
    print(f"  - Column selection: {args.skip_country_code}")
    print(f"  - Adaptive chunk sizing: True")
    
    overall_start_time = time.perf_counter()
    
    final_dataframes = []
    num_processes_used = 1
    
    if os.path.isdir(args.input_path):
        print(f"\nProcessing directory: {args.input_path}")
        final_dataframes, num_processes_used = process_directory_moderate(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            max_workers=args.max_workers,
            use_optimized_dtypes=args.use_optimized_dtypes,
            enable_memory_mapping=args.enable_memory_mapping,
            enable_batch_processing=args.enable_batch_processing,
            skip_country_code=args.skip_country_code
        )
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.txt'):
        print(f"\nProcessing single file: {args.input_path}")
        resource_manager = ResourceManager()
        df = load_file_with_advanced_optimization(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            skip_country_code=args.skip_country_code,
            use_optimized_dtypes=args.use_optimized_dtypes,
            enable_memory_mapping=args.enable_memory_mapping,
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
    
    # Final statistics
    total_rows = sum(len(df) for df in final_dataframes)
    num_files = len(final_dataframes)
    
    print(f"\n=== Level 2 Final Results ===")
    print(f"Files processed: {num_files}")
    print(f"Total rows: {total_rows:,}")
    print(f"Execution time: {total_execution_time:.4f} seconds")
    print(f"Overall throughput: {total_rows/total_execution_time:,.0f} rows/second")
    
    # Generate performance report
    os.makedirs("outputs", exist_ok=True)
    report_filename = f"outputs/ingestion_moderate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    report_content = (
        f"--- Level 2 Moderate Performance Ingestion Report ---\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Input Path: {args.input_path}\n"
        f"Configuration:\n"
        f"  - Chunk Size: {args.chunk_size or 'Adaptive'}\n"
        f"  - Optimized Types: {args.use_optimized_dtypes}\n"
        f"  - Memory Mapping: {args.enable_memory_mapping}\n"
        f"  - Batch Processing: {args.enable_batch_processing}\n"
        f"  - Column Selection: {args.skip_country_code}\n"
        f"Workers Used: {num_processes_used}\n"
        f"Files Processed: {num_files}\n"
        f"Total Rows: {total_rows}\n"
        f"Execution Time: {total_execution_time:.4f} seconds\n"
        f"Throughput: {total_rows/total_execution_time:.0f} rows/second\n"
    )
    
    try:
        with open(report_filename, 'w') as f:
            f.write(report_content)
        print(f"Report saved: {report_filename}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    # Show summary if requested
    if args.output_summary and final_dataframes:
        print(f"\n--- Detailed Summary ---")
        if len(final_dataframes) > 1:
            try:
                combined_df = pd.concat(final_dataframes, ignore_index=True)
                print(f"Combined DataFrame:")
                combined_df.info(memory_usage="deep")
                if 'timestamp' in combined_df.columns:
                    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            except MemoryError:
                print("Memory error combining DataFrames. Showing first DataFrame:")
                final_dataframes[0].info(memory_usage="deep")
        else:
            final_dataframes[0].info(memory_usage="deep")

if __name__ == '__main__':
    main()

