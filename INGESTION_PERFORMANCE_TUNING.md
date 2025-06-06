# Performance Tuning Guide - Data Ingestion

## Current Performance Baseline

**Current Performance (Apple M4 Pro):**
- **Processing Speed**: 6.57 million rows/second
- **Execution Time**: 48.66 seconds for 319.9M rows
- **Throughput**: ~506 MB/second
- **Parallel Processes**: 14 (matching core count)

## Tunable Parameters for Performance Optimization

### 1. **Parallel Processing Parameters**

#### `num_processes` (Currently: Auto-detected)
```python
# Current implementation
num_processes_used = min(len(files_to_process), os.cpu_count() or 1)
```

**Optimization Options:**
- **Conservative**: `num_processes = cpu_count() - 2` (preserve system responsiveness)
- **Aggressive**: `num_processes = cpu_count() + 2` (I/O bound workload)
- **Memory-limited**: `num_processes = min(files, available_memory_gb // 2)` (prevent OOM)
- **Custom**: Manual specification for specific hardware

**Expected Impact:** ±10-30% throughput depending on I/O vs CPU bottleneck

### 2. **Chunk Size Parameters**

#### `chunk_size` (Currently: None - full file reading)
```python
# Current: reads entire file at once
df = pd.read_csv(file_path, **common_read_params)
```

**Optimization Options:**
- **Small chunks**: `chunk_size = 50_000` (memory efficient)
- **Medium chunks**: `chunk_size = 500_000` (balanced)
- **Large chunks**: `chunk_size = 2_000_000` (I/O efficient)
- **Adaptive chunks**: Based on file size and available memory

**Trade-offs:**
- Smaller chunks: Lower memory usage, higher I/O overhead
- Larger chunks: Higher memory usage, better I/O efficiency
- **Recommended**: Start with `file_size_mb * 10_000` rows

### 3. **pandas read_csv() Parameters**

#### **Buffer Size** (Currently: Default)
```python
# Add to common_read_params
'engine': 'c',          # Use C engine (fastest)
'low_memory': False,    # Read entire file into memory
'memory_map': True,     # Use memory mapping for large files
```

#### **Data Type Optimization**
```python
# Current dtype specifications are good, but can be optimized
dtype_optimized = {
    'square_id': 'uint16',      # If values < 65,536
    'timestamp_ms': 'int64',    # Keep as is for timestamp
    'country_code': 'uint8',    # If limited country codes
    'sms_in': 'float32',        # Reduce precision if acceptable
    'sms_out': 'float32',
    'call_in': 'float32',
    'call_out': 'float32',
    'internet_activity': 'float32'
}
```

**Expected Impact:** 20-40% memory reduction, 5-15% speed improvement

### 4. **I/O Optimization Parameters**

#### **File Reading Strategy**
```python
# Option 1: Memory mapping (for very large files)
common_read_params['memory_map'] = True

# Option 2: Custom buffer size
common_read_params['engine'] = 'c'
common_read_params['c_parser_encoding'] = 'utf-8'

# Option 3: Skip validation for known good data
common_read_params['na_filter'] = False  # If no NaN values expected
```

### 5. **Memory Management Parameters**

#### **Process Pool Configuration**
```python
# Current: Basic pool
with multiprocessing.Pool(processes=num_processes_used) as pool:
    results = pool.starmap(load_and_preprocess_single_file, tasks)

# Optimized: With memory management
with multiprocessing.Pool(
    processes=num_processes_used,
    maxtasksperchild=1,  # Prevent memory leaks
    initializer=set_worker_memory_limit,  # Custom function
    initargs=(max_memory_per_worker,)
) as pool:
    results = pool.starmap(load_and_preprocess_single_file, tasks)
```

### 6. **Preprocessing Optimization**

#### **Conditional Processing** (Currently: Always enabled)
```python
# Make timestamp conversion optional per file
if should_convert_timestamp(file_path):
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

# Skip column reordering if not needed
if output_format_requires_reordering:
    cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
    df = df[cols]
```

### 7. **File Batching Parameters**

#### **Batch Processing Strategy**
```python
# Current: Process all files simultaneously
# Optimized: Process in batches
def process_files_in_batches(files, batch_size=num_cores):
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        yield process_batch(batch)
```

**Benefits:**
- More predictable memory usage
- Better error isolation
- Progress tracking

## Recommended Optimization Strategy

### **Phase 1: Low-Risk Optimizations**
1. **Data type optimization** (float32 instead of float64)
2. **Engine specification** (engine='c')
3. **Memory mapping** for large files
4. **Disable na_filter** if data is clean

**Expected improvement:** 15-25% faster

### **Phase 2: Memory Optimizations**
1. **Chunk size tuning** based on file size
2. **Process pool optimization** with memory limits
3. **Batch processing** for very large datasets

**Expected improvement:** 20-40% more memory efficient

### **Phase 3: Advanced Optimizations**
1. **Adaptive parallelism** based on system load
2. **Async I/O** for file operations
3. **Column selection** (skip unused columns)
4. **Custom C extensions** for critical paths

**Expected improvement:** 30-60% faster (implementation dependent)

## Hardware-Specific Tuning

### **Apple Silicon Optimizations**
```python
# Leverage unified memory architecture
optimal_processes = min(files, 16)  # Sweet spot for M4 Pro
chunk_size = 1_000_000  # Larger chunks for high memory bandwidth
memory_map = True       # Efficient memory usage

# Take advantage of efficiency cores
high_priority_files = large_files[:num_performance_cores]
low_priority_files = small_files
```

### **SSD Storage Optimizations**
```python
# Sequential read optimization
files_sorted_by_size = sorted(files, key=get_file_size, reverse=True)

# Prefetch next files
with ThreadPoolExecutor(max_workers=2) as prefetch_pool:
    prefetch_pool.submit(preload_file, next_file)
```

## Performance Monitoring Parameters

### **Real-time Metrics**
```python
import psutil
import time

def monitor_performance():
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_percent': psutil.virtual_memory().percent,
        'io_read_mb': psutil.disk_io_counters().read_bytes / 1024**2,
        'timestamp': time.time()
    }
```

### **Bottleneck Detection**
```python
# Identify if CPU, memory, or I/O bound
def detect_bottleneck(metrics):
    if metrics['cpu_percent'] > 90:
        return 'CPU_BOUND'
    elif metrics['memory_percent'] > 85:
        return 'MEMORY_BOUND'
    elif metrics['io_wait'] > 50:
        return 'IO_BOUND'
    else:
        return 'BALANCED'
```

## Implementation Priority

### **Quick Wins (1-2 hours implementation)**
1. Add `engine='c'` and `memory_map=True`
2. Optimize data types (float32)
3. Add chunk size parameter with smart defaults
4. Implement process count tuning

### **Medium Term (1-2 days implementation)**
1. Batch processing for large datasets
2. Memory monitoring and limits
3. Adaptive chunk sizing
4. Performance metrics collection

### **Advanced Features (1 week+ implementation)**
1. Async I/O with asyncio
2. Custom memory-mapped readers
3. Hardware-aware optimization
4. Real-time performance tuning

## Expected Results

### **Conservative Improvements (Low-risk changes)**
- **Speed**: 15-25% faster (6.57M → 8.2M rows/sec)
- **Memory**: 20-30% more efficient
- **Total time**: 48.66s → 38-41s

### **Aggressive Improvements (All optimizations)**
- **Speed**: 40-80% faster (6.57M → 11.8M rows/sec)
- **Memory**: 50-70% more efficient
- **Total time**: 48.66s → 27-35s

### **Hardware Scaling Potential**
- **M4 Max (16 cores)**: Additional 15% improvement
- **Mac Studio (24 cores)**: Additional 35% improvement
- **Mac Pro (28 cores)**: Additional 45% improvement

These optimizations will make the ingestion pipeline even more suitable for real-time and edge computing scenarios.

