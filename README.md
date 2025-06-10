# Edge-Mobile Anomaly Detection (Apple Silicon Focus)

## Project Overview

This project aims to develop a Python pipeline for detecting anomalies in telecommunications data, specifically using the Milan Telecom dataset. A key secondary goal is to evaluate the performance and energy efficiency of Apple Silicon (SoC) machines for such Machine Learning tasks.

This repository contains the scripts and resources for this endeavor.

## Current Status: Data Ingestion and Preprocessing

The project now includes two main components:
1. **Data Ingestion**: `scripts/01_data_ingestion.py`
2. **Data Preprocessing**: `scripts/02_data_preprocessing.py`

### `01_data_ingestion.py` - Data Loading

This script is responsible for loading and performing initial preprocessing on the raw telecom data, which is expected in text files (`.txt`) with space-separated values.

Key features include:

*   **Flexible Input**: Can process a single data file or all `.txt` files within a specified directory.
*   **Parallel Processing**: Utilizes Python's `multiprocessing` to load and preprocess multiple files in parallel, significantly speeding up ingestion for large datasets.
*   **Column Definition**: Handles a predefined set of columns: `square_id`, `timestamp_ms`, `country_code`, `sms_in`, `sms_out`, `call_in`, `call_out`, and `internet_activity`.
*   **Type Casting**: Converts `square_id`, `timestamp_ms`, and `country_code` to `Int64` (nullable integers), and activity columns to `float`.
*   **Timestamp Conversion**: Converts the `timestamp_ms` (Unix milliseconds) column to Pandas `datetime64[ms]` objects. This can be disabled via a command-line flag.
*   **Column Reordering**: Optionally brings the new `timestamp` column to the front. This can be disabled.
*   **Chunked Reading**: Supports reading large files in chunks to manage memory, though currently, full file reading is the default for parallel processing optimization.
*   **Error Handling**: Includes basic error handling for file parsing and loading issues.
*   **Performance Reporting**: Generates a timestamped performance report in the `outputs/` directory, detailing execution time, files processed, rows processed, and the number of parallel processes used.
*   **Output Summary**: Can display a summary of the combined DataFrame (if processing a directory) or a single DataFrame using the `--output_summary` flag.

**Usage:**

```bash
uv run python scripts/01_data_ingestion.py <input_path> [options]
```

**Arguments:**

*   `input_path`: (Required) Path to a single `.txt` data file or a directory containing `.txt` data files.
*   `--chunk_size INT`: (Optional) Size of chunks for reading CSVs (e.g., 100000). If not specified, files are read whole.
*   `--no_timestamp_conversion`: (Optional) Disables the conversion of `timestamp_ms` to datetime objects.
*   `--no_reorder_columns`: (Optional) Disables the reordering of columns to place the `timestamp` first.
*   `--output_summary`: (Optional) Displays a detailed summary of the resulting DataFrame(s).

**Example:**

```bash
uv run python scripts/01_data_ingestion.py data/milan_telecom_dataset/ --output_summary
```

### `02_data_preprocessing.py` - Data Consolidation and Aggregation

This script performs advanced preprocessing steps on the raw telecom data, including data aggregation, consolidation, and comprehensive validation.

**Preprocessing Steps:**

1. **Column Assignment**: Loads data with specific column names: `cell_id`, `timestamp`, `country_code`, `sms_in`, `sms_out`, `call_in`, `call_out`, `internet_traffic`
2. **Column Removal**: Drops the `country_code` column from each file after loading
3. **Data Aggregation**: Groups data by `cell_id` and `timestamp` and aggregates using sum operations for all numeric columns:
   - `sms_in`: sum
   - `sms_out`: sum  
   - `call_in`: sum
   - `call_out`: sum
   - `internet_traffic`: sum
4. **Column Merging**: Combines directional columns into totals:
   - `sms_total`: sum of `sms_in` + `sms_out`
   - `calls_total`: sum of `call_in` + `call_out`
   - Removes individual directional columns as direction is not relevant for anomaly analysis
5. **Data Consolidation**: Concatenates all individual text files into a single DataFrame

**Validation Steps:**

1. **Duplicate Check**: Verifies that there are no duplicate rows based on the combination of `cell_id` and `timestamp`
2. **Null Value Check**: Ensures that there are no missing/null values anywhere in the dataset
3. **Error Handling**: The validation raises `ValueError` exceptions if any issues are found, stopping the process

**Key Features:**

*   **Parallel Processing**: Uses multiprocessing for efficient file loading
*   **Memory Efficient**: Processes files individually before consolidation
*   **Flexible Output**: Supports both CSV and Parquet output formats
*   **Comprehensive Validation**: Ensures data quality and integrity
*   **Performance Monitoring**: Generates detailed performance reports
*   **Preview Option**: Optional data preview with statistics

**Usage:**

```bash
uv run python scripts/02_data_preprocessing.py <input_path> [options]
```

**Arguments:**

*   `input_path`: (Required) Directory containing the `.txt` data files
*   `--output_path PATH`: (Optional) Output file path (.csv or .parquet)
*   `--max_workers INT`: (Optional) Maximum number of parallel processes (default: number of CPUs)
*   `--preview`: (Optional) Show preview of the final DataFrame

**Examples:**

```bash
# Basic preprocessing
uv run python scripts/02_data_preprocessing.py data/milan_telecom_dataset/

# Save to Parquet format with preview
uv run python scripts/02_data_preprocessing.py data/milan_telecom_dataset/ --output_path data/processed/consolidated_data.parquet --preview

# Use specific number of workers
uv run python scripts/02_data_preprocessing.py data/milan_telecom_dataset/ --max_workers 8
```

## Performance Results

The project demonstrates excellent performance on Apple Silicon hardware. Key results:

- **Hardware**: Apple M4 Pro (14 cores, 24 GB RAM)
- **Dataset**: 319.9M rows, 24 GB raw data (Milan Telecom)
- **Total Processing Time**: 130.85 seconds
- **Final Output**: 89.2M rows, 1.8 GB (Parquet)
- **Overall Throughput**: 2.44M rows/second
- **Storage Efficiency**: 92.5% compression
- **Data Quality**: 100% integrity validation

For detailed performance analysis, benchmarks, and technical measurements, see [PERFORMANCE_MEASUREMENTS.md](PERFORMANCE_MEASUREMENTS.md).

## Performance Optimization

The project maintains **multiple performance implementation levels** (0-3) for comprehensive benchmarking and academic research:

### Performance Level Hierarchy

#### Level 0: Baseline (`01_data_ingestion.py`)
- **Purpose**: Reference implementation, original code
- **Features**: Basic multiprocessing, standard pandas dtypes
- **Performance**: 6.57M rows/second baseline

#### Level 1: Conservative (`01_data_ingestion_optimized.py`)
- **Purpose**: Low-risk optimizations with proven benefits
- **Features**: 
  - Automatic resource monitoring (CPU, memory usage)
  - Optimized data types (float32, uint16, uint8)
  - Memory mapping for large files
  - Adaptive chunk sizing
  - Smart process count adjustment
- **Performance**: 5.8M rows/second (108% faster than baseline on test)

#### Level 2: Moderate (`01_data_ingestion_moderate.py`) 
- **Purpose**: Balanced risk/reward optimizations
- **Features**:
  - All Level 1 optimizations +
  - Advanced resource management
  - Batch processing for large datasets
  - Column selection optimization
  - Custom buffer sizes
  - File size-based optimization strategies
- **Expected Performance**: 25-40% faster than Level 1

#### Level 3: Aggressive (`01_data_ingestion_aggressive.py`)
- **Purpose**: High-performance optimizations for research
- **Features**:
  - All Level 2 optimizations +
  - Async I/O operations
  - Memory-mapped file readers
  - Advanced chunking strategies
  - Performance profiling
  - Hardware-specific optimizations
- **Expected Performance**: 50-70% faster than Level 1

### Comprehensive Benchmarking

**Benchmark All Levels**: `scripts/benchmark_all_levels.py`

```bash
# Test all performance levels
uv run python scripts/benchmark_all_levels.py data/test_dataset/

# Test specific levels
uv run python scripts/benchmark_all_levels.py data/test_dataset/ --levels 0 1 3

# Custom output directory
uv run python scripts/benchmark_all_levels.py data/test_dataset/ --output_dir my_benchmarks/
```

**Features**:
- Automated testing of all performance levels
- Resource monitoring during execution
- Comparative performance reports
- Visualization charts
- JSON/CSV data export

### Performance Documentation

- **[PERFORMANCE_LEVELS.md](PERFORMANCE_LEVELS.md)**: Complete level hierarchy and usage guidelines
- **[INGESTION_PERFORMANCE_TUNING.md](INGESTION_PERFORMANCE_TUNING.md)**: Detailed tuning parameters
- **[PERFORMANCE_MEASUREMENTS.md](PERFORMANCE_MEASUREMENTS.md)**: Comprehensive benchmark results

### Academic Value

This multi-level approach enables:
- **Incremental analysis**: Study impact of individual optimizations
- **Trade-off studies**: Performance vs complexity vs maintainability 
- **Hardware evaluation**: Apple Silicon vs traditional architectures
- **Scalability research**: Performance characteristics across dataset sizes

## Stage 04: OSP Anomaly Detection ✅ IMPLEMENTED

The fourth stage implements Orthogonal Subspace Projection (OSP) based anomaly detection:

### Key Features
- **Per-cell SVD modeling**: Individual anomaly detection models for each cell
- **Reference week training**: Uses weeks selected in Stage 03 for normal behavior modeling
- **Configurable parameters**: SVD components, anomaly thresholds, standardization
- **Apple Silicon optimization**: Leverages optimized NumPy/SciPy libraries
- **Parallel processing**: Multi-core execution for large-scale analysis
- **Comprehensive benchmarking**: Automated performance evaluation framework

### Performance Characteristics
- **Throughput**: 2,205,410 samples/sec on Apple Silicon (full dataset)
- **Scalability**: Successfully processed all 89,245,318 samples across 10,000 cells
- **Processing Time**: 40.47 seconds for complete dataset
- **Memory efficiency**: 5.6GB for full dataset processing
- **Success Rate**: 100% (10,000/10,000 cells processed successfully)
- **Anomaly detection**: 6.23% overall rate, configurable sensitivity (1.5-3.0 std thresholds)

### Usage Examples
```bash
# Basic OSP anomaly detection
uv run python scripts/04_anomaly_detection_osp.py \
    data/processed/consolidated_milan_telecom_merged.parquet \
    reports/reference_weeks_20250607_135521.parquet

# Optimal configuration (full dataset)
uv run python scripts/04_anomaly_detection_osp.py \
    data/processed/consolidated_milan_telecom_merged.parquet \
    reports/reference_weeks_20250607_135521.parquet \
    --n_components 3 --anomaly_threshold 2.0 --max_workers 8

# Comprehensive benchmarking
uv run python scripts/benchmark_osp_anomaly_detection.py \
    data/processed/consolidated_milan_telecom_merged.parquet \
    reports/reference_weeks_20250607_135521.parquet \
    --config_type standard
```

## Current Pipeline Status

✅ **Stage 01**: Data Ingestion (319M rows → 89M rows, 48.67s)  
✅ **Stage 02**: Data Preprocessing (Aggregation & validation, 82.56s)  
✅ **Stage 03**: Reference Week Selection (40k weeks selected, 72.47s)  
✅ **Stage 04**: OSP Anomaly Detection (5.56M anomalies detected, 40.47s)  

**Complete Pipeline Performance**: 244.17s (4.07 minutes) for end-to-end processing

### Stage 04 Full Dataset Results
- **100% Success Rate**: All 10,000 cells processed successfully
- **Outstanding Performance**: 15x faster than projected (40.47s vs 598s estimated)
- **Anomaly Detection**: 5,561,393 anomalies detected (6.23% overall rate)
- **Range**: 0.00% - 49.08% anomaly rate across cells
- **Geographic Patterns**: Higher anomaly rates in cells 5000s and 7000s ranges
- **Output Files**: 309MB detailed results, 243KB summary data

## Next Steps

Upcoming development priorities:

*   **Stage 05**: Alternative anomaly detection algorithms (Isolation Forest, One-Class SVM)
*   **Stage 06**: Deep learning approaches (Autoencoders with PyTorch MPS)
*   **Stage 07**: Anomaly visualization and interpretation tools
*   **Stage 08**: Energy efficiency analysis for edge deployment
*   **Stage 09**: Real-time anomaly detection pipeline

(Refer to the `.context` file for a more detailed project roadmap.)

