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

The project includes performance tuning capabilities for data ingestion:

### Optimized Ingestion Script

`scripts/01_data_ingestion_optimized.py` provides enhanced performance with:

- **Automatic resource monitoring** (CPU, memory usage)
- **Optimized data types** (float32 vs float64, uint16/uint8 for smaller integers)
- **Memory mapping** for large files
- **Adaptive chunk sizing** based on file size
- **Smart process count** adjustment based on system load
- **Batch processing** for very large datasets
- **Real-time performance metrics**

### Performance Improvements Demonstrated

**Test Results (3 files, 14.3M rows):**
- **Original script**: 5.12 seconds, ~2.8M rows/second
- **Optimized script**: 2.45 seconds, ~5.8M rows/second
- **Improvement**: **108% faster**, 53% less memory usage

### Tunable Parameters

For detailed performance tuning options, see [INGESTION_PERFORMANCE_TUNING.md](INGESTION_PERFORMANCE_TUNING.md):

- Process count optimization
- Chunk size tuning
- Data type optimization
- Memory management
- I/O optimization strategies

## Next Steps

Future development will focus on:

*   Advanced feature engineering (temporal patterns, statistical features)
*   Implementation and benchmarking of anomaly detection algorithms (Isolation Forest, Autoencoders, etc.)
*   Apple Silicon-optimized ML pipeline (PyTorch with MPS backend)
*   Visualization of results and anomaly patterns
*   Energy efficiency analysis for edge computing scenarios

(Refer to the `.context` file for a more detailed project roadmap.)

