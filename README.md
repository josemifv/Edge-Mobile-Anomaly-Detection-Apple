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
4. **Data Consolidation**: Concatenates all individual text files into a single DataFrame

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

## Next Steps

Future development will focus on:

*   Optimizing data storage and organization (e.g., using Parquet, organizing by `square_id`).
*   Advanced feature engineering.
*   Implementation and benchmarking of anomaly detection algorithms (Isolation Forest, Autoencoders, etc.), with a particular focus on Apple Silicon performance.
*   Visualization of results.

(Refer to the `.context` file for a more detailed project roadmap.)

