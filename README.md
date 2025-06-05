# Edge-Mobile Anomaly Detection (Apple Silicon Focus)

## Project Overview

This project aims to develop a Python pipeline for detecting anomalies in telecommunications data, specifically using the Milan Telecom dataset. A key secondary goal is to evaluate the performance and energy efficiency of Apple Silicon (SoC) machines for such Machine Learning tasks.

This repository contains the scripts and resources for this endeavor.

## Current Status: Data Ingestion

The primary component developed so far is the data ingestion script, located at `scripts/01_data_ingestion.py`.

### `01_data_ingestion.py` - Functionality

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

### Usage

To run the script:

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

## Next Steps

Future development will focus on:

*   Optimizing data storage and organization (e.g., using Parquet, organizing by `square_id`).
*   Advanced feature engineering.
*   Implementation and benchmarking of anomaly detection algorithms (Isolation Forest, Autoencoders, etc.), with a particular focus on Apple Silicon performance.
*   Visualization of results.

(Refer to the `.context` file for a more detailed project roadmap.)

