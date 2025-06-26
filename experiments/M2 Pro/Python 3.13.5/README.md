# M2 Pro - Python 3.13.5 Benchmark Results

This directory contains benchmark results for the Edge Mobile Anomaly Detection pipeline running on M2 Pro with Python 3.13.5.

## Configuration

- **Platform**: Apple M2 Pro
- **Python Version**: 3.13.5
- **Benchmark Date**: June 26, 2025
- **Number of Runs**: 10

## Directory Structure

```
Python 3.13.5/
├── .gitignore                     # Excludes .parquet files
├── README.md                      # This file
└── benchmark_[timestamp]/         # Will be created when benchmark runs
    ├── benchmark_results.json     # Consolidated results
    └── results_run_[1-10]/        # Individual run results
        ├── pipeline_execution.log
        ├── pipeline_status.json
        └── reports/
            ├── anomaly_analysis_summary.txt
            ├── anomalies_by_hour.png
            └── severity_distribution.png
```

## Running Benchmarks

To run the benchmark and populate this directory, use:

```bash
cd /Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple
python scripts/simple_benchmark.py --output_dir "experiments/M2 Pro/Python 3.13.5"
```

## Notes

- Parquet files (*.parquet) are excluded from version control due to their large size
- All other result files including JSON, logs, images, and text files are included
- Each benchmark run creates a timestamped directory for complete isolation of results
