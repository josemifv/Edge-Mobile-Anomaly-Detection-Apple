# Benchmark Experiments

This directory contains the results of benchmark experiments performed on different platforms and Python versions.

## Directory Structure

```
experiments/
├── M2 Pro/
│   ├── Python 3.13.2/
│   │   └── benchmark_20250625_193055/
│   │       ├── benchmark_results.json
│   │       └── results_run_[1-10]/
│   │           ├── pipeline_execution.log
│   │           ├── pipeline_status.json
│   │           └── reports/
│   │               ├── anomaly_analysis_summary.txt
│   │               ├── anomalies_by_hour.png
│   │               └── severity_distribution.png
│   └── Python 3.13.5/
│       └── benchmark_[timestamp]/
│           ├── benchmark_results.json
│           └── results_run_[1-10]/
│               ├── pipeline_execution.log
│               ├── pipeline_status.json
│               └── reports/
│                   ├── anomaly_analysis_summary.txt
│                   ├── anomalies_by_hour.png
│                   └── severity_distribution.png
└── M4 Pro/
    └── Python 3.13.5/
        └── benchmark_20250625_210121/
            └── results_run_[1-10]/
                ├── pipeline_execution.log
                ├── pipeline_status.json
                └── reports/
                    ├── anomaly_analysis_summary.txt
                    ├── anomalies_by_hour.png
                    └── severity_distribution.png
```

## Included Files

- **pipeline_execution.log**: Detailed pipeline execution logs
- **pipeline_status.json**: Pipeline status and performance metrics
- **benchmark_results.json**: Consolidated benchmark results (M2 Pro only)
- **reports/**: Visualizations and anomaly analysis
  - **anomaly_analysis_summary.txt**: Textual analysis summary
  - **anomalies_by_hour.png**: Temporal distribution of anomalies
  - **severity_distribution.png**: Distribution by severity

## Excluded Files

The following file types are excluded from the repository due to size:
- **\*.parquet**: Binary data files (very heavy)
- **03_reference_weeks.parquet**: Weekly reference data
- **04_individual_anomalies.parquet**: Individual anomaly data

## Tested Platforms

- **M2 Pro**: 
  - Python 3.13.2: 10 runs
  - Python 3.13.5: 10 runs
- **M4 Pro**: 10 runs with Python 3.13.5

## Experiment Dates

- **M2 Pro**: 
  - Python 3.13.2: June 25, 2025, 19:30:55
  - Python 3.13.5: June 26, 2025, [timestamp]
- **M4 Pro**: June 25, 2025, 21:01:21

## Access to Complete Data

To access the complete `.parquet` files, contact the research team or run the benchmarks locally.
