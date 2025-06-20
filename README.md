# Edge-Mobile Anomaly Detection (CMMSE 2025)

## Project Overview
Enhanced mobile network anomaly detection pipeline optimized for Apple Silicon, featuring:
- 4-stage modular pipeline for telecommunications anomaly detection
- OSP (Orthogonal Subspace Projection) based anomaly detection
- Hardware-accelerated processing with Apple Silicon optimization
- Academic research focus for CMMSE 2025 conference

## Installation
```bash
# Clone repository
git clone https://github.com/josemifv/Edge-Mobile-Anomaly-Detection-Apple.git
cd Edge-Mobile-Anomaly-Detection-Apple

# Create virtual environment with uv
uv venv

# Install dependencies
uv pip install -e ".[dev]"
```

## Configuration
Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

## Pipeline Stages

The pipeline consists of 4 sequential stages:

### Stage 1: Data Ingestion (`01_data_ingestion.py`)
- Loads raw telecommunications data from .txt files
- Parallel processing for multiple files
- Timestamp conversion and initial cleaning
- Apple Silicon optimized

### Stage 2: Data Preprocessing (`02_data_preprocessing.py`)
- Aggregates data by cell_id and timestamp
- Merges directional columns (SMS/calls in+out → totals)
- Data quality validation
- Outputs clean, consolidated dataset

### Stage 3: Reference Week Selection (`03_week_selection.py`)
- Applies Median Absolute Deviation (MAD) analysis
- Selects "normal" reference weeks per cell
- Configurable selection criteria
- Provides training data for anomaly detection

### Stage 4: OSP Anomaly Detection (`04_anomaly_detection_osp.py`)
- Implements Orthogonal Subspace Projection using SVD
- Per-cell anomaly detection models
- Uses reference weeks for normal behavior modeling
- Parallel processing for scalability

## Usage

### Individual Stage Execution
```bash
# Stage 1: Data Ingestion
python scripts/01_data_ingestion.py data/raw/ --output_path data/processed/ingested_data.parquet

# Stage 2: Data Preprocessing
python scripts/02_data_preprocessing.py data/processed/ingested_data.parquet --output_path data/processed/preprocessed_data.parquet

# Stage 3: Reference Week Selection
python scripts/03_week_selection.py data/processed/preprocessed_data.parquet --output_path data/processed/reference_weeks.parquet

# Stage 4: OSP Anomaly Detection
python scripts/04_anomaly_detection_osp.py data/processed/preprocessed_data.parquet data/processed/reference_weeks.parquet --output_path results/anomalies.parquet
```

### Complete Pipeline Execution
```bash
# Run complete 4-stage pipeline
python scripts/run_pipeline.py data/raw/ --output_dir results/

# With custom parameters
python scripts/run_pipeline.py data/raw/ --output_dir results/ --n_components 5 --anomaly_threshold 2.5 --preview
```

## Development Status
✅ Core pipeline implementation completed for CMMSE 2025 submission

