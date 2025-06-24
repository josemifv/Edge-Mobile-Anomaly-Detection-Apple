# Edge-Mobile Anomaly Detection (CMMSE 2025)

## Project Overview
Enhanced mobile network anomaly detection pipeline optimized for Apple Silicon, featuring:
- 5-stage modular pipeline for telecommunications anomaly detection
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
uv pip install -r requirements.txt
```

## Configuration

All scripts use command-line arguments for configuration. No environment files needed.

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_workers` | auto | Maximum parallel processes (optimized for Apple Silicon) |
| `--output_path` | varies | Output file path (supports .csv and .parquet) |
| `--preview` | False | Show data preview and statistics |

### Stage-Specific Parameters

**Stage 3: Reference Week Selection**
- `--num_weeks` (default: 4) - Number of reference weeks per cell
- `--mad_threshold` (default: 1.5) - MAD threshold for normal weeks

**Stage 4: OSP Anomaly Detection**
- `--n_components` (default: 3) - SVD components for subspace projection
- `--anomaly_threshold` (default: 2.0) - Anomaly detection threshold (std deviations)

**Complete Pipeline Runner**
- `--n_components` (default: 3) - OSP SVD components
- `--anomaly_threshold` (default: 2.0) - OSP anomaly threshold
- `--preview` - Show data previews for all stages

## Pipeline Stages

The pipeline consists of 6 sequential stages with modular architecture:

### Stage 1: Data Ingestion (`01_data_ingestion.py`)
- Loads raw telecommunications data from .txt files
- Parallel processing for multiple files
- Timestamp conversion and initial cleaning
- Apple Silicon optimized
- **Output**: Ingested data (parquet format, ~320M rows)

### Stage 2: Data Preprocessing (`02_data_preprocessing.py`)
- Aggregates data by cell_id and timestamp
- Merges directional columns (SMS/calls in+out → totals)
- Data quality validation and 72% compression
- Outputs clean, consolidated dataset
- **Output**: Preprocessed data (parquet format, ~89M rows)

### Stage 3: Reference Week Selection (`03_week_selection.py`)
- Applies Median Absolute Deviation (MAD) analysis
- Selects "normal" reference weeks per cell
- Configurable selection criteria
- Provides training data for anomaly detection
- **Output**: Reference weeks dataset (~39K reference periods)

### Stage 4: Individual Anomaly Detection (`04_anomaly_detection_individual.py`)
- Implements Orthogonal Subspace Projection using SVD
- Per-cell anomaly detection models with individual record tracking
- Uses reference weeks for normal behavior modeling
- Outputs individual anomaly records for detailed analysis
- **Output**: Individual anomaly records (parquet format, ~5M anomalies)

### Stage 5: Comprehensive Anomaly Analysis (`05_analyze_anomalies.py`)
- Analyzes individual anomaly records from Stage 4
- Generates statistical summaries and visualizations
- Creates temporal and cell-level pattern analysis
- Produces research-grade reports and insights
- **Output**: Analysis reports, visualizations, statistical summaries

### Stage 6: Geographic Anomaly Visualization (`06_generate_anomaly_map.py`)
- Processes individual anomaly records from Stage 4
- Aggregates records into cell-level statistics for mapping
- Generates interactive Folium and Plotly maps
- Percentile-based classification for geographic patterns
- **Output**: Interactive HTML maps with anomaly distribution visualization

## Usage

### Individual Stage Execution
```bash
# Stage 1: Data Ingestion
python scripts/01_data_ingestion.py data/raw/ --output_path data/processed/ingested_data.parquet

# Stage 2: Data Preprocessing
python scripts/02_data_preprocessing.py data/processed/ingested_data.parquet --output_path data/processed/preprocessed_data.parquet

# Stage 3: Reference Week Selection
python scripts/03_week_selection.py data/processed/preprocessed_data.parquet --output_path data/processed/reference_weeks.parquet

# Stage 4: Individual Anomaly Detection
python scripts/04_anomaly_detection_individual.py data/processed/preprocessed_data.parquet data/processed/reference_weeks.parquet --output_path data/processed/individual_anomalies.parquet

# Stage 5: Comprehensive Anomaly Analysis
python scripts/05_analyze_anomalies.py data/processed/individual_anomalies.parquet --output_dir results/analysis/

# Stage 6: Geographic Anomaly Visualization
python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet --output_dir results/maps/
```

### Complete Pipeline Execution
```bash
# Run complete 5-stage pipeline
python scripts/run_pipeline.py data/raw/ --output_dir results/

# With custom parameters
python scripts/run_pipeline.py data/raw/ --output_dir results/ --n_components 5 --anomaly_threshold 2.5 --preview
```

## Performance Results

Tested on **Apple Silicon M4 Pro** with Milan Telecom Dataset (62 files, 319M rows):

| Stage | Processing Time | Throughput | Output |
|-------|----------------|------------|--------|
| **Stage 1: Data Ingestion** | 181.02s | 1.77M rows/sec | 319.9M rows |
| **Stage 2: Data Preprocessing** | 159.17s | 564K rows/sec | 89.2M rows (72% compression) |
| **Stage 3: Reference Week Selection** | 112.27s | 795K rows/sec | 39.4K reference weeks |
| **Stage 4: OSP Anomaly Detection** | 469.97s | 190K samples/sec | 5.65M anomalies (6.33% rate) |
| **Total Pipeline** | **922.43s** | **347K rows/sec** | **100% success rate** |

### Key Achievements
- **✅ Complete dataset processing**: 10,000 cells across 62 days
- **✅ High compression ratio**: 72% data reduction through aggregation
- **✅ Robust anomaly detection**: 6.33% anomaly rate with configurable thresholds
- **✅ Apple Silicon optimization**: Leveraging parallel processing and MPS acceleration
- **✅ Academic reproducibility**: Clean, documented code for research publication

## Geographic Visualization Features

### Interactive Maps Generated

**Stage 6** creates interactive geographic visualizations showing anomaly patterns across the Milano cell network:

#### Data Flow for Geographic Visualization
1. **Input**: Individual anomaly records from Stage 4 (parquet format)
2. **Cell-level Aggregation**: Transform individual records into statistical summaries per cell
3. **Percentile Classification**: Classify cells based on anomaly patterns using percentiles
4. **Map Generation**: Create interactive Folium and Plotly maps with rich information

#### Map Types and Features
- **Folium Interactive Maps**: HTML maps with popups, tooltips, and legends
- **Plotly Choropleth Maps**: Continuous color scales and hover information
- **Percentile-based Classification**: 5-6 meaningful categories per metric
- **Milano Grid Integration**: Full geographic coverage with cell boundaries

#### Generated Outputs
- Interactive HTML maps (Folium and Plotly formats)
- Cell classification datasets (CSV format)
- Statistical summary reports
- Category distribution analysis

### Geographic Visualization Usage

```bash
# Basic usage with Stage 4 output
python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet --output_dir results/maps/

# Custom metrics and classification
python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet \
    --output_dir maps/ --metrics anomaly_count max_severity

# Complete pipeline with geographic visualization
python scripts/run_pipeline.py data/raw/ --output_dir results/
python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet --output_dir results/maps/
```

### Map Classification Metrics

| Metric | Description | Categories |
|--------|-------------|------------|
| **anomaly_count** | Number of anomalies per cell | No Anomalies, Low/Moderate/High/Very High/Extreme Activity |
| **avg_severity** | Average severity score (σ) | Very Low, Low, Moderate, High, Very High |
| **max_severity** | Maximum severity observed | Percentile-based classification |

### Example Map Outputs
- `milano_anomaly_map_anomaly_count_folium.html` - Interactive anomaly count visualization
- `milano_anomaly_map_avg_severity_plotly.html` - Severity distribution choropleth
- `cell_classification_anomaly_count.csv` - Cell-level classification data
- `classification_summary_anomaly_count.txt` - Statistical summary report

## Development Status
✅ Core pipeline implementation completed and verified for CMMSE 2025 submission
✅ Geographic visualization system integrated and tested
✅ Complete end-to-end pipeline with interactive mapping capabilities

