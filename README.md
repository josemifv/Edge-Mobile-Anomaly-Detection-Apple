# Edge-Mobile Anomaly Detection (CMMSE 2025)

## Project Overview
Enhanced mobile network anomaly detection pipeline optimized for Apple Silicon, featuring:
- 5-stage modular pipeline for telecommunications anomaly detection
- OSP (Orthogonal Subspace Projection) based anomaly detection
- Hardware-accelerated processing with Apple Silicon optimization
- Academic research focus for CMMSE 2025 conference

## Repository Structure

```
Edge-Mobile-Anomaly-Detection-Apple/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .env.example                # Environment configuration template
├── data/                       # Data storage (symlinked to datasets)
├── scripts/                    # Core pipeline and utility scripts
│   ├── 01_data_ingestion.py to 07_*.py  # Pipeline stages (1-7)
│   ├── run_pipeline.py         # Complete pipeline orchestrator
│   └── utils/                  # Utility and analysis scripts
├── experiments/                # Experimental results and analysis outputs
├── outputs/                    # Benchmark results and reports
├── tests/                      # Test suite for validation
├── docs/                       # Supporting documentation
│   ├── README.md               # Documentation index
│   ├── VALIDATION_GUIDE.md     # Data validation procedures
│   └── codex_review.md         # Code review documentation
└── archive/                    # Historical backups and legacy files
```

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

## Throughput and Compression Metrics

### Comprehensive Performance Analysis

The project includes a comprehensive throughput and compression metrics system that analyzes benchmark runs in detail:

#### Key Metrics Computed

**1. Stage Throughput (Rows/Second)**
- Stage 1: Data Ingestion - 319.9M rows processing rate
- Stage 2: Data Preprocessing - 89.2M rows processing rate  
- Stage 3: Reference Week Selection - 39.4K references processing rate
- Stage 4: Individual Anomaly Detection - 5.65M anomalies processing rate
- Stage 5: Comprehensive Analysis - 10K cells processing rate
- Row count constants stored for accurate throughput calculations

**2. Compression Ratio Analysis**
- Ingested → Preprocessed: ~2.2x compression (4.4GB → 2.0GB)
- Preprocessed → Individual Anomalies: ~13.5x compression (2.0GB → 148MB)
- End-to-End: ~30x overall compression (4.4GB → 148MB)
- Space saved percentages and compression efficiency metrics

**3. CPU Efficiency Metrics**
- Mean and peak CPU utilization percentages
- Process-specific and system-wide CPU usage analysis
- Per-core CPU utilization tracking (optimized for Apple Silicon)
- CPU efficiency scores: work done per CPU utilization unit
- CPU time product analysis for performance optimization

**4. Thermal Headroom Analysis**
- Maximum temperature reached during execution
- Apple Silicon thermal limit tracking (100°C reference)
- Thermal headroom calculation (limit - max_temp)
- Thermal efficiency categorization (good/moderate/high)
- Temperature monitoring integration with powermetrics and sysctl

#### Usage Examples

```bash
# Compute metrics for existing benchmark
uv run python scripts/utils/compute_throughput_metrics.py outputs/benchmarks/20241215_143022/

# Run enhanced benchmark with automatic metrics
uv run python scripts/utils/enhanced_benchmark_runner.py data/raw/ --runs 5 --verbose

# Test metrics computation system
uv run python scripts/utils/test_throughput_metrics.py --verbose
```

#### Enhanced Benchmark Runner

The `enhanced_benchmark_runner.py` automatically computes comprehensive metrics for each benchmark run:

```bash
# Basic enhanced benchmark with automatic metrics
uv run python scripts/utils/enhanced_benchmark_runner.py data/raw/

# Custom configuration with detailed analysis
uv run python scripts/utils/enhanced_benchmark_runner.py data/raw/ \
    --runs 5 --n_components 3 --anomaly_threshold 2.0 --verbose
```

#### Metrics Output Structure

```
outputs/benchmarks/YYYYMMDD_HHMMSS/
├── run_1/, run_2/, ..., run_N/           # Individual run data
│   ├── run_metrics.json                  # Basic run metrics
│   ├── resource_monitor.csv              # Resource monitoring data
│   └── data/*.parquet                    # Pipeline stage outputs
└── summary/                              # Metrics analysis
    ├── run_1_throughput_metrics.json     # Individual run metrics
    ├── run_2_throughput_metrics.json
    ├── all_runs_throughput_metrics.json  # Consolidated metrics
    └── throughput_metrics_summary.json   # Summary statistics
```

#### Academic Applications

- **Performance Characterization**: Complete analysis for CMMSE 2025 submission
- **Hardware Optimization**: Quantitative Apple Silicon optimization benefits
- **Benchmarking Framework**: Reproducible metrics for peer review
- **Thermal Analysis**: Detailed efficiency analysis for hardware research
- **Compression Studies**: Data reduction efficiency across pipeline stages

## System Resource Monitoring

### Low-Overhead System Monitoring (`scripts/utils/monitoring.py`)

Comprehensive system monitoring utility optimized for Apple Silicon with low-overhead resource tracking:

#### Key Features
- **Apple Silicon Optimization**: Native ARM64 architecture detection and optimization
- **Multi-threaded Monitoring**: Background thread with configurable sampling intervals (default: 1 Hz)
- **Comprehensive Metrics**: CPU, memory, temperature, and frequency monitoring
- **powermetrics Integration**: Advanced Apple Silicon metrics with root access (graceful fallback)
- **Non-root Compatible**: Full functionality without requiring elevated privileges
- **DataFrame Output**: Pandas DataFrame with timestamped resource samples

#### Monitored Metrics

**CPU Metrics:**
- Process and system-wide CPU utilization percentages
- Per-core CPU utilization tracking (optimized for M-series processors)
- CPU efficiency scores and time product analysis

**Memory Metrics:**
- RSS (Resident Set Size) and VMS (Virtual Memory Size)
- Peak memory tracking via tracemalloc and getrusage
- Process and system memory percentage utilization

**Apple Silicon Specific Metrics (requires `powermetrics` privileges):**
- CPU die temperature monitoring via powermetrics --samplers smc
- CPU frequency monitoring (max/current) via sysctl
- Thermal headroom calculation (100°C reference limit)
- Fallback behavior: Uses sysctl for temperature when powermetrics unavailable

#### Usage Examples

```python
# Basic monitoring
from scripts.utils.monitoring import start_monitor, stop_monitor
monitor = start_monitor()
# Your code here...
df = stop_monitor(monitor)
print(df.describe())

# Custom sampling rate with Apple Silicon metrics
monitor = start_monitor(interval=0.5, enable_apple_metrics=True)  # 2 Hz sampling
df = stop_monitor(monitor)

# Get system information
from scripts.utils.monitoring import get_system_info
info = get_system_info()
```

#### Command Line Interface

```bash
# Test monitoring for 5 seconds
uv run python scripts/utils/monitoring.py --test --duration 5 --interval 1.0

# Display system information and capabilities
uv run python scripts/utils/monitoring.py --system-info

# Benchmark monitoring overhead at different sampling rates
uv run python scripts/utils/monitoring.py --benchmark --duration 10

# Save monitoring results to CSV
uv run python scripts/utils/monitoring.py --test --duration 10 --output monitor_results.csv
```

#### Sample Output

```
System Information:
==================
{
  "system": {
    "platform": "Darwin",
    "machine": "arm64", 
    "is_apple_silicon": true
  },
  "cpu": {
    "cpu_count": 10,
    "cpu_count_logical": 10
  },
  "memory": {
    "total": 17179869184,
    "available": 8589934592,
    "percent": 50.0
  }
}

Collected 50 samples
Average CPU usage: 15.2%
Peak memory RSS: 245.3 MB
CPU die temperature: 42.5°C (avg)
Thermal headroom: 57.5°C
```

#### powermetrics Requirements and Fallback Behavior

**Root Access (powermetrics):**
- **Requirement**: `sudo` privileges needed for `powermetrics --samplers smc`
- **Benefits**: Detailed thermal monitoring, precise CPU die temperature
- **Usage**: Automatic detection when running with appropriate privileges

**Non-Root Fallback:**
- **Automatic**: Graceful degradation when powermetrics unavailable
- **Alternative**: Uses `sysctl -a` for temperature data when possible
- **Functionality**: All core monitoring features remain available
- **Performance**: No impact on monitoring performance or accuracy

#### Integration with Benchmarking

The monitoring utilities integrate seamlessly with the benchmarking framework:

```bash
# Enhanced benchmark with resource monitoring
uv run scripts/utils/enhanced_benchmark_runner.py data/raw/ --runs 5 --verbose

# Monitoring data automatically collected per benchmark run
outputs/benchmarks/YYYYMMDD_HHMMSS/
├── run_1/resource_monitor.csv     # Per-run monitoring data
├── run_2/resource_monitor.csv
└── summary/performance_plot.png   # Includes resource utilization analysis
```

#### Academic Research Applications

- **Apple Silicon Performance Analysis**: Quantitative thermal and efficiency characterization
- **CMMSE 2025 Conference**: Production-ready monitoring for academic publication
- **Reproducible Research**: Consistent resource tracking across benchmark runs
- **Hardware Optimization Studies**: Detailed metrics for performance optimization research
- **Cross-Platform Comparison**: Baseline for comparing different hardware architectures


## Development Status
✅ Core pipeline implementation completed and verified for CMMSE 2025 submission
✅ Geographic visualization system integrated and tested
✅ Complete end-to-end pipeline with interactive mapping capabilities
✅ Repository structure organized for academic publication

