# Data Validation System for CMMSE 2025 Mobile Network Anomaly Detection

## Overview

This document describes the comprehensive data validation system implemented for the mobile network anomaly detection pipeline. The validation system provides robust checks for data quality, consistency, and pipeline integrity.

## Features

### 1. Required Columns Validation
- Verifies that all expected columns are present in parquet files
- Supports stage-specific schema validation
- Detects missing or extra columns

### 2. Data Types and Ranges Validation
- Validates data types match expected schemas
- Checks value ranges for numeric columns
- Handles type compatibility (e.g., float32 vs float64)
- Supports complex data types (arrays, objects)

### 3. Missing Values Handling
- Comprehensive null value detection and reporting
- Percentage-based thresholds for critical errors vs warnings
- Column-wise null value analysis

### 4. Cell ID Consistency with GeoJSON
- Validates cell IDs against geographic reference data
- Handles different cell ID formats (direct integer, arrays/lists)
- Reports coverage statistics and missing cell IDs
- Supports both cellId and cell_id property names in GeoJSON

## Validation Components

### Core Validation Module (`scripts/data_validation.py`)

The main validation module provides:

```python
class DataValidator:
    """Comprehensive data validation utility"""
    
    def validate_parquet_file(self, file_path: str, schema_type: str = 'auto') -> ValidationResult
    def validate_multiple_files(self, file_paths: List[str]) -> Dict[str, ValidationResult]
    def _validate_schema(self, df: pd.DataFrame, schema_type: str, result: ValidationResult)
    def _validate_data_types_and_ranges(self, df: pd.DataFrame, result: ValidationResult)
    def _validate_missing_values(self, df: pd.DataFrame, result: ValidationResult)
    def _validate_cell_id_consistency(self, df: pd.DataFrame, result: ValidationResult)
```

### Pipeline Integration (`scripts/validate_pipeline_data.py`)

Provides end-to-end pipeline validation:
- Validates all pipeline stages automatically
- Generates comprehensive summary reports
- Performs cross-stage data quality checks
- Identifies inconsistencies between pipeline stages

## Supported Schema Types

### 1. Ingested Data (`ingested`)
```python
{
    'required_columns': ['timestamp', 'cell_id', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic'],
    'dtypes': {
        'timestamp': 'datetime64[ns]',
        'cell_id': 'int64',
        'sms_in': 'float64',
        'sms_out': 'float64', 
        'call_in': 'float64',
        'call_out': 'float64',
        'internet_traffic': 'float64'
    }
}
```

### 2. Preprocessed Data (`preprocessed`)
```python
{
    'required_columns': ['cell_id', 'timestamp', 'sms_total', 'calls_total', 'internet_traffic'],
    'dtypes': {
        'cell_id': 'int64',
        'timestamp': 'datetime64[ns]',
        'sms_total': 'float64',
        'calls_total': 'float64',
        'internet_traffic': 'float64'
    }
}
```

### 3. Reference Weeks (`reference_weeks`)
```python
{
    'required_columns': ['cell_id', 'reference_week'],
    'dtypes': {
        'cell_id': 'int64',
        'reference_week': 'object'  # String format like "2013_W47"
    }
}
```

### 4. Individual Anomalies (`individual_anomalies`)
```python
{
    'required_columns': ['cell_id', 'timestamp', 'anomaly_score', 'sms_total', 'calls_total', 'internet_traffic', 'severity_score'],
    'dtypes': {
        'cell_id': 'int64',
        'timestamp': 'datetime64[ns]',
        'anomaly_score': 'float64',
        'sms_total': 'float64',
        'calls_total': 'float64',
        'internet_traffic': 'float64',
        'severity_score': 'float64'
    }
}
```

### 5. Cell Statistics (`cell_statistics`)
```python
{
    'required_columns': ['cell_id', 'anomaly_count', 'avg_severity', 'max_severity'],
    'dtypes': {
        'cell_id': 'int64',
        'anomaly_count': 'int64',
        'avg_severity': 'float64',
        'max_severity': 'float64'
    }
}
```

## Data Range Validation

The system validates the following data ranges:

- **cell_id**: 1 to 100,000 (Milano grid typical range)
- **sms_total**: 0 to 50,000 (reasonable SMS activity range)
- **calls_total**: 0 to 20,000 (reasonable call activity range)
- **internet_traffic**: 0 to 1e12 (internet traffic in bytes)
- **anomaly_score**: ≥ 0 (positive values only)
- **severity_score**: ≥ 2.0 (typically starts at 2 sigma)
- **anomaly_count**: ≥ 0 (non-negative counts)

## Usage Examples

### 1. Single File Validation

```bash
# Validate with auto-detected schema
uv run scripts/data_validation.py results/04_individual_anomalies.parquet

# Validate with specific schema
uv run scripts/data_validation.py results/04_individual_anomalies.parquet --schema individual_anomalies

# Validate with GeoJSON consistency check
uv run scripts/data_validation.py results/04_individual_anomalies.parquet --geojson data/raw/milano-grid.geojson
```

### 2. Multiple File Validation

```bash
# Validate multiple files with summary
uv run scripts/data_validation.py results/02_preprocessed_data.parquet results/04_individual_anomalies.parquet --summary

# Validate all pipeline files
uv run scripts/data_validation.py results/*.parquet --summary
```

### 3. Complete Pipeline Validation

```bash
# Validate entire pipeline with quality checks
uv run scripts/validate_pipeline_data.py --quality_check

# Validate with GeoJSON consistency
uv run scripts/validate_pipeline_data.py --geojson data/raw/milano-grid.geojson --quality_check

# Quick validation without detailed reports
uv run scripts/validate_pipeline_data.py --no_report --quality_check
```

## Validation Output

### Validation Report Structure

Each validation produces a structured report with:

- **Overall Status**: ✅ PASSED or ❌ FAILED
- **Errors**: Critical issues that must be resolved
- **Warnings**: Issues that should be reviewed but don't fail validation
- **Summary Information**: Detailed metrics and statistics

### Example Output

```
================================================================================
VALIDATION REPORT
================================================================================
Overall Status: ✅ PASSED

⚠️  WARNINGS (1):
  1. No GeoJSON provided - cannot validate cell ID consistency

📊 SUMMARY INFORMATION:
  Load time (seconds): 0.533
  File size (MB): 146.95
  Number of rows: 4,841,471
  Number of columns: 7
  Memory usage (MB): 738.75
  anomaly_score range: [2.06e-07, 0.00134]
  severity_score range: [2.0, 16301.78]
  Null values: None found ✓
  Cell ID format: Array/List (using first element)
  Unique cell IDs: 10,000
  Timestamp range: 2013-10-31 23:00:00 to 2014-01-01 22:50:00
  Severe anomalies (>10σ): 526,817
  Extreme anomalies (>100σ): 36,771
================================================================================
```

## Data Quality Checks

### Cross-Stage Consistency
- Cell count consistency across pipeline stages
- Timestamp range consistency
- Schema compatibility verification

### Critical File Detection
- Ensures critical pipeline files exist
- Validates file accessibility and format

### Anomaly-Specific Validation
- Severity score distribution analysis
- Anomaly score validation
- Traffic pattern consistency

## Integration with Pipeline Stages

### Stage 1: Data Ingestion
- Validates raw data format and completeness
- Checks for basic data quality issues
- Handles expected null values in raw telecom data

### Stage 2: Data Preprocessing
- Validates aggregated data structure
- Ensures no data loss during aggregation
- Checks for negative values in traffic metrics

### Stage 3: Reference Week Selection
- Validates reference week format
- Ensures complete cell coverage
- Checks stability score consistency

### Stage 4: Individual Anomaly Detection
- Validates anomaly detection output format
- Checks severity score calculations
- Ensures proper timestamp preservation

### Stage 5-6: Analysis and Visualization
- Validates aggregated statistics
- Ensures data consistency for visualization
- Checks geographic data alignment

## Error Handling and Recovery

### Common Issues and Solutions

1. **Missing Required Columns**
   - Error: Schema validation failure
   - Solution: Check column names and ensure proper data processing

2. **Data Type Mismatches**
   - Warning: Type compatibility issues
   - Solution: Usually acceptable (e.g., float32 vs float64)

3. **Cell ID Inconsistencies**
   - Error: Cell IDs not in GeoJSON
   - Solution: Verify cell ID mapping and GeoJSON completeness

4. **Excessive Null Values**
   - Error: High percentage of missing data
   - Solution: Investigate data source quality or processing logic

5. **Range Violations**
   - Warning: Values outside expected ranges
   - Solution: Review data source or adjust validation thresholds

## Performance Considerations

### Optimization Features
- Lazy loading for large datasets
- Efficient null value counting
- Optimized data type checking
- Memory usage reporting

### Scalability
- Handles files up to several GB efficiently
- Supports batch validation of multiple files
- Progress reporting for long operations

## Configuration and Customization

### Schema Extension
Add new schemas to `EXPECTED_SCHEMAS` dictionary:

```python
'new_schema': {
    'required_columns': ['col1', 'col2'],
    'dtypes': {
        'col1': 'int64',
        'col2': 'float64'
    }
}
```

### Range Customization
Modify `DATA_RANGES` dictionary:

```python
'new_column': (min_value, max_value)  # None for no limit
```

### Validation Thresholds
Adjust thresholds in validation methods:
- Null value percentage thresholds
- Data quality warning levels
- Performance reporting intervals

## Best Practices

### 1. Regular Validation
- Run validation after each pipeline stage
- Validate before critical analysis steps
- Include validation in CI/CD pipelines

### 2. Progressive Validation
- Start with basic file-level validation
- Add schema-specific validation
- Include cross-stage consistency checks

### 3. Error Prioritization
- Address errors before warnings
- Focus on data consistency issues first
- Review warnings for data quality improvements

### 4. Documentation
- Document validation results
- Track validation metrics over time
- Include validation reports in research documentation

## Integration with Academic Research

### CMMSE 2025 Requirements
- Ensures research reproducibility
- Validates data quality for publication
- Provides comprehensive documentation
- Supports peer review requirements

### Research Validation
- Verifies data preprocessing quality
- Ensures anomaly detection reliability
- Validates geographic data consistency
- Provides performance metrics for comparison

---

*This validation system ensures the highest data quality standards for the CMMSE 2025 mobile network anomaly detection research project, providing comprehensive validation capabilities for all pipeline stages while maintaining performance and usability.*
