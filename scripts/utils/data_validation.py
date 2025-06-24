#!/usr/bin/env python3
"""
data_validation.py

CMMSE 2025: Comprehensive Data Validation Module
Provides robust validation for:
- Required columns in parquet files
- Data types and ranges
- Missing values handling
- Cell ID consistency with GeoJSON

This module can be used standalone or integrated into the pipeline stages.
"""

import pandas as pd
import polars as pl
import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings

# Suppress specific warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: Dict[str, Any]
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, key: str, value: Any):
        """Add info data."""
        self.info[key] = value
    
    def print_report(self):
        """Print a formatted validation report."""
        print("\n" + "="*80)
        print("VALIDATION REPORT")
        print("="*80)
        
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        print(f"Overall Status: {status}")
        
        if self.errors:
            print(f"\n🚨 ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if self.info:
            print(f"\n📊 SUMMARY INFORMATION:")
            for key, value in self.info.items():
                if isinstance(value, (int, float)) and value > 1000:
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*80)


class DataValidator:
    """Comprehensive data validation utility for mobile network anomaly detection pipeline."""
    
    # Expected schemas for different pipeline stages
    EXPECTED_SCHEMAS = {
        'ingested': {
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
        },
        'preprocessed': {
            'required_columns': ['cell_id', 'timestamp', 'sms_total', 'calls_total', 'internet_traffic'],
            'dtypes': {
                'cell_id': 'int64',
                'timestamp': 'datetime64[ns]',
                'sms_total': 'float64',
                'calls_total': 'float64',
                'internet_traffic': 'float64'
            }
        },
        'reference_weeks': {
            'required_columns': ['cell_id', 'reference_week'],
            'dtypes': {
                'cell_id': 'int64',
                'reference_week': 'object'  # String format like "2013_W47"
            }
        },
        'individual_anomalies': {
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
        },
        'cell_statistics': {
            'required_columns': ['cell_id', 'anomaly_count', 'avg_severity', 'max_severity'],
            'dtypes': {
                'cell_id': 'int64',
                'anomaly_count': 'int64',
                'avg_severity': 'float64',
                'max_severity': 'float64'
            }
        }
    }
    
    # Data ranges for validation
    DATA_RANGES = {
        'cell_id': (1, 100000),  # Milano grid typical range
        'sms_total': (0, 50000),  # Reasonable SMS activity range
        'calls_total': (0, 20000),  # Reasonable call activity range  
        'internet_traffic': (0, 1e12),  # Internet traffic in bytes
        'anomaly_score': (0, None),  # Positive values only
        'severity_score': (2.0, None),  # Typically starts at 2 sigma
        'anomaly_count': (0, None)  # Non-negative counts
    }
    
    def __init__(self, geojson_path: Optional[str] = None):
        """
        Initialize the validator.
        
        Args:
            geojson_path: Optional path to GeoJSON file for cell ID validation
        """
        self.geojson_path = Path(geojson_path) if geojson_path else None
        self.geojson_cell_ids = None
        
        if self.geojson_path and self.geojson_path.exists():
            self._load_geojson_cell_ids()
    
    def _load_geojson_cell_ids(self):
        """Load cell IDs from GeoJSON file."""
        try:
            with open(self.geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            self.geojson_cell_ids = set()
            for feature in geojson_data.get('features', []):
                properties = feature.get('properties', {})
                cell_id = properties.get('cellId') or properties.get('cell_id')
                if cell_id is not None:
                    self.geojson_cell_ids.add(int(cell_id))
            
            print(f"Loaded {len(self.geojson_cell_ids)} cell IDs from GeoJSON")
        except Exception as e:
            print(f"Warning: Could not load GeoJSON cell IDs: {e}")
            self.geojson_cell_ids = None
    
    def validate_parquet_file(self, file_path: str, schema_type: str = 'auto') -> ValidationResult:
        """
        Validate a parquet file comprehensively.
        
        Args:
            file_path: Path to the parquet file
            schema_type: Expected schema type ('ingested', 'preprocessed', etc.) or 'auto'
            
        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info={})
        
        try:
            # Basic file validation
            file_path = Path(file_path)
            if not file_path.exists():
                result.add_error(f"File does not exist: {file_path}")
                return result
            
            if not file_path.suffix.lower() == '.parquet':
                result.add_error(f"File is not a parquet file: {file_path}")
                return result
            
            # Load data
            print(f"Loading parquet file: {file_path}")
            start_time = time.perf_counter()
            
            try:
                df = pd.read_parquet(file_path)
                load_time = time.perf_counter() - start_time
                result.add_info("Load time (seconds)", round(load_time, 3))
            except Exception as e:
                result.add_error(f"Failed to read parquet file: {e}")
                return result
            
            # Basic info
            result.add_info("File size (MB)", round(file_path.stat().st_size / (1024*1024), 2))
            result.add_info("Number of rows", len(df))
            result.add_info("Number of columns", len(df.columns))
            result.add_info("Memory usage (MB)", round(df.memory_usage(deep=True).sum() / (1024*1024), 2))
            
            # Auto-detect schema type if needed
            if schema_type == 'auto':
                schema_type = self._detect_schema_type(df)
                result.add_info("Detected schema type", schema_type)
            
            # Validate schema
            if schema_type in self.EXPECTED_SCHEMAS:
                self._validate_schema(df, schema_type, result)
            else:
                result.add_warning(f"Unknown schema type: {schema_type}. Skipping schema validation.")
            
            # Validate data types and ranges
            self._validate_data_types_and_ranges(df, result)
            
            # Validate missing values
            self._validate_missing_values(df, result)
            
            # Validate cell ID consistency
            self._validate_cell_id_consistency(df, result)
            
            # Validate timestamps if present
            if 'timestamp' in df.columns:
                self._validate_timestamps(df, result)
            
            # Stage-specific validations
            if schema_type == 'individual_anomalies':
                self._validate_anomaly_data(df, result)
            elif schema_type == 'preprocessed':
                self._validate_aggregated_data(df, result)
            
        except Exception as e:
            result.add_error(f"Unexpected error during validation: {e}")
        
        return result
    
    def _detect_schema_type(self, df: pd.DataFrame) -> str:
        """Auto-detect the schema type based on columns."""
        columns = set(df.columns)
        
        for schema_name, schema_info in self.EXPECTED_SCHEMAS.items():
            required_cols = set(schema_info['required_columns'])
            if required_cols.issubset(columns):
                return schema_name
        
        return 'unknown'
    
    def _validate_schema(self, df: pd.DataFrame, schema_type: str, result: ValidationResult):
        """Validate that the dataframe matches the expected schema."""
        expected = self.EXPECTED_SCHEMAS[schema_type]
        
        # Check required columns
        missing_cols = set(expected['required_columns']) - set(df.columns)
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
        
        extra_cols = set(df.columns) - set(expected['required_columns'])
        if extra_cols:
            result.add_info("Extra columns", list(extra_cols))
        
        # Check data types
        for col, expected_dtype in expected['dtypes'].items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_compatible(actual_dtype, expected_dtype):
                    result.add_warning(f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'")
    
    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        # Handle pandas/numpy dtype variations
        dtype_mappings = {
            'datetime64[ns]': ['datetime64[ns]', 'datetime64[ms]', '<M8[ns]'],
            'int64': ['int64', 'Int64', 'object'],  # object for arrays/lists
            'float64': ['float64', 'Float64', 'float32'],  # Allow float32 as compatible
            'object': ['object', 'string']
        }
        
        expected_variants = dtype_mappings.get(expected, [expected])
        return actual in expected_variants
    
    def _validate_data_types_and_ranges(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data types and value ranges."""
        for col in df.columns:
            if col in self.DATA_RANGES:
                min_val, max_val = self.DATA_RANGES[col]
                
                # Special handling for cell_id which might be arrays
                if col == 'cell_id' and df[col].dtype == 'object':
                    # Skip range validation for array-type cell_id, handled in cell_id consistency check
                    continue
                
                # Check for non-numeric data in numeric columns
                if df[col].dtype in ['object', 'string']:
                    result.add_error(f"Column '{col}' should be numeric but has dtype {df[col].dtype}")
                    continue
                
                # Check ranges
                col_min = df[col].min()
                col_max = df[col].max()
                
                if min_val is not None and col_min < min_val:
                    result.add_error(f"Column '{col}' has values below minimum ({col_min} < {min_val})")
                
                if max_val is not None and col_max > max_val:
                    result.add_warning(f"Column '{col}' has values above expected maximum ({col_max} > {max_val})")
                
                result.add_info(f"{col} range", f"[{col_min}, {col_max}]")
    
    def _validate_missing_values(self, df: pd.DataFrame, result: ValidationResult):
        """Validate missing values across the dataset."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
            result.add_info("Total null values", total_nulls)
            result.add_info("Null percentage", f"{null_percentage:.2f}%")
            
            # Report columns with null values
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            for col, count in cols_with_nulls.items():
                pct = (count / len(df)) * 100
                if pct > 5:  # More than 5% nulls is concerning
                    result.add_error(f"Column '{col}' has {count} ({pct:.1f}%) null values")
                else:
                    result.add_warning(f"Column '{col}' has {count} ({pct:.1f}%) null values")
        else:
            result.add_info("Null values", "None found ✓")
    
    def _validate_cell_id_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """Validate cell ID consistency with GeoJSON data."""
        if 'cell_id' not in df.columns:
            return
        
        # Handle different cell_id formats (int, array, list)
        try:
            cell_id_series = df['cell_id'].dropna()
            
            # Check if cell_id contains arrays/lists
            if len(cell_id_series) > 0 and isinstance(cell_id_series.iloc[0], (np.ndarray, list)):
                # Extract first element from each array/list
                unique_cell_ids = set()
                for cell_id in cell_id_series:
                    if isinstance(cell_id, (np.ndarray, list)) and len(cell_id) > 0:
                        unique_cell_ids.add(int(cell_id[0]))
                result.add_info("Cell ID format", "Array/List (using first element)")
            else:
                # Direct integer cell IDs
                unique_cell_ids = set(cell_id_series.astype(int))
                result.add_info("Cell ID format", "Direct integer")
            
            result.add_info("Unique cell IDs", len(unique_cell_ids))
            
            if self.geojson_cell_ids:
                # Check consistency with GeoJSON
                missing_in_geojson = unique_cell_ids - self.geojson_cell_ids
                missing_in_data = self.geojson_cell_ids - unique_cell_ids
                
                if missing_in_geojson:
                    result.add_error(f"{len(missing_in_geojson)} cell IDs in data but not in GeoJSON")
                    if len(missing_in_geojson) <= 10:
                        result.add_info("Cell IDs missing in GeoJSON", sorted(list(missing_in_geojson)))
                
                if missing_in_data:
                    result.add_info(f"Cell IDs in GeoJSON but not in data", len(missing_in_data))
                
                coverage = len(unique_cell_ids & self.geojson_cell_ids) / len(self.geojson_cell_ids) * 100
                result.add_info("GeoJSON coverage", f"{coverage:.1f}%")
            else:
                result.add_warning("No GeoJSON provided - cannot validate cell ID consistency")
            
            # Check for invalid cell IDs
            invalid_count = 0
            for cell_id in cell_id_series:
                if isinstance(cell_id, (np.ndarray, list)):
                    if len(cell_id) == 0 or cell_id[0] <= 0:
                        invalid_count += 1
                else:
                    if cell_id <= 0:
                        invalid_count += 1
            
            if invalid_count > 0:
                result.add_error(f"Found {invalid_count} rows with invalid cell IDs")
                
        except Exception as e:
            result.add_error(f"Error validating cell IDs: {e}")
    
    def _validate_timestamps(self, df: pd.DataFrame, result: ValidationResult):
        """Validate timestamp data."""
        ts_col = df['timestamp']
        
        # Check if timestamps are valid
        if pd.api.types.is_datetime64_any_dtype(ts_col):
            min_ts = ts_col.min()
            max_ts = ts_col.max()
            result.add_info("Timestamp range", f"{min_ts} to {max_ts}")
            
            # Check for reasonable date range (Milano dataset is 2013-2014)
            if min_ts.year < 2010 or max_ts.year > 2025:
                result.add_warning(f"Unusual timestamp range: {min_ts.year}-{max_ts.year}")
            
            # Check for duplicates (only if cell_id is simple format)
            if 'cell_id' in df.columns and df['cell_id'].dtype != 'object':
                duplicates = df.duplicated(subset=['cell_id', 'timestamp']).sum()
                if duplicates > 0:
                    result.add_error(f"Found {duplicates} duplicate cell_id-timestamp pairs")
            elif 'cell_id' in df.columns:
                # For complex cell_id formats, skip duplicate check
                result.add_info("Duplicate check", "Skipped (complex cell_id format)")
        else:
            result.add_error("Timestamp column is not datetime type")
    
    def _validate_anomaly_data(self, df: pd.DataFrame, result: ValidationResult):
        """Validate anomaly-specific data."""
        if 'severity_score' in df.columns:
            # Check severity score distribution
            severe_count = (df['severity_score'] > 10).sum()
            extreme_count = (df['severity_score'] > 100).sum()
            
            result.add_info("Severe anomalies (>10σ)", severe_count)
            result.add_info("Extreme anomalies (>100σ)", extreme_count)
            
            # Check for negative severity scores
            negative_severity = (df['severity_score'] < 0).sum()
            if negative_severity > 0:
                result.add_error(f"Found {negative_severity} negative severity scores")
        
        if 'anomaly_score' in df.columns:
            # Check anomaly score consistency
            zero_scores = (df['anomaly_score'] <= 0).sum()
            if zero_scores > 0:
                result.add_warning(f"Found {zero_scores} anomaly records with zero or negative scores")
    
    def _validate_aggregated_data(self, df: pd.DataFrame, result: ValidationResult):
        """Validate aggregated data quality."""
        # Check for negative values in traffic data
        for col in ['sms_total', 'calls_total', 'internet_traffic']:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    result.add_error(f"Found {negative_count} negative values in {col}")
                
                zero_count = (df[col] == 0).sum()
                zero_pct = (zero_count / len(df)) * 100
                if zero_pct > 50:
                    result.add_warning(f"Column {col} has {zero_pct:.1f}% zero values")
    
    def validate_multiple_files(self, file_paths: List[str], schema_types: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """
        Validate multiple parquet files.
        
        Args:
            file_paths: List of file paths to validate
            schema_types: List of schema types (same length as file_paths) or None for auto-detection
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        if schema_types and len(schema_types) != len(file_paths):
            raise ValueError("schema_types length must match file_paths length")
        
        results = {}
        for i, file_path in enumerate(file_paths):
            schema_type = schema_types[i] if schema_types else 'auto'
            print(f"\n{'='*60}")
            print(f"Validating: {file_path}")
            print(f"{'='*60}")
            
            results[file_path] = self.validate_parquet_file(file_path, schema_type)
            results[file_path].print_report()
        
        return results
    
    def generate_validation_summary(self, results: Dict[str, ValidationResult]) -> ValidationResult:
        """Generate a summary of multiple validation results."""
        summary = ValidationResult(is_valid=True, errors=[], warnings=[], info={})
        
        total_files = len(results)
        passed_files = sum(1 for r in results.values() if r.is_valid)
        failed_files = total_files - passed_files
        
        summary.add_info("Total files validated", total_files)
        summary.add_info("Files passed", passed_files)
        summary.add_info("Files failed", failed_files)
        
        if failed_files > 0:
            summary.is_valid = False
            summary.add_error(f"{failed_files} out of {total_files} files failed validation")
        
        # Aggregate errors and warnings
        all_errors = []
        all_warnings = []
        for file_path, result in results.items():
            for error in result.errors:
                all_errors.append(f"{Path(file_path).name}: {error}")
            for warning in result.warnings:
                all_warnings.append(f"{Path(file_path).name}: {warning}")
        
        summary.errors = all_errors
        summary.warnings = all_warnings
        
        return summary


def main():
    """Main function for standalone validation."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Data Validation Tool")
    parser.add_argument("files", nargs="+", help="Parquet files to validate")
    parser.add_argument("--geojson", help="Path to GeoJSON file for cell ID validation")
    parser.add_argument("--schema", choices=list(DataValidator.EXPECTED_SCHEMAS.keys()) + ['auto'], 
                       default='auto', help="Expected schema type")
    parser.add_argument("--summary", action="store_true", help="Generate summary report for multiple files")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CMMSE 2025: Data Validation Tool")
    print("="*80)
    
    # Initialize validator
    validator = DataValidator(geojson_path=args.geojson)
    
    # Validate files
    if len(args.files) == 1:
        # Single file validation
        result = validator.validate_parquet_file(args.files[0], args.schema)
        result.print_report()
        exit_code = 0 if result.is_valid else 1
    else:
        # Multiple file validation
        schema_types = [args.schema] * len(args.files) if args.schema != 'auto' else None
        results = validator.validate_multiple_files(args.files, schema_types)
        
        if args.summary:
            print(f"\n{'='*80}")
            print("VALIDATION SUMMARY")
            print(f"{'='*80}")
            summary = validator.generate_validation_summary(results)
            summary.print_report()
            exit_code = 0 if summary.is_valid else 1
        else:
            exit_code = 0 if all(r.is_valid for r in results.values()) else 1
    
    print(f"\nValidation {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)


if __name__ == "__main__":
    main()
