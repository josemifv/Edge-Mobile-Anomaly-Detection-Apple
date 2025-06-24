#!/usr/bin/env python3
"""
validate_pipeline_data.py

CMMSE 2025: Pipeline Data Validation Integration
Demonstrates how to integrate comprehensive data validation into the pipeline stages.

This script validates all pipeline outputs and generates a comprehensive report
with validation results for each stage.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_validation import DataValidator, ValidationResult

def validate_pipeline_outputs(
    base_path: str = "results/",
    geojson_path: str = None,
    generate_report: bool = True
) -> Dict[str, ValidationResult]:
    """
    Validate all pipeline output files with stage-specific schemas.
    
    Args:
        base_path: Base directory containing pipeline outputs
        geojson_path: Optional path to GeoJSON file for cell ID validation
        generate_report: Whether to generate a detailed validation report
        
    Returns:
        Dictionary mapping file paths to validation results
    """
    base_path = Path(base_path)
    
    # Define pipeline files and their expected schemas
    pipeline_files = {
        "01_ingested_data.parquet": "ingested",
        "02_preprocessed_data.parquet": "preprocessed", 
        "03_reference_weeks.parquet": "reference_weeks",
        "04_individual_anomalies.parquet": "individual_anomalies",
        "06_cell_statistics.parquet": "cell_statistics"
    }
    
    # Initialize validator
    validator = DataValidator(geojson_path=geojson_path)
    
    # Validate each file
    results = {}
    available_files = []
    
    print("="*80)
    print("CMMSE 2025: Pipeline Data Validation")
    print("="*80)
    print(f"Base directory: {base_path}")
    if geojson_path:
        print(f"GeoJSON file: {geojson_path}")
    print()
    
    # Check which files exist
    for filename, schema_type in pipeline_files.items():
        file_path = base_path / filename
        if file_path.exists():
            available_files.append((str(file_path), schema_type))
            print(f"✓ Found: {filename}")
        else:
            print(f"✗ Missing: {filename}")
    
    print(f"\nValidating {len(available_files)} available files...\n")
    
    # Validate available files
    for file_path, schema_type in available_files:
        filename = Path(file_path).name
        print(f"{'='*60}")
        print(f"Validating: {filename} (Schema: {schema_type})")
        print(f"{'='*60}")
        
        start_time = time.perf_counter()
        result = validator.validate_parquet_file(file_path, schema_type)
        validation_time = time.perf_counter() - start_time
        
        result.add_info("Validation time (seconds)", round(validation_time, 3))
        results[file_path] = result
        
        if generate_report:
            result.print_report()
            print()
    
    return results

def generate_summary_report(results: Dict[str, ValidationResult]) -> ValidationResult:
    """Generate a comprehensive summary report of all validations."""
    validator = DataValidator()
    summary = validator.generate_validation_summary(results)
    
    # Add additional summary statistics
    total_rows = sum(r.info.get("Number of rows", 0) for r in results.values())
    total_size_mb = sum(r.info.get("File size (MB)", 0) for r in results.values())
    total_memory_mb = sum(r.info.get("Memory usage (MB)", 0) for r in results.values())
    
    summary.add_info("Total rows across all files", total_rows)
    summary.add_info("Total file size (MB)", round(total_size_mb, 2))
    summary.add_info("Total memory usage (MB)", round(total_memory_mb, 2))
    
    # Data quality summary
    files_with_no_nulls = sum(1 for r in results.values() if r.info.get("Null values") == "None found ✓")
    summary.add_info("Files with no null values", files_with_no_nulls)
    
    # Cell ID coverage summary
    unique_cell_counts = [r.info.get("Unique cell IDs", 0) for r in results.values() if "Unique cell IDs" in r.info]
    if unique_cell_counts:
        max_cells = max(unique_cell_counts)
        summary.add_info("Maximum unique cells found", max_cells)
    
    return summary

def check_data_quality_issues(results: Dict[str, ValidationResult]) -> List[str]:
    """
    Check for common data quality issues across pipeline stages.
    
    Returns:
        List of data quality issue descriptions
    """
    issues = []
    
    # Check for inconsistent cell counts across stages
    cell_counts = {}
    for file_path, result in results.items():
        filename = Path(file_path).name
        unique_cells = result.info.get("Unique cell IDs")
        if unique_cells:
            cell_counts[filename] = unique_cells
    
    if len(set(cell_counts.values())) > 1:
        issues.append(f"Inconsistent cell counts across files: {cell_counts}")
    
    # Check for files with validation errors
    error_files = [Path(fp).name for fp, r in results.items() if not r.is_valid]
    if error_files:
        issues.append(f"Files with validation errors: {error_files}")
    
    # Check for unusual timestamp ranges
    timestamp_ranges = {}
    for file_path, result in results.items():
        filename = Path(file_path).name
        ts_range = result.info.get("Timestamp range")
        if ts_range:
            timestamp_ranges[filename] = ts_range
    
    if len(set(timestamp_ranges.values())) > 1:
        issues.append(f"Inconsistent timestamp ranges: {timestamp_ranges}")
    
    # Check for missing critical files
    critical_files = ["02_preprocessed_data.parquet", "04_individual_anomalies.parquet"]
    available_files = [Path(fp).name for fp in results.keys()]
    missing_critical = [f for f in critical_files if f not in available_files]
    if missing_critical:
        issues.append(f"Missing critical pipeline files: {missing_critical}")
    
    return issues

def main():
    """Main function for pipeline validation."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Pipeline Data Validation")
    parser.add_argument("--base_path", default="results/", help="Base directory containing pipeline outputs")
    parser.add_argument("--geojson", help="Path to GeoJSON file for cell ID validation")
    parser.add_argument("--no_report", action="store_true", help="Skip detailed individual file reports")
    parser.add_argument("--quality_check", action="store_true", help="Perform additional data quality checks")
    
    args = parser.parse_args()
    
    # Validate pipeline outputs
    results = validate_pipeline_outputs(
        base_path=args.base_path,
        geojson_path=args.geojson,
        generate_report=not args.no_report
    )
    
    if not results:
        print("❌ No pipeline files found to validate!")
        exit(1)
    
    # Generate summary report
    print("="*80)
    print("OVERALL VALIDATION SUMMARY")
    print("="*80)
    
    summary = generate_summary_report(results)
    summary.print_report()
    
    # Perform data quality checks if requested
    if args.quality_check:
        print("\n" + "="*80)
        print("DATA QUALITY ANALYSIS")
        print("="*80)
        
        issues = check_data_quality_issues(results)
        if issues:
            print("🚨 DATA QUALITY ISSUES DETECTED:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ No data quality issues detected")
    
    # Exit with appropriate code
    exit_code = 0 if summary.is_valid else 1
    
    print(f"\nPipeline Validation: {'✅ PASSED' if summary.is_valid else '❌ FAILED'}")
    if not summary.is_valid:
        print(f"Found {len(summary.errors)} errors across {len(results)} files")
    
    exit(exit_code)

if __name__ == "__main__":
    main()
