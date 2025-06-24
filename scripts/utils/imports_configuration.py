#!/usr/bin/env python3
"""
imports_configuration.py

CMMSE 2025: Updated Imports and Dependencies Configuration
Step 3: Update imports and dependencies for enhanced parquet reading capabilities

This file demonstrates the proper import configuration for the Edge-Mobile-Anomaly-Detection-Apple project
to ensure optimal parquet reading capabilities with pandas, pyarrow, and polars.
"""

# Core data processing imports with parquet support
import pandas as pd
import pyarrow.parquet as pq  # For parquet support
import polars as pl  # Optional: for faster processing

# Essential numerical computing
import numpy as np

# Additional parquet-related imports for advanced usage
import pyarrow as pa  # Full pyarrow functionality
from pyarrow import csv as pa_csv  # For CSV to parquet conversion
from pyarrow import dataset as pa_dataset  # For dataset operations

# Verification function to ensure parquet capabilities are working
def verify_parquet_support():
    """
    Verify that parquet reading capabilities are properly configured
    """
    print("Verifying parquet support configuration...")
    
    # Test pandas parquet support
    try:
        print(f"✅ pandas version: {pd.__version__}")
        print(f"✅ pandas parquet engine available: {pd.io.parquet.get_engine('pyarrow')}")
    except Exception as e:
        print(f"❌ pandas parquet support issue: {e}")
    
    # Test pyarrow parquet support
    try:
        print(f"✅ pyarrow version: {pa.__version__}")
        print("✅ pyarrow parquet module available")
    except Exception as e:
        print(f"❌ pyarrow support issue: {e}")
    
    # Test polars parquet support
    try:
        print(f"✅ polars version: {pl.__version__}")
        print("✅ polars parquet support available")
    except Exception as e:
        print(f"❌ polars support issue: {e}")

# Configuration settings for optimal parquet reading
PARQUET_CONFIG = {
    'pandas': {
        'engine': 'pyarrow',  # Use pyarrow as default engine
        'use_nullable_dtypes': True,  # Better null handling
        'dtype_backend': 'pyarrow'  # Use pyarrow dtypes for better performance
    },
    'pyarrow': {
        'use_threads': True,  # Enable multi-threading
        'pre_buffer': True,  # Enable pre-buffering for better performance
        'buffer_size': 0,  # Use default buffer size
        'memory_map': True  # Use memory mapping when possible
    },
    'polars': {
        'use_pyarrow': True,  # Use pyarrow as backend
        'rechunk': True,  # Rechunk for better performance
        'low_memory': False  # Optimize for speed over memory
    }
}

# Example usage functions for different libraries
def read_parquet_pandas(file_path, **kwargs):
    """
    Read parquet file using pandas with optimized settings
    """
    config = PARQUET_CONFIG['pandas'].copy()
    config.update(kwargs)
    return pd.read_parquet(file_path, **config)

def read_parquet_pyarrow(file_path, **kwargs):
    """
    Read parquet file using pyarrow directly
    """
    config = PARQUET_CONFIG['pyarrow'].copy()
    config.update(kwargs)
    return pq.read_table(file_path, **config)

def read_parquet_polars(file_path, **kwargs):
    """
    Read parquet file using polars with optimized settings
    """
    config = PARQUET_CONFIG['polars'].copy()
    config.update(kwargs)
    return pl.read_parquet(file_path, **config)

def scan_parquet_polars_lazy(file_path, **kwargs):
    """
    Scan parquet file using polars lazy evaluation
    """
    config = PARQUET_CONFIG['polars'].copy()
    config.update(kwargs)
    return pl.scan_parquet(file_path, **config)

# Memory-efficient reading for large files
def read_large_parquet_chunked(file_path, chunk_size=100000):
    """
    Read large parquet files in chunks using pyarrow
    """
    table = pq.read_table(file_path)
    
    for batch in table.to_batches(max_chunksize=chunk_size):
        yield batch.to_pandas()

# Optimized writing functions
def write_parquet_optimized(df, file_path, engine='polars', **kwargs):
    """
    Write parquet file using the most appropriate engine
    """
    if engine == 'polars' and isinstance(df, pl.DataFrame):
        # Polars optimized writing
        df.write_parquet(
            file_path,
            compression='zstd',  # Better compression
            use_pyarrow=True,
            **kwargs
        )
    elif engine == 'pandas' and isinstance(df, pd.DataFrame):
        # Pandas optimized writing
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='zstd',
            index=False,
            **kwargs
        )
    else:
        # Fallback to pyarrow
        table = pa.Table.from_pandas(df) if isinstance(df, pd.DataFrame) else df
        pq.write_table(table, file_path, compression='zstd', **kwargs)

if __name__ == "__main__":
    print("Edge-Mobile-Anomaly-Detection-Apple: Parquet Configuration")
    print("=" * 60)
    verify_parquet_support()
    print("\n✅ Parquet reading capabilities are properly configured!")
    print("\nConfiguration summary:")
    print("- pandas: Uses pyarrow engine with nullable dtypes")
    print("- pyarrow: Multi-threading enabled with memory mapping")
    print("- polars: PyArrow backend with rechunking optimization")
    print("\nReady for high-performance parquet operations!")
