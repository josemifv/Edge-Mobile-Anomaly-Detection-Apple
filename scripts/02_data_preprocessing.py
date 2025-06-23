#!/usr/bin/env python3
"""
02_data_preprocessing.py

CMMSE 2025: Stage 2 - Data Preprocessing and Aggregation
Performs high-performance data aggregation and validation on ingested data using Polars.
"""

import polars as pl
import argparse
import time
from pathlib import Path

def run_preprocessing_stage(input_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Preprocesses the data by aggregating, merging columns, and validating.

    Args:
        input_path: Path to the ingested Parquet file from Stage 1.
        output_path: Path to save the preprocessed Parquet file.

    Returns:
        A Polars DataFrame with the preprocessed data.
    """
    print(f"Starting lazy preprocessing from: {input_path}")

    # 1. Iniciar un plan "lazy" leyendo el Parquet. No se carga en memoria aún.
    lazy_df = pl.scan_parquet(input_path)

    # 2. Encadenar todas las operaciones de forma declarativa.
    preprocessed_lazy_df = (
        lazy_df
        # Agrupa por cell_id y timestamp
        .group_by(['cell_id', 'timestamp'])
        # Agrega las columnas de tráfico sumándolas
        .agg([
            pl.sum('sms_in'),
            pl.sum('sms_out'),
            pl.sum('call_in'),
            pl.sum('call_out'),
            pl.sum('internet_traffic')
        ])
        # Crea las nuevas columnas 'sms_total' y 'calls_total'
        .with_columns([
            (pl.col('sms_in') + pl.col('sms_out')).alias('sms_total'),
            (pl.col('call_in') + pl.col('call_out')).alias('calls_total')
        ])
        # Selecciona las columnas finales, descartando las originales
        .select(['cell_id', 'timestamp', 'sms_total', 'calls_total', 'internet_traffic'])
    )

    # 3. Ejecutar el plan optimizado y materializar el resultado.
    print("Executing the optimized query plan...")
    final_df = preprocessed_lazy_df.collect(engine='streaming')

    # 4. Validar los datos resultantes
    print("Validating data quality...")
    
    # Comprobar duplicados
    duplicates = final_df.select(pl.struct(['cell_id', 'timestamp']).is_duplicated()).sum().item()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate rows based on cell_id + timestamp")
    print("  ✓ No duplicates found")

    # Comprobar nulos
    null_counts = final_df.null_count()
    total_nulls = null_counts.sum_horizontal().item()
    if total_nulls > 0:
        raise ValueError(f"Found {total_nulls} null values in the dataset")
    print("  ✓ No null values found")

    # 5. Guardar el resultado final
    print(f"Saving preprocessed data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(output_path)
    
    return final_df

def main():
    """Main function to run the data preprocessing stage."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 2 - Data Preprocessing")
    parser.add_argument("input_path", type=Path, help="Path to ingested data file from Stage 1")
    parser.add_argument("--output_path", type=Path, default="outputs/02_preprocessed_data.parquet", help="Output file path")
    args = parser.parse_args()

    print("="*60)
    print("Stage 2: Data Preprocessing")
    print("="*60)
    
    start_time = time.perf_counter()
    
    final_df = run_preprocessing_stage(args.input_path, args.output_path)
    
    total_time = time.perf_counter() - start_time

    print("\n--- STAGE 2 PERFORMANCE SUMMARY ---")
    print(f"Output rows: {len(final_df):,}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Processing rate: {len(final_df) / total_time:,.0f} rows/second")
    print("✅ Stage 2 completed successfully.")

if __name__ == "__main__":
    main()
