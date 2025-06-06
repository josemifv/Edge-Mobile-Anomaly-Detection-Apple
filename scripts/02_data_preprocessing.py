import pandas as pd
import os
import argparse
import time
import multiprocessing
from datetime import datetime
from pathlib import Path

# --- Definición de Nombres de Columnas ---
COLUMN_NAMES = [
    'cell_id',
    'timestamp', 
    'country_code',
    'sms_in',
    'sms_out',
    'call_in',
    'call_out',
    'internet_traffic'
]

# Columnas numéricas para agregación
NUMERIC_COLUMNS = ['sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']

# --- 1. Función para Cargar un Único Archivo ---
def load_single_file(file_path):
    """
    Carga un archivo de datos de telecomunicaciones con formato predefinido.
    
    Args:
        file_path (str): Ruta al archivo de texto.
    
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados.
    """
    print(f"Cargando archivo: {file_path}")
    
    try:
        # Leer archivo con separador de espacios
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            names=COLUMN_NAMES,
            na_filter=True,
            dtype={
                'cell_id': 'Int64',
                'timestamp': 'Int64', 
                'country_code': 'Int64',
                'sms_in': float,
                'sms_out': float,
                'call_in': float,
                'call_out': float,
                'internet_traffic': float
            }
        )
        
        print(f"  Filas cargadas: {len(df):,}")
        return df
        
    except Exception as e:
        print(f"  Error al cargar {file_path}: {e}")
        return None

# --- 2. Función para Preprocesar un DataFrame ---
def preprocess_dataframe(df):
    """
    Aplica los pasos de preprocesamiento a un DataFrame:
    1. Elimina la columna country_code
    2. Convierte timestamp de milisegundos a datetime
    3. Agrupa por cell_id y timestamp y suma las columnas numéricas
    
    Args:
        df (pandas.DataFrame): DataFrame original.
    
    Returns:
        pandas.DataFrame: DataFrame preprocesado.
    """
    if df is None or df.empty:
        return None
        
    print(f"  Preprocesando DataFrame con {len(df):,} filas...")
    
    # Paso 1: Eliminar columna country_code
    df_processed = df.drop('country_code', axis=1)
    
    # Paso 2: Convertir timestamp de milisegundos a datetime
    df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], unit='ms')
    
    # Paso 3: Agrupar por cell_id y timestamp, y sumar las columnas numéricas
    print("  Agregando datos por cell_id y timestamp...")
    df_aggregated = df_processed.groupby(['cell_id', 'timestamp']).agg({
        'sms_in': 'sum',
        'sms_out': 'sum', 
        'call_in': 'sum',
        'call_out': 'sum',
        'internet_traffic': 'sum'
    }).reset_index()
    
    print(f"  Filas después de agregación: {len(df_aggregated):,}")
    return df_aggregated

# --- 3. Función para Validar el DataFrame Final ---
def validate_dataframe(df):
    """
    Valida el DataFrame final según los criterios especificados:
    1. No debe haber duplicados en la combinación cell_id + timestamp
    2. No debe haber valores nulos
    
    Args:
        df (pandas.DataFrame): DataFrame a validar.
    
    Raises:
        ValueError: Si la validación falla.
    """
    print("\n--- Validando DataFrame Final ---")
    
    # Verificar duplicados
    print("Verificando duplicados...")
    duplicates = df.duplicated(subset=['cell_id', 'timestamp']).sum()
    if duplicates > 0:
        raise ValueError(f"Se encontraron {duplicates} filas duplicadas basadas en cell_id + timestamp")
    print(f"✓ Sin duplicados encontrados")
    
    # Verificar valores nulos
    print("Verificando valores nulos...")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        print("Conteo de valores nulos por columna:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        raise ValueError(f"Se encontraron {total_nulls} valores nulos en el dataset")
    print(f"✓ Sin valores nulos encontrados")
    
    print("✓ Validación completada exitosamente")

# --- 4. Función Principal de Preprocesamiento ---
def preprocess_telecom_data(input_path, output_path=None, max_workers=None):
    """
    Función principal que procesa todos los archivos de telecomunicaciones.
    
    Args:
        input_path (str): Directorio con archivos .txt de datos.
        output_path (str, optional): Ruta para guardar el DataFrame consolidado.
        max_workers (int, optional): Número máximo de procesos paralelos.
    
    Returns:
        pandas.DataFrame: DataFrame consolidado y preprocesado.
    """
    start_time = time.perf_counter()
    print(f"\n=== Iniciando Preprocesamiento de Datos de Telecomunicaciones ===")
    print(f"Directorio de entrada: {input_path}")
    
    # Encontrar todos los archivos .txt
    input_dir = Path(input_path)
    txt_files = list(input_dir.glob('*.txt'))
    
    if not txt_files:
        raise FileNotFoundError(f"No se encontraron archivos .txt en {input_path}")
    
    print(f"Archivos encontrados: {len(txt_files)}")
    
    # Determinar número de workers
    if max_workers is None:
        max_workers = min(len(txt_files), multiprocessing.cpu_count())
    
    print(f"Usando {max_workers} procesos paralelos")
    
    # Cargar archivos en paralelo
    print("\n--- Cargando Archivos ---")
    with multiprocessing.Pool(max_workers) as pool:
        dataframes = pool.map(load_single_file, txt_files)
    
    # Filtrar DataFrames válidos
    valid_dataframes = [df for df in dataframes if df is not None]
    
    if not valid_dataframes:
        raise RuntimeError("No se pudieron cargar archivos válidos")
    
    print(f"\nArchivos cargados exitosamente: {len(valid_dataframes)}/{len(txt_files)}")
    
    # Preprocesar cada DataFrame
    print("\n--- Preprocesando Archivos ---")
    processed_dataframes = []
    
    for i, df in enumerate(valid_dataframes, 1):
        print(f"Preprocesando archivo {i}/{len(valid_dataframes)}")
        processed_df = preprocess_dataframe(df)
        if processed_df is not None:
            processed_dataframes.append(processed_df)
    
    if not processed_dataframes:
        raise RuntimeError("No se pudieron preprocesar archivos válidos")
    
    # Consolidar todos los DataFrames
    print("\n--- Consolidando Datos ---")
    print("Concatenando DataFrames...")
    consolidated_df = pd.concat(processed_dataframes, ignore_index=True)
    
    print(f"Filas totales antes de consolidación: {sum(len(df) for df in processed_dataframes):,}")
    print(f"Filas después de concatenación: {len(consolidated_df):,}")
    
    # Validar el resultado final
    validate_dataframe(consolidated_df)
    
    # Guardar resultado si se especifica ruta de salida
    if output_path:
        print(f"\n--- Guardando Datos ---")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.suffix.lower() == '.parquet':
            consolidated_df.to_parquet(output_path, index=False)
            print(f"Datos guardados en formato Parquet: {output_path}")
        else:
            consolidated_df.to_csv(output_path, index=False)
            print(f"Datos guardados en formato CSV: {output_path}")
    
    # Reporte final
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"\n=== Preprocesamiento Completado ===")
    print(f"Tiempo total de ejecución: {execution_time:.4f} segundos")
    print(f"Archivos procesados: {len(processed_dataframes)}")
    print(f"Filas finales: {len(consolidated_df):,}")
    print(f"Columnas finales: {list(consolidated_df.columns)}")
    print(f"Rango de fechas: {consolidated_df['timestamp'].min()} a {consolidated_df['timestamp'].max()}")
    print(f"Número de celdas únicas: {consolidated_df['cell_id'].nunique():,}")
    
    # Generar reporte de rendimiento
    generate_performance_report(input_path, output_path, execution_time, 
                              len(processed_dataframes), len(consolidated_df), max_workers)
    
    return consolidated_df

# --- 5. Función para Generar Reporte de Rendimiento ---
def generate_performance_report(input_path, output_path, execution_time, files_processed, total_rows, workers_used):
    """
    Genera un reporte detallado del rendimiento del preprocesamiento.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"preprocessing_report_{timestamp}.txt"
    report_path = Path("outputs") / report_filename
    
    # Crear directorio si no existe
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("--- Reporte de Rendimiento de Preprocesamiento de Datos ---\n")
        f.write(f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Directorio de Entrada: {input_path}\n")
        f.write(f"Archivo de Salida: {output_path or 'No especificado'}\n")
        f.write(f"Número de Procesos Utilizados: {workers_used}\n")
        f.write(f"Número de Archivos Procesados: {files_processed}\n")
        f.write(f"Número Total de Filas Finales: {total_rows}\n")
        f.write(f"Tiempo Total de Ejecución: {execution_time:.4f} segundos\n")
        f.write(f"Velocidad de Procesamiento: {total_rows/execution_time:.0f} filas/segundo\n")
    
    print(f"Reporte de rendimiento guardado: {report_path}")

# --- 6. Función Principal para CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos de telecomunicaciones con agregación y validación"
    )
    
    parser.add_argument(
        "input_path",
        help="Directorio que contiene los archivos .txt de datos de telecomunicaciones"
    )
    
    parser.add_argument(
        "--output_path", 
        help="Ruta del archivo de salida (.csv o .parquet)"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Número máximo de procesos paralelos (por defecto: número de CPUs)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mostrar preview del DataFrame final"
    )
    
    args = parser.parse_args()
    
    try:
        # Ejecutar preprocesamiento
        result_df = preprocess_telecom_data(
            input_path=args.input_path,
            output_path=args.output_path,
            max_workers=args.max_workers
        )
        
        # Mostrar preview si se solicita
        if args.preview:
            print("\n--- Preview del DataFrame Final ---")
            print(f"Forma: {result_df.shape}")
            print(f"\nPrimeras 10 filas:")
            print(result_df.head(10))
            print(f"\nÚltimas 10 filas:")
            print(result_df.tail(10))
            print(f"\nInformación del DataFrame:")
            print(result_df.info())
            print(f"\nEstadísticas descriptivas:")
            print(result_df.describe())
        
        print("\n✓ Preprocesamiento completado exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error durante el preprocesamiento: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

