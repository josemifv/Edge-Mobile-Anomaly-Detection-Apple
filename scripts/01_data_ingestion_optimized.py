import pandas as pd
import os
import argparse
import time
import multiprocessing
import psutil
from datetime import datetime
from pathlib import Path

# --- Configuración de Optimización ---
OPTIMIZED_DTYPES = {
    'square_id': 'uint16',      # Optimized for values < 65,536
    'timestamp_ms': 'int64',    # Keep as is for timestamp precision
    'country_code': 'uint8',    # Optimized for limited country codes
    'sms_in': 'float32',        # Reduced precision
    'sms_out': 'float32',
    'call_in': 'float32',
    'call_out': 'float32',
    'internet_activity': 'float32'
}

COLUMN_NAMES = [
    'square_id',
    'timestamp_ms',
    'country_code',
    'sms_in',
    'sms_out',
    'call_in',
    'call_out',
    'internet_activity'
]

# --- Funciones de Optimización ---
def get_optimal_chunk_size(file_path, target_memory_mb=200):
    """
    Calcula el tamaño de chunk óptimo basado en el tamaño del archivo.
    """
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # Estimar filas basado en tamaño promedio por fila (~80 bytes)
        estimated_rows = int(file_size_mb * 1024 * 1024 / 80)
        
        # Calcular chunk size para mantenerse dentro del target de memoria
        optimal_chunk = min(estimated_rows, int(target_memory_mb * 1024 * 1024 / 80))
        
        # Asegurar un mínimo razonable
        return max(optimal_chunk, 50_000)
    except:
        return 500_000  # Default fallback

def get_optimal_process_count(num_files, memory_gb_available=None):
    """
    Determina el número óptimo de procesos basado en hardware y carga.
    """
    cpu_count = os.cpu_count() or 1
    
    # Detectar carga actual del sistema
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    # Ajustar basado en carga del sistema
    if cpu_percent > 80:
        adjustment = -2  # Sistema ocupado
    elif cpu_percent < 30:
        adjustment = 2   # Sistema libre, usar más cores
    else:
        adjustment = 0
    
    # Limitación por memoria si se especifica
    if memory_gb_available:
        memory_limited_processes = int(memory_gb_available / 2)  # ~2GB per process
        return min(num_files, cpu_count + adjustment, memory_limited_processes)
    
    return min(num_files, cpu_count + adjustment)

def monitor_system_resources():
    """
    Monitorea recursos del sistema en tiempo real.
    """
    return {
        'cpu_percent': psutil.cpu_percent(interval=0),
        'memory_percent': psutil.virtual_memory().percent,
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024**2)
    }

# --- Función Optimizada para Cargar un Archivo ---
def load_and_preprocess_single_file_optimized(
    file_path, 
    chunk_size=None, 
    convert_timestamp=True, 
    reorder_cols=True,
    use_optimized_dtypes=True,
    enable_memory_mapping=True
):
    """
    Versión optimizada de carga y preprocesamiento de archivos.
    """
    print(f"\n--- Procesando archivo: {Path(file_path).name} ---")
    start_file_time = time.perf_counter()
    
    # Seleccionar tipos de datos
    dtypes = OPTIMIZED_DTYPES if use_optimized_dtypes else {
        COLUMN_NAMES[0]: 'Int64',
        COLUMN_NAMES[1]: 'Int64',
        COLUMN_NAMES[2]: 'Int64',
        COLUMN_NAMES[3]: float,
        COLUMN_NAMES[4]: float,
        COLUMN_NAMES[5]: float,
        COLUMN_NAMES[6]: float,
        COLUMN_NAMES[7]: float,
    }
    
    # Parámetros optimizados de lectura
    optimized_read_params = {
        "sep": r'\s+',
        "header": None,
        "names": COLUMN_NAMES,
        "dtype": dtypes,
        "engine": 'c',                    # Usar motor C (más rápido)
        "memory_map": enable_memory_mapping,  # Memory mapping para archivos grandes
        "na_filter": True,                # Mantener filtrado NA para datos reales
        "low_memory": False,              # Leer archivo completo en memoria
    }
    
    try:
        # Determinar chunk size automáticamente si no se especifica
        if chunk_size is None:
            chunk_size = get_optimal_chunk_size(file_path)
            auto_chunk = True
        else:
            auto_chunk = False
        
        if chunk_size and chunk_size > 0:
            if auto_chunk:
                print(f"Usando chunk size automático: {chunk_size:,}")
            else:
                print(f"Usando chunk size especificado: {chunk_size:,}")
            
            chunks = []
            chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, **optimized_read_params)
            
            for chunk_num, chunk in enumerate(chunk_reader):
                chunks.append(chunk)
                
            df = pd.concat(chunks, ignore_index=True)
        else:
            print("Leyendo archivo completo de una vez.")
            df = pd.read_csv(file_path, **optimized_read_params)
        
        print(f"Datos cargados: {df.shape[0]:,} filas, {df.shape[1]} columnas, "
              f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
        return None
    
    # --- Preprocesamiento Optimizado ---
    if convert_timestamp:
        if 'timestamp_ms' in df.columns:
            # Conversión optimizada de timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
            
            # Verificación optimizada de errores
            null_timestamps = df['timestamp'].isnull().sum()
            if null_timestamps > 0:
                print(f"Advertencia: {null_timestamps} timestamps no pudieron ser convertidos.")
            
            df = df.drop(columns=['timestamp_ms'])
    
    if reorder_cols and 'timestamp' in df.columns:
        # Reordenamiento optimizado
        cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
        df = df[cols]
    
    end_file_time = time.perf_counter()
    processing_time = end_file_time - start_file_time
    rows_per_second = len(df) / processing_time if processing_time > 0 else 0
    
    print(f"Completado en {processing_time:.4f}s ({rows_per_second:,.0f} filas/seg)")
    
    return df

# --- Función Optimizada para Procesar Directorio ---
def process_directory_optimized(
    directory_path, 
    chunk_size=None, 
    convert_timestamp=True, 
    reorder_cols=True,
    max_workers=None,
    use_optimized_dtypes=True,
    enable_memory_mapping=True,
    batch_processing=False
):
    """
    Versión optimizada del procesamiento de directorio con monitoreo de recursos.
    """
    # Encontrar archivos
    files_to_process = [
        f for f in os.listdir(directory_path) 
        if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))
    ]
    
    if not files_to_process:
        print(f"No se encontraron archivos .txt en {directory_path}")
        return [], 0
    
    print(f"Encontrados {len(files_to_process)} archivos para procesar")
    
    # Monitorear recursos del sistema
    initial_resources = monitor_system_resources()
    print(f"Recursos iniciales - CPU: {initial_resources['cpu_percent']:.1f}%, "
          f"Memoria: {initial_resources['memory_percent']:.1f}%, "
          f"Memoria disponible: {initial_resources['available_memory_gb']:.1f} GB")
    
    # Determinar número óptimo de procesos
    if max_workers is None:
        optimal_workers = get_optimal_process_count(
            len(files_to_process), 
            initial_resources['available_memory_gb']
        )
    else:
        optimal_workers = max_workers
    
    print(f"Usando {optimal_workers} procesos paralelos")
    
    # Preparar argumentos para procesamiento
    tasks = [
        (
            os.path.join(directory_path, filename),
            chunk_size,
            convert_timestamp,
            reorder_cols,
            use_optimized_dtypes,
            enable_memory_mapping
        ) 
        for filename in files_to_process
    ]
    
    start_time = time.perf_counter()
    
    try:
        # Configuración optimizada del pool de procesos
        with multiprocessing.Pool(
            processes=optimal_workers,
            maxtasksperchild=1  # Prevenir memory leaks
        ) as pool:
            
            if batch_processing and len(files_to_process) > optimal_workers * 2:
                # Procesamiento por lotes para datasets muy grandes
                print("Usando procesamiento por lotes")
                batch_size = optimal_workers
                all_results = []
                
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    print(f"Procesando lote {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
                    
                    batch_results = pool.starmap(
                        load_and_preprocess_single_file_optimized, 
                        batch
                    )
                    all_results.extend(batch_results)
                    
                    # Monitorear recursos entre lotes
                    current_resources = monitor_system_resources()
                    if current_resources['memory_percent'] > 85:
                        print(f"Advertencia: Uso de memoria alto ({current_resources['memory_percent']:.1f}%)")
                
                results = all_results
            else:
                # Procesamiento estándar
                results = pool.starmap(
                    load_and_preprocess_single_file_optimized, 
                    tasks
                )
        
        # Filtrar resultados exitosos
        successful_dataframes = [df for df in results if df is not None]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Estadísticas de rendimiento
        successful_files = len(successful_dataframes)
        failed_files = len(results) - successful_files
        total_rows = sum(len(df) for df in successful_dataframes)
        
        print(f"\n--- Resumen de Procesamiento ---")
        print(f"Archivos exitosos: {successful_files}/{len(files_to_process)}")
        print(f"Archivos fallidos: {failed_files}")
        print(f"Total de filas procesadas: {total_rows:,}")
        print(f"Tiempo total: {total_time:.4f} segundos")
        print(f"Velocidad promedio: {total_rows/total_time:,.0f} filas/segundo")
        
        # Recursos finales
        final_resources = monitor_system_resources()
        print(f"Recursos finales - CPU: {final_resources['cpu_percent']:.1f}%, "
              f"Memoria: {final_resources['memory_percent']:.1f}%")
        
        return successful_dataframes, optimal_workers
        
    except Exception as e:
        print(f"Error durante el procesamiento paralelo: {e}")
        return [], 0

# --- Función Principal Optimizada ---
def main():
    parser = argparse.ArgumentParser(
        description="Carga optimizada de datos de telecomunicaciones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_path", 
        help="Ruta a archivo o directorio con archivos .txt"
    )
    
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=None,
        help="Tamaño de chunks (auto si no se especifica)"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=None,
        help="Número máximo de procesos (auto si no se especifica)"
    )
    
    parser.add_argument(
        "--no_timestamp_conversion", 
        action="store_false", 
        dest="convert_timestamp",
        help="Desactivar conversión de timestamp"
    )
    
    parser.add_argument(
        "--no_reorder_columns", 
        action="store_false", 
        dest="reorder_cols",
        help="Desactivar reordenamiento de columnas"
    )
    
    parser.add_argument(
        "--no_dtype_optimization", 
        action="store_false", 
        dest="use_optimized_dtypes",
        help="Desactivar optimización de tipos de datos"
    )
    
    parser.add_argument(
        "--no_memory_mapping", 
        action="store_false", 
        dest="enable_memory_mapping",
        help="Desactivar memory mapping"
    )
    
    parser.add_argument(
        "--batch_processing", 
        action="store_true",
        help="Activar procesamiento por lotes"
    )
    
    parser.add_argument(
        "--output_summary", 
        action="store_true",
        help="Mostrar resumen detallado"
    )
    
    args = parser.parse_args()
    
    overall_start_time = time.perf_counter()
    
    print("=== Ingesta Optimizada de Datos de Telecomunicaciones ===")
    print(f"Configuración de optimización:")
    print(f"  - Tipos de datos optimizados: {args.use_optimized_dtypes}")
    print(f"  - Memory mapping: {args.enable_memory_mapping}")
    print(f"  - Procesamiento por lotes: {args.batch_processing}")
    print(f"  - Chunk size: {'Automático' if args.chunk_size is None else args.chunk_size}")
    
    final_dataframes = []
    num_processes_used = 1
    
    if os.path.isdir(args.input_path):
        print(f"\nProcesando directorio: {args.input_path}")
        final_dataframes, num_processes_used = process_directory_optimized(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            max_workers=args.max_workers,
            use_optimized_dtypes=args.use_optimized_dtypes,
            enable_memory_mapping=args.enable_memory_mapping,
            batch_processing=args.batch_processing
        )
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.txt'):
        print(f"\nProcesando archivo único: {args.input_path}")
        df = load_and_preprocess_single_file_optimized(
            args.input_path,
            chunk_size=args.chunk_size,
            convert_timestamp=args.convert_timestamp,
            reorder_cols=args.reorder_cols,
            use_optimized_dtypes=args.use_optimized_dtypes,
            enable_memory_mapping=args.enable_memory_mapping
        )
        if df is not None:
            final_dataframes.append(df)
    else:
        print(f"Error: '{args.input_path}' no es un archivo .txt válido ni un directorio")
        return
    
    overall_end_time = time.perf_counter()
    total_execution_time = overall_end_time - overall_start_time
    
    if not final_dataframes:
        print("\nNo se procesaron datos exitosamente.")
        return
    
    # Estadísticas finales
    total_rows_processed = sum(len(df) for df in final_dataframes)
    num_files_processed = len(final_dataframes)
    
    print(f"\n=== Resumen Final ===")
    print(f"Archivos procesados: {num_files_processed}")
    print(f"Filas totales: {total_rows_processed:,}")
    print(f"Tiempo total: {total_execution_time:.4f} segundos")
    print(f"Velocidad global: {total_rows_processed/total_execution_time:,.0f} filas/segundo")
    
    # Generar reporte de rendimiento optimizado
    report_content = (
        f"--- Reporte de Rendimiento de Ingesta Optimizada ---\n"
        f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Directorio de Entrada: {args.input_path}\n"
        f"Configuración:\n"
        f"  - Chunk Size: {args.chunk_size if args.chunk_size else 'Automático'}\n"
        f"  - Tipos Optimizados: {args.use_optimized_dtypes}\n"
        f"  - Memory Mapping: {args.enable_memory_mapping}\n"
        f"  - Procesamiento por Lotes: {args.batch_processing}\n"
        f"  - Conversión Timestamp: {args.convert_timestamp}\n"
        f"  - Reordenamiento Columnas: {args.reorder_cols}\n"
        f"Número de Procesos Utilizados: {num_processes_used}\n"
        f"Número de Archivos Procesados: {num_files_processed}\n"
        f"Número Total de Filas Procesadas: {total_rows_processed}\n"
        f"Tiempo Total de Ejecución: {total_execution_time:.4f} segundos\n"
        f"Velocidad de Procesamiento: {total_rows_processed/total_execution_time:.0f} filas/segundo\n"
    )
    
    # Guardar reporte
    os.makedirs("outputs", exist_ok=True)
    report_filename = f"outputs/ingestion_optimized_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_filename, 'w') as f:
            f.write(report_content)
        print(f"Reporte guardado en: {report_filename}")
    except IOError as e:
        print(f"Error al guardar reporte: {e}")
    
    # Mostrar resumen detallado si se solicita
    if args.output_summary and final_dataframes:
        print(f"\n--- Resumen Detallado ---")
        if len(final_dataframes) > 1:
            try:
                combined_df = pd.concat(final_dataframes, ignore_index=True)
                print(f"DataFrame combinado:")
                combined_df.info(memory_usage="deep")
                if 'timestamp' in combined_df.columns:
                    print(f"Rango de fechas: {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
            except MemoryError:
                print("Error de memoria al combinar DataFrames. Mostrando primer DataFrame:")
                final_dataframes[0].info(memory_usage="deep")
        else:
            final_dataframes[0].info(memory_usage="deep")
            if 'timestamp' in final_dataframes[0].columns:
                print(f"Rango de fechas: {final_dataframes[0]['timestamp'].min()} a {final_dataframes[0]['timestamp'].max()}")

if __name__ == '__main__':
    main()

