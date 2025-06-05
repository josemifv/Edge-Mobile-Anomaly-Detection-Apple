import pandas as pd
import os
import argparse
import time
import multiprocessing
from datetime import datetime

# --- Definición de Nombres de Columnas ---
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

# --- 1. Función para Cargar y Preprocesar un ÚNICO Archivo CSV ---
def load_and_preprocess_single_file(file_path, chunk_size=None, convert_timestamp=True, reorder_cols=True):
    """
    Carga y preprocesa datos desde un único archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.
        chunk_size (int, optional): Tamaño de los trozos para leer el CSV.
                                    Si es None, lee el archivo completo.
        convert_timestamp (bool): Si es True, convierte la columna de timestamp.
        reorder_cols (bool): Si es True, reordena las columnas después del preprocesamiento.


    Returns:
        pandas.DataFrame: DataFrame con los datos cargados y preprocesados.
                          Devuelve None si ocurre un error.
    """
    print(f"\n--- Procesando archivo: {file_path} ---")
    start_file_time = time.perf_counter()

    try:
        common_read_params = {
            "sep": r'\s+',  # Use regex for one or more spaces, handles the deprecation
            "header": None,
            "names": COLUMN_NAMES,
            "na_filter": True,
            "dtype": {
                COLUMN_NAMES[0]: 'Int64',
                COLUMN_NAMES[1]: 'Int64',
                COLUMN_NAMES[2]: 'Int64',
                COLUMN_NAMES[3]: float,
                COLUMN_NAMES[4]: float,
                COLUMN_NAMES[5]: float,
                COLUMN_NAMES[6]: float,
                COLUMN_NAMES[7]: float,
            }
        }

        if chunk_size and chunk_size > 0:
            print(f"Leyendo en trozos de tamaño: {chunk_size}")
            chunks = []
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, **common_read_params)):
                # El preprocesamiento se podría hacer por chunk aquí si es muy intensivo
                # print(f"Procesando trozo {chunk_num + 1} con {len(chunk)} filas...")
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            print("Leyendo el archivo completo de una vez.")
            df = pd.read_csv(file_path, **common_read_params)

        print(f"Datos crudos cargados de {os.path.basename(file_path)}: {df.shape[0]} filas y {df.shape[1]} columnas.")

    except pd.errors.ParserError as e:
        print(f"Error al parsear el archivo {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {file_path}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar {file_path}: {e}")
        return None

    # --- Preprocesamiento Adicional ---
    if convert_timestamp:
        if 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
            if df['timestamp'].isnull().any() and df['timestamp_ms'].notnull().any():
                print(f"Advertencia en {os.path.basename(file_path)}: Algunos timestamps no pudieron ser convertidos.")
            df = df.drop(columns=['timestamp_ms']) # Eliminar columna original
        else:
            print(f"Advertencia: Columna 'timestamp_ms' no encontrada en {os.path.basename(file_path)} para convertir.")


    if reorder_cols:
        if 'timestamp' in df.columns:
            cols = ['timestamp'] + [col for col in df.columns if col != 'timestamp']
            df = df[cols]

    end_file_time = time.perf_counter()
    print(f"Preprocesamiento de {os.path.basename(file_path)} completado en {end_file_time - start_file_time:.4f} segundos.")
    # df.info() # Descomentar para ver info detallada por archivo
    return df

# --- 2. Función para Procesar un Directorio de Archivos CSV ---
def process_directory(directory_path, chunk_size=None, convert_timestamp=True, reorder_cols=True):
    """
    Procesa todos los archivos CSV en el directorio especificado.

    Args:
        directory_path (str): Ruta al directorio que contiene los archivos CSV.
        chunk_size (int, optional): Tamaño de los trozos para leer cada CSV.
        convert_timestamp (bool): Controla la conversión de timestamp.
        reorder_cols (bool): Controla el reordenamiento de columnas.


    Returns:
        list: Una lista de DataFrames, uno por cada archivo CSV procesado exitosamente.
              Los archivos que fallen se omitirán de la lista.
    """
    all_dataframes = []
    files_to_process = [f for f in os.listdir(directory_path) if f.endswith('.txt') and os.path.isfile(os.path.join(directory_path, f))]
    # files_to_process = [\"sms-call-internet-mi-2013-11-01.txt\", \"sms-call-internet-mi-2013-11-02.txt\"] # Temporary for testing

    if not files_to_process:
        print(f"No se encontraron archivos .txt en el directorio: {directory_path}") # Actualizado a .txt
        return all_dataframes

    print(f"Se encontraron {len(files_to_process)} archivos .txt para procesar en '{directory_path}'.") # Actualizado a .txt

    # Preparar argumentos para starmap
    tasks = [(os.path.join(directory_path, filename), chunk_size, convert_timestamp, reorder_cols) for filename in files_to_process]

    # Determinar el número de procesos (puedes ajustarlo)
    # Usar os.cpu_count() puede ser una buena heurística, pero considera la E/S y la memoria
    num_processes_used = min(len(files_to_process), os.cpu_count() or 1)
    print(f"Usando {num_processes_used} procesos para la carga paralela...")

    try:
        with multiprocessing.Pool(processes=num_processes_used) as pool:
            results = pool.starmap(load_and_preprocess_single_file, tasks)

        # Filtrar los None (archivos que fallaron al procesar)
        all_dataframes = [df for df in results if df is not None]

        # La impresión detallada por archivo ahora se hace dentro de load_and_preprocess_single_file
        # Aquí solo confirmamos el número total de DataFrames recuperados del pool
        successful_files = sum(1 for df in results if df is not None)
        failed_files = len(results) - successful_files
        print(f"Procesamiento paralelo completado. {successful_files} archivos procesados exitosamente, {failed_files} fallaron.")

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento paralelo: {e}")
        # Podrías querer un manejo más granular o reintentar aquí
        return [] # Devolver lista vacía en caso de error mayor en el pool

    return all_dataframes, num_processes_used


# --- Función Principal para Ejecutar el Script ---
def main():
    parser = argparse.ArgumentParser(description="Carga y preprocesa datos de telecomunicaciones desde archivos CSV.")
    parser.add_argument("input_path", type=str, help="Ruta a un archivo CSV o un directorio que contenga archivos CSV.")
    parser.add_argument("--chunk_size", type=int, default=None, help="Tamaño de los trozos para leer los CSV (ej. 100000). Si no se especifica, lee el archivo completo.")
    parser.add_argument("--no_timestamp_conversion", action="store_false", dest="convert_timestamp", help="Desactiva la conversión de la columna timestamp_ms a datetime.")
    parser.add_argument("--no_reorder_columns", action="store_false", dest="reorder_cols", help="Desactiva el reordenamiento de columnas.")
    parser.add_argument("--output_summary", action="store_true", help="Muestra un resumen detallado del DataFrame combinado (si se procesa un directorio) o del único DataFrame.")

    args = parser.parse_args()

    overall_start_time = time.perf_counter()

    final_dataframes = []
    num_processes_for_report = 1 # Default para procesamiento de archivo único
    if os.path.isdir(args.input_path):
        print(f"Procesando directorio: {args.input_path}")
        final_dataframes, num_processes_for_report = process_directory(args.input_path, args.chunk_size, args.convert_timestamp, args.reorder_cols)
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.txt'):
        print(f"Procesando archivo único: {args.input_path}")
        df = load_and_preprocess_single_file(args.input_path, args.chunk_size, args.convert_timestamp, args.reorder_cols)
        if df is not None:
            final_dataframes.append(df)
    else:
        print(f"Error: La ruta de entrada '{args.input_path}' no es un archivo CSV válido ni un directorio.")
        return

    overall_end_time = time.perf_counter()
    print(f"\n--- Procesamiento Total Completado ---")
    print(f"Tiempo total de ejecución: {overall_end_time - overall_start_time:.4f} segundos.")
    total_execution_time = overall_end_time - overall_start_time
    # La siguiente línea está duplicada y el manejo de fallo está ahora integrado con el reporte.
    # print(f"Tiempo total de ejecución: {total_execution_time:.4f} segundos.") 

    if not final_dataframes: # Si no se procesó ningún dataframe, se escribe un reporte de fallo
        print("No se procesaron datos exitosamente.")
        report_content = (
            f"--- Reporte de Fallo de Ingesta de Datos ---\
"
            f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\
"
            f"Directorio de Entrada: {args.input_path}\
"
            f"Chunk Size: {args.chunk_size if args.chunk_size else 'Completo'}\
"
            f"Resultado: Fallo - No se procesaron datos.\
"
            f"Tiempo total de ejecución: {total_execution_time:.4f} segundos\
"
        )
        report_filename_fail = f"outputs/ingestion_report_FAILURE_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_filename_fail, 'w') as f_report:
                f_report.write(report_content)
            print(f"Reporte de fallo guardado en: {report_filename_fail}")
        except IOError as e:
            print(f"Error al escribir el archivo de reporte de fallo: {e}")
        return
    
    # Si llegamos aquí, final_dataframes tiene al menos un elemento.
    print(f"Se procesaron {len(final_dataframes)} DataFrames en total.")
    total_rows_processed = 0
    if final_dataframes: # Asegurarse de que hay dataframes para contar filas
        if args.output_summary and 'combined_df' in locals() and combined_df is not None:
            # Si el output_summary ya combinó los DFs, usar la longitud de ese DF combinado.
            # Esto asume que combined_df se creó correctamente en el bloque de output_summary.
            total_rows_processed = len(combined_df)
        else:
            # Si no hay output_summary, o si combined_df no se creó (ej. MemoryError al combinar),
            # sumar las filas de los DataFrames individuales.
            total_rows_processed = sum(len(df) for df in final_dataframes)

    # Escribir el reporte de rendimiento

    # Escribir el reporte de rendimiento
    num_files_processed = len(final_dataframes)
    num_processes_actually_used = num_processes_for_report

    report_content = (
        f"--- Reporte de Rendimiento de Ingesta de Datos ---\n"
        f"Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Directorio de Entrada: {args.input_path}\n"
        f"Chunk Size: {args.chunk_size if args.chunk_size else 'Completo'}\n"
        f"Conversión de Timestamp: {args.convert_timestamp}\n"
        f"Reordenamiento de Columnas: {args.reorder_cols}\n"
        f"Número de Procesos Utilizados: {num_processes_actually_used}\n"
        f"Número de Archivos Procesados: {num_files_processed}\n"
        f"Número Total de Filas Procesadas: {total_rows_processed}\n"
        f"Tiempo Total de Ejecución: {total_execution_time:.4f} segundos\n"
    )

    report_filename = f"outputs/ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_filename, 'w') as f_report:
            f_report.write(report_content)
        print(f"Reporte de rendimiento guardado en: {report_filename}")
    except IOError as e:
        print(f"Error al escribir el archivo de reporte: {e}")

    print(f"Se procesaron {num_files_processed} DataFrames en total.")
    if args.output_summary:
        # Si quieres combinar todos los dataframes en uno para el resumen:
        if len(final_dataframes) > 1:
            print("\nCombinando todos los DataFrames procesados para el resumen...")
            # Ten cuidado con la memoria si los DataFrames son muy grandes
            # Se puede hacer un muestreo o procesar de otra forma si es necesario
            try:
                combined_df = pd.concat(final_dataframes, ignore_index=True)
                print("Resumen del DataFrame combinado:")
                combined_df.info(verbose=True, show_counts=True, memory_usage="deep")
                if not combined_df.empty and 'timestamp' in combined_df.columns:
                     print(f"Rango de fechas general: de {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
            except MemoryError:
                print("Error de memoria al intentar combinar todos los DataFrames para el resumen.")
                print("Mostrando resumen del primer DataFrame procesado en su lugar (si existe):")
                if final_dataframes:
                    final_dataframes[0].info(verbose=True, show_counts=True, memory_usage="deep")
            except Exception as e:
                print(f"Error al combinar DataFrames para el resumen: {e}")

        elif final_dataframes: # Solo un DataFrame
            print("\nResumen del DataFrame procesado:")
            final_dataframes[0].info(verbose=True, show_counts=True, memory_usage="deep")
            if not final_dataframes[0].empty and 'timestamp' in final_dataframes[0].columns:
                print(f"Rango de fechas: de {final_dataframes[0]['timestamp'].min()} a {final_dataframes[0]['timestamp'].max()}")


# --- Punto de Entrada del Script ---
if __name__ == '__main__':
    main()
# --- Fin del Script ---
