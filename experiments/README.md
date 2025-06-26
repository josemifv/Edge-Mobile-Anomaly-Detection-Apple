# Experimentos de Benchmark

Este directorio contiene los resultados de los experimentos de benchmark realizados en diferentes plataformas y versiones de Python.

## Estructura del Directorio

```
experiments/
├── M2 Pro/
│   └── Python 3.13.2/
│       └── benchmark_20250625_193055/
│           ├── benchmark_results.json
│           └── results_run_[1-10]/
│               ├── pipeline_execution.log
│               ├── pipeline_status.json
│               └── reports/
│                   ├── anomaly_analysis_summary.txt
│                   ├── anomalies_by_hour.png
│                   └── severity_distribution.png
└── M4 Pro/
    └── Python 3.13.5/
        └── benchmark_20250625_210121/
            └── results_run_[1-10]/
                ├── pipeline_execution.log
                ├── pipeline_status.json
                └── reports/
                    ├── anomaly_analysis_summary.txt
                    ├── anomalies_by_hour.png
                    └── severity_distribution.png
```

## Archivos Incluidos

- **pipeline_execution.log**: Logs detallados de la ejecución del pipeline
- **pipeline_status.json**: Estado y métricas de rendimiento del pipeline
- **benchmark_results.json**: Resultados consolidados del benchmark (solo M2 Pro)
- **reports/**: Visualizaciones y análisis de anomalías
  - **anomaly_analysis_summary.txt**: Resumen textual del análisis
  - **anomalies_by_hour.png**: Distribución temporal de anomalías
  - **severity_distribution.png**: Distribución por severidad

## Archivos Excluidos

Los siguientes tipos de archivos están excluidos del repositorio por tamaño:
- **\*.parquet**: Archivos de datos binarios (muy pesados)
- **03_reference_weeks.parquet**: Datos de referencia semanal
- **04_individual_anomalies.parquet**: Datos individuales de anomalías

## Plataformas Testadas

- **M2 Pro**: 10 runs con Python 3.13.2
- **M4 Pro**: 10 runs con Python 3.13.5

## Fecha de Experimentos

- **M2 Pro**: 25 de junio de 2025, 19:30:55
- **M4 Pro**: 25 de junio de 2025, 21:01:21

## Acceso a Datos Completos

Para acceder a los archivos `.parquet` completos, contactar al equipo de investigación o ejecutar los benchmarks localmente.
