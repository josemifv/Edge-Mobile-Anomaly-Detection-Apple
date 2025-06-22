Below is a high‐level tour of this refactored “Edge‑Mobile Anomaly Detection” codebase.  I’ve organized the explanation into:

1. Project overview (from the README)
2. Top‑level directory layout (what’s new vs. what’s in backup/)
3. Data folders
4. The new “scripts/” pipeline and auxiliary tools
5. Results and artefacts
6. Legacy (“v1”) code and reports in backup/

----------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Project overview

The README gives a concise summary of goals and features:

    # Edge-Mobile Anomaly Detection (CMMSE 2025)

    ## Project Overview
    Enhanced mobile network anomaly detection pipeline optimized for Apple Silicon, featuring:
    - 4-stage modular pipeline for telecommunications anomaly detection
    - OSP (Orthogonal Subspace Projection) based anomaly detection
    - Hardware-accelerated processing with Apple Silicon optimization
    - Academic research focus for CMMSE 2025 conference

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

----------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Top‑level layout

    ├── backup/              ← old v1 code & reports (see §6 below)
    ├── data/                ← raw inputs & processed intermediates
    ├── results/             ← full pipeline outputs (anomalies, plots, maps,…)
    ├── scripts/             ← refactored pipeline stages & analysis tools
    ├── requirements.txt
    ├── README.md
    └── (virtual‑env files, .gitignore, LICENSE, etc.)

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

----------------------------------------------------------------------------------------------------------------------------------------------------

## 3. Data folders

Under data/ you’ll find two subfolders:

┌─────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────┐
│ Folder          │ Contents                                                                                                                         
    │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────┤
│ data/raw/       │ original telecom text files (e.g. sms‑call‑internet‑mi‑YYYY‑MM‑DD.txt), plus geojson grid
    │
├─────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────┤
│ data/processed/ │ Parquet outputs of each pipeline stage: <br/>• ingested_data.parquet <br/>• preprocessed_data.parquet <br/>•
reference_weeks.parquet │
└─────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────┘

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

----------------------------------------------------------------------------------------------------------------------------------------------------

## 4. Refactored “scripts/” pipeline

The core pipeline remains a four‑stage modular flow, now accompanied by additional analysis/visualization scripts:

### 4.1. Four pipeline stages

    ## Pipeline Stages

    The pipeline consists of 4 sequential stages:

    ### Stage 1: Data Ingestion (`01_data_ingestion.py`)
    - Loads raw telecommunications data from .txt files
    - Parallel processing for multiple files
    - Timestamp conversion and initial cleaning
    - Apple Silicon optimized

    ### Stage 2: Data Preprocessing (`02_data_preprocessing.py`)
    - Aggregates data by cell_id and timestamp
    - Merges directional columns (SMS/calls in+out → totals)
    - Data quality validation
    - Outputs clean, consolidated dataset

    ### Stage 3: Reference Week Selection (`03_week_selection.py`)
    - Applies Median Absolute Deviation (MAD) analysis
    - Selects "normal" reference weeks per cell
    - Configurable selection criteria
    - Provides training data for anomaly detection

    ### Stage 4: OSP Anomaly Detection (`04_anomaly_detection_osp.py`)
    - Implements Orthogonal Subspace Projection using SVD
    - Per-cell anomaly detection models
    - Uses reference weeks for normal behavior modeling
    - Parallel processing for scalability

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

### 4.2. Running the pipeline

You can invoke individual stages or run the full 4‑stage pipeline via a wrapper:

    # Individual stage example:
    python scripts/01_data_ingestion.py data/raw/ --output_path data/processed/ingested_data.parquet

    # Run the complete pipeline end‑to‑end:
    python scripts/run_pipeline.py data/raw/ --output_dir results/ --n_components 5 --anomaly_threshold 2.5 --preview

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

----------------------------------------------------------------------------------------------------------------------------------------------------

## 5. Analysis, benchmarks & results

In addition to the four core stages, the refactored branch adds a suite of scripts under scripts/ for:

┌───────────────────────────────────────┬───────────────────────────────────────────────────┐
│ Script                                │ Purpose                                           │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 05_analyze_anomalies.py               │ Summarize and explore anomaly results             │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 06_generate_anomaly_map.py            │ Create geo‑maps of detected anomalies             │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 07_plot_extreme_cell_timeline.py      │ Plot time‑series for cells with extreme anomalies │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ analyze_severe_anomalies.py           │ Quick summary/top‑N severe anomalies              │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ benchmark_micro_test.py               │ Micro‑benchmarks                                  │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ benchmark_parameter_sweep.py          │ Parameter sweep for threshold testing             │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ run_pipeline.py                       │ High‑level wrapper to run all stages              │
├───────────────────────────────────────┼───────────────────────────────────────────────────┤
│ … and a few more benchmarking scripts │                                                   │
└───────────────────────────────────────┴───────────────────────────────────────────────────┘

This richer toolkit supports deeper post‑processing and visualization beyond the original 4‑stage flow.

----------------------------------------------------------------------------------------------------------------------------------------------------

## 6. Legacy code & reports (backup/)

All of the pre‑refactor (“v1”) scripts, performance reports, and documentation have been moved into the backup/edge_mobile_v1_20250620/ folder so you
 can still reference them.  Inside that backup you’ll find:

    backup/edge_mobile_v1_20250620/
    ├── scripts/           ← v1 versions of ingestion, preprocessing, OSP, plus benchmarks
    └── reports/
        ├── documentation/ ← stage‑by‑stage tuning docs (e.g. APPLE_SILICON_OPTIMIZATION.md, etc.)
        └── benchmarks/    ← raw performance logs and summary reports

----------------------------------------------------------------------------------------------------------------------------------------------------

### In a nutshell

    * **Refactored “scripts/”**: streamlined, one variant per pipeline stage + new analysis/plotting tools
    * **`data/` → raw & processed**: clear separation of inputs vs. intermediates
    * **`results/` → outputs**: all anomaly outputs, figures, maps, detailed summaries
    * **`backup/`** preserves the old codebase (v1), performance tuning docs, and raw benchmark logs
    * **README.md** provides install, config, usage, and performance tables

----------------------------------------------------------------------------------------------------------------------------------------------------

Below is a focused, functionality‑oriented review of the refactored “Edge‑Mobile Anomaly Detection” scripts.  I’ve grouped feedback into (1)
general/cross‑cutting observations and (2) script‑by‑script issues.  Wherever possible I’ve cited the exact lines in the code.

----------------------------------------------------------------------------------------------------------------------------------------------------

## 1. General / Cross‑cutting observations

### 1.1 Warnings‑silencing hides potentially useful diagnostics

Many of the analysis scripts begin by blanket‑ignoring warnings:

    import warnings
    warnings.filterwarnings('ignore')

scripts/04_anomaly_detection_osp.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)scripts/04_anomaly_detection_individual.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_individual.py)scripts/05_analyze_anomalies.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/05_analyze_anomalies.py)scripts/06_generate_anomaly_map.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/06_generate_anomaly_map.py)scripts/07_plot_extreme_cell_timeline.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/07_plot_extreme_cell_timeline.py)

    Suggestion: Remove or scope these filters so you don’t inadvertently suppress real issues (e.g. deprecated‑API warnings or parsing errors).

----------------------------------------------------------------------------------------------------------------------------------------------------

### 1.2 Preview flags not actually wired up

Several stages define a --preview CLI flag but never use it:

    parser.add_argument("--preview", action="store_true", help="Show analysis preview")

scripts/05_analyze_anomalies.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/05_analyze_anomalies.py)

Yet in main() nothing checks args.preview, so --preview is a no‑op.

    Suggestion: Either implement preview logic (e.g. show summary to console) or remove the flag to avoid confusion.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 1.3 Inconsistent use of timing functions

Some scripts use time.perf_counter(), others time.time().  Example:

    start_time = time.perf_counter()
    …
    processing_time = time.time() - start_time

scripts/04_anomaly_detection_osp.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)scripts/04_anomaly_detection_osp.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)

    Suggestion: Standardize on time.perf_counter() for elapsed timing.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 1.4 Unused / stale dependencies in requirements.txt

The requirements.txt includes packages that none of the refactored scripts import (e.g. torch, psutil):

    torch>=2.2.0
    psutil>=7.0.0

requirements.txt (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/requirements.txt)requirements.txt 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/requirements.txt)

    Suggestion: Prune out unused dependencies (or re‑introduce Torch usage if planned).

----------------------------------------------------------------------------------------------------------------------------------------------------

### 1.5 README vs. actual pipeline mismatch

The README describes a 4‑stage pipeline:

    The pipeline consists of 4 sequential stages:
    ### Stage 4: OSP Anomaly Detection (`04_anomaly_detection_osp.py`)

README.md (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/README.md)

But run_pipeline.py actually invokes five stages (ends with Stage 5 anomaly analysis):

    # Stage 5: Comprehensive Anomaly Analysis
    stage_times['stage5'] = run_stage(5, "05_analyze_anomalies.py", stage5_args, "Comprehensive Anomaly Analysis")

scripts/run_pipeline.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/run_pipeline.py)

    Suggestion: Update README (and doc‑strings) to reflect the extra analysis stage.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 1.6 In‑console previews print an extra “None”

All scripts that do a .info() preview wrap it in a print(), which prints the info and then “None”:

    print(df_processed.info())

scripts/02_data_preprocessing.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/02_data_preprocessing.py)

    print(reference_weeks.info())

scripts/03_week_selection.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/03_week_selection.py)

    print(results_df.info())

scripts/04_anomaly_detection_osp.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)

    print(anomalies_df.info())

scripts/04_anomaly_detection_individual.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_individual.py)

    Suggestion: Call .info() on its own line (no print) so you don’t see the spurious None.

----------------------------------------------------------------------------------------------------------------------------------------------------

## 2. Script‑by‑script feedback

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.1 01_data_ingestion.py

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES,
        …
    )

scripts/01_data_ingestion.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/01_data_ingestion.py)

    * **Regex sep → Python engine:**  Using `sep=r'\s+'` forces the slower Python engine.  You could switch to `delim_whitespace=True, engine='c'`
for better performance.

    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
    …
    df = df.drop(columns=['timestamp_ms', 'country_code'])

scripts/01_data_ingestion.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/01_data_ingestion.py)

    * **Silent coercion to NaT:**  `errors='coerce'` may introduce NaT rows.  Consider warning or dropping rows where `timestamp` is NaT.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.2 02_data_preprocessing.py

    if args.preview:
        print(df_processed.info())
        print(df_processed.head())
        …

scripts/02_data_preprocessing.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/02_data_preprocessing.py)

    * **Preview bug:** As noted above, `print(df_processed.info())` prints extra `None`.
    * **Compression metric mislabeled:**  The summary prints “Compression: X%” based on row‑count reduction, not file‑size compression.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.3 03_week_selection.py

    for _, week_row in selected_weeks.iterrows():
        reference_weeks.append({
            …
            'selection_rank': len(reference_weeks) % num_weeks + 1
        })

scripts/03_week_selection.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/03_week_selection.py)

    * **Non‑obvious rank logic:**  Computing `selection_rank` via the global length of `reference_weeks` is clever but hard to read.  A clearer
approach is to enumerate the local `selected_weeks`:

          for rank, (_, week_row) in enumerate(selected_weeks.iterrows(), start=1):
              …
              'selection_rank': rank

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.4 04_anomaly_detection_osp.py

    warnings.filterwarnings('ignore')
    …
    with multiprocessing.Pool(max_workers) as pool:
        results = pool.map(process_single_cell, tasks)

scripts/04_anomaly_detection_osp.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)scripts/04_anomaly_detection_osp.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_osp.py)

    * **Pool argument mismatch:**  It should be `Pool(processes=max_workers)` rather than `Pool(max_workers)` (the latter passes `processes` as the
first positional argument, but is less explicit).  (It works, but for clarity use `Pool(processes=…)`.)
    * **Inconsistent timing functions** (see §1.3).

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.5 04_anomaly_detection_individual.py

    parser.add_argument("--sample_cells", type=int, help="Process only N cells for testing")
    …
    if args.sample_cells:
        unique_cells = unique_cells[:args.sample_cells]

scripts/04_anomaly_detection_individual.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_individ
ual.py)scripts/04_anomaly_detection_individual.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/04_anomaly_detection_individual.py)

    * **Good functionality** for sampling, but note that `run_pipeline.py` mistakenly passes `--sample_cells` to the *OSP* script instead of the
*individual* script.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.6 run_pipeline.py

    # Stage 4: Individual OSP Anomaly Detection
    stage_times['stage4'] = run_stage(
        4, "04_anomaly_detection_individual.py", stage4_args, "Individual OSP Anomaly Detection"
    )
    …
    # Stage 5: Comprehensive Anomaly Analysis
    stage_times['stage5'] = run_stage(
        5, "05_analyze_anomalies.py", stage5_args, "Comprehensive Anomaly Analysis"
    )

scripts/run_pipeline.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/run_pipeline.py)scripts/run_pipeline.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/run_pipeline.py)

    * **Pipeline/README mismatch** (see §1.5).
    * **Parameter propagation:**  `--sample_cells` and `--preview` flags in `run_pipeline.py` must align with the target scripts’ CLI (e.g.
sample_cells applies to the individual script, but was attached to the osp script in an earlier version).

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.7 05_analyze_anomalies.py

    parser.add_argument("--preview", action="store_true", help="Show analysis preview")
    …
    # (no subsequent use of args.preview)

scripts/05_analyze_anomalies.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/05_analyze_anomalies.py)

    * **Unused preview flag** (see §1.2).

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.8 06_generate_anomaly_map.py

    all_cell_ids = set(range(1, 10001))
    analyzed_cell_ids = set(anomaly_df['cell_id'])
    missing_cell_ids = all_cell_ids - analyzed_cell_ids

scripts/06_generate_anomaly_map.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/06_generate_anomaly_map.py)

    * **Hard‑coded 10 000 cells:**  If your grid changes size or cell IDs are non‑contiguous, this will break or mis‑classify.  Better to derive the
full cell list from the GeoJSON:

          all_cell_ids = {f['properties']['cellId'] for f in self.geojson_data['features']}

    import folium
    from folium import plugins
    import plotly.express as px
    import plotly.graph_objects as go

scripts/06_generate_anomaly_map.py (/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/06_generate_anomaly_map.py)

    * **Missing dependencies:**  Neither `folium` nor `plotly` are listed in `requirements.txt`.  Attempting to run will error out.

----------------------------------------------------------------------------------------------------------------------------------------------------

### 2.9 07_plot_extreme_cell_timeline.py

    warnings.filterwarnings('ignore')

scripts/07_plot_extreme_cell_timeline.py 
(/Users/jmfranco/Código/Doctorado/Edge-Mobile-Anomaly-Detection-Apple/scripts/07_plot_extreme_cell_timeline.py)

    * **Again silencing warnings** (see §1.1).
    * Generally this script is functional, but you may want to parameterize the quantile used for threshold lines (currently hard‑coded to the 95 th
percentile).

----------------------------------------------------------------------------------------------------------------------------------------------------

## 3. Summary of recommended improvements

    * **Remove or scope `warnings.filterwarnings('ignore')`.**
    * **Fix all “print(.info())” calls** so they don’t print `None`.
    * **Wire up or delete unused `--preview` flags**.
    * **Prune stale dependencies** from `requirements.txt` (e.g. `torch`, `psutil`) and add missing ones (`folium`, `plotly`).
    * **Un‑hardcode magic numbers**, especially the 1–10 000 range in map generation.
    * **Align README docs with the actual 5‑stage pipeline**.
    * **Standardize timing calls** (use `time.perf_counter()` everywhere).
    * **Document or simplify non‑obvious logic** (e.g. the `selection_rank` trick in week selection).
    * **Consider performance tweaks** (e.g. switch to `delim_whitespace=True, engine='c'` in ingestion).

----------------------------------------------------------------------------------------------------------------------------------------------------

Overall, the refactored pipeline is well‑structured and covers all academic use‑cases.  The above tweaks will sharpen its correctness and make the
code easier to maintain and use.  Let me know if you’d like to dive deeper into any particular fix or submit patches for these items!

╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
  ctrl+c to exit | "/" to see commands | enter to send — 98% context left

