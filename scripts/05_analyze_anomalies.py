#!/usr/bin/env python3
"""
05_analyze_anomalies.py

CMMSE 2025: Stage 5 - Comprehensive Anomaly Analysis
Analyzes individual anomaly records from Stage 4 to generate insights and visualizations.
"""

import pandas as pd
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class AnomalyAnalyzer:
    """Tool for comprehensive analysis of anomaly detection results."""
    def __init__(self, anomalies_df: pd.DataFrame, output_dir: Path):
        self.anomalies_df = anomalies_df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._add_temporal_features()

    def _add_temporal_features(self):
        """Adds temporal features for analysis."""
        df = self.anomalies_df
        
        # Convert cell_id arrays to strings for hashability
        if df['cell_id'].dtype == 'object':
            # Convert numpy arrays to first element (assuming single cell IDs)
            df['cell_id'] = df['cell_id'].apply(lambda x: x[0] if hasattr(x, '__len__') and len(x) > 0 else x)
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()

    def run_analysis(self):
        """Runs all analysis and visualization steps."""
        if self.anomalies_df.empty:
            print("No anomalies to analyze.")
            return

        print("Generating summary statistics...")
        self.generate_summary_report()
        print("Creating visualizations...")
        self.create_visualizations()
        print("Analysis complete.")

    def generate_summary_report(self):
        """Generates and saves a text summary report."""
        summary_path = self.output_dir / "anomaly_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CMMSE 2025: Anomaly Analysis Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total Anomalies Detected: {len(self.anomalies_df):,}\n")
            f.write(f"Unique Cells with Anomalies: {self.anomalies_df['cell_id'].nunique():,}\n")
            f.write(f"Date Range: {self.anomalies_df['timestamp'].min().date()} to {self.anomalies_df['timestamp'].max().date()}\n\n")
            
            f.write("Severity Score Statistics:\n")
            f.write(str(self.anomalies_df['severity_score'].describe().round(2)) + "\n\n")

            f.write("Top 5 Cells with Most Anomalies:\n")
            f.write(str(self.anomalies_df['cell_id'].value_counts().head(5)) + "\n\n")
            
            f.write("Anomalies by Day of Week:\n")
            f.write(str(self.anomalies_df['day_of_week'].value_counts()) + "\n\n")
            
            f.write("Anomalies by Hour of Day:\n")
            f.write(str(self.anomalies_df.groupby('hour').size()) + "\n")
        print(f"Summary report saved to {summary_path}")

    def create_visualizations(self):
        """Creates and saves visualization plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Severity Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.anomalies_df['severity_score'], bins=50, kde=True)
        plt.title('Distribution of Anomaly Severity Scores')
        plt.xlabel('Severity Score (σ)')
        plt.ylabel('Frequency')
        plt.savefig(self.output_dir / 'severity_distribution.png', dpi=300)
        plt.close()

        # Anomalies by Hour
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='hour',
            data=self.anomalies_df,
            hue='hour',          # evita el warning de Seaborn ≥ 0.14
            palette='viridis',
            legend=False
        )        
        plt.title('Number of Anomalies by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Anomaly Count')
        plt.savefig(self.output_dir / 'anomalies_by_hour.png', dpi=300)
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 5 - Anomaly Analysis")
    parser.add_argument("anomalies_path", type=Path, help="Path to individual anomalies file from Stage 4")
    parser.add_argument("--output_dir", type=Path, default="reports/", help="Output directory for analysis")
    parser.add_argument("--preview", action="store_true", help="Show a preview of the loaded data.")
    args = parser.parse_args()

    print("="*60); print("Stage 5: Comprehensive Anomaly Analysis"); print("="*60)
    start_time = time.perf_counter()

    anomalies_df = pd.read_parquet(args.anomalies_path)
    print(f"Loaded {len(anomalies_df):,} individual anomaly records.")

    if args.preview:
        print("\n--- DATA PREVIEW ---")
        print(f"Shape: {anomalies_df.shape}"); print("Head(5):"); print(anomalies_df.head(5))
        print("\nInfo:"); anomalies_df.info()

    analyzer = AnomalyAnalyzer(anomalies_df, args.output_dir)
    analyzer.run_analysis()
    
    total_time = time.perf_counter() - start_time
    print("\n--- STAGE 5 PERFORMANCE SUMMARY ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("✅ Stage 5 completed successfully.")

if __name__ == "__main__":
    main()