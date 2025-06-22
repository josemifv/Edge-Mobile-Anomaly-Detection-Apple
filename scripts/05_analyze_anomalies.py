#!/usr/bin/env python3
"""
05_analyze_anomalies.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 5: Comprehensive Anomaly Analysis

Analyzes individual anomaly records from Stage 4 to generate insights, statistics,
and visualizations for research and operational purposes.

Usage:
    python scripts/05_analyze_anomalies.py <anomalies_path> [--output_dir <dir>]

Example:
    python scripts/05_analyze_anomalies.py data/processed/individual_anomalies.parquet --output_dir results/analysis/
"""

import pandas as pd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class AnomalyAnalyzer:
    """Comprehensive anomaly analysis tool for CMMSE 2025 research."""
    
    def __init__(self, anomalies_df: pd.DataFrame, output_dir: str = "results/analysis/"):
        """
        Initialize anomaly analyzer.
        
        Args:
            anomalies_df: DataFrame with individual anomaly records
            output_dir: Output directory for analysis results
        """
        self.anomalies_df = anomalies_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.anomalies_df['timestamp']):
            self.anomalies_df['timestamp'] = pd.to_datetime(self.anomalies_df['timestamp'])
        
        # Add temporal features for analysis
        self._add_temporal_features()
        
    def _add_temporal_features(self):
        """Add temporal features for time-based analysis."""
        self.anomalies_df['hour'] = self.anomalies_df['timestamp'].dt.hour
        self.anomalies_df['day_of_week'] = self.anomalies_df['timestamp'].dt.day_name()
        self.anomalies_df['date'] = self.anomalies_df['timestamp'].dt.date
        self.anomalies_df['week'] = self.anomalies_df['timestamp'].dt.isocalendar().week
        self.anomalies_df['month'] = self.anomalies_df['timestamp'].dt.month
        
    def generate_summary_statistics(self) -> dict:
        """Generate comprehensive summary statistics."""
        print("Generating summary statistics...")
        
        total_anomalies = len(self.anomalies_df)
        unique_cells = self.anomalies_df['cell_id'].nunique()
        date_range = (self.anomalies_df['timestamp'].min(), self.anomalies_df['timestamp'].max())
        
        # Severity analysis
        severity_stats = self.anomalies_df['severity_score'].describe()
        
        # Temporal patterns
        hourly_dist = self.anomalies_df['hour'].value_counts().sort_index()
        daily_dist = self.anomalies_df['day_of_week'].value_counts()
        
        # Cell-level analysis
        cell_counts = self.anomalies_df['cell_id'].value_counts()
        
        # Traffic pattern analysis
        traffic_stats = self.anomalies_df[['sms_total', 'calls_total', 'internet_traffic']].describe()
        
        summary = {
            'overview': {
                'total_anomalies': total_anomalies,
                'unique_cells': unique_cells,
                'date_range': date_range,
                'avg_anomalies_per_cell': total_anomalies / unique_cells if unique_cells > 0 else 0
            },
            'severity': severity_stats.to_dict(),
            'temporal': {
                'peak_hour': hourly_dist.idxmax(),
                'peak_day': daily_dist.idxmax(),
                'hourly_distribution': hourly_dist.to_dict(),
                'daily_distribution': daily_dist.to_dict()
            },
            'cells': {
                'most_anomalous_cell': cell_counts.idxmax() if len(cell_counts) > 0 else None,
                'max_anomalies_per_cell': cell_counts.max() if len(cell_counts) > 0 else 0,
                'cells_with_anomalies': len(cell_counts),
                'top_10_cells': cell_counts.head(10).to_dict()
            },
            'traffic_patterns': traffic_stats.to_dict()
        }
        
        return summary
    
    def identify_severe_anomalies(self, top_n: int = 20) -> pd.DataFrame:
        """Identify and analyze the most severe anomalies."""
        print(f"Identifying top {top_n} most severe anomalies...")
        
        top_anomalies = self.anomalies_df.nlargest(top_n, 'severity_score').copy()
        
        # Add additional analysis columns
        top_anomalies['excess_factor'] = top_anomalies['anomaly_score'] / top_anomalies['error_threshold']
        top_anomalies['timestamp_str'] = top_anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return top_anomalies
    
    def analyze_temporal_patterns(self) -> dict:
        """Analyze temporal patterns in anomalies."""
        print("Analyzing temporal patterns...")
        
        # Hourly analysis
        hourly_stats = self.anomalies_df.groupby('hour').agg({
            'severity_score': ['count', 'mean', 'max'],
            'anomaly_score': 'mean'
        }).round(3)
        
        # Daily analysis
        daily_stats = self.anomalies_df.groupby('day_of_week').agg({
            'severity_score': ['count', 'mean', 'max'],
            'anomaly_score': 'mean'
        }).round(3)
        
        # Weekly analysis
        weekly_stats = self.anomalies_df.groupby('week').agg({
            'severity_score': ['count', 'mean', 'max'],
            'anomaly_score': 'mean'
        }).round(3)
        
        return {
            'hourly': hourly_stats,
            'daily': daily_stats,
            'weekly': weekly_stats
        }
    
    def analyze_cell_patterns(self) -> dict:
        """Analyze anomaly patterns by cell."""
        print("Analyzing cell-level patterns...")
        
        cell_stats = self.anomalies_df.groupby('cell_id').agg({
            'severity_score': ['count', 'mean', 'max', 'std'],
            'anomaly_score': 'mean',
            'sms_total': 'mean',
            'calls_total': 'mean',
            'internet_traffic': 'mean'
        }).round(3)
        
        # Flatten column names
        cell_stats.columns = ['_'.join(col).strip() for col in cell_stats.columns]
        cell_stats = cell_stats.reset_index()
        
        # Sort by anomaly count
        cell_stats = cell_stats.sort_values('severity_score_count', ascending=False)
        
        return cell_stats
    
    def create_visualizations(self):
        """Create comprehensive visualization plots."""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Overview Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Analysis Dashboard - CMMSE 2025', fontsize=16, fontweight='bold')
        
        # Severity distribution
        axes[0, 0].hist(self.anomalies_df['severity_score'], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Severity Score (σ)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Severity Score Distribution')
        axes[0, 0].axvline(self.anomalies_df['severity_score'].mean(), color='blue', linestyle='--', 
                          label=f'Mean: {self.anomalies_df["severity_score"].mean():.2f}σ')
        axes[0, 0].legend()
        
        # Hourly distribution
        hourly_counts = self.anomalies_df['hour'].value_counts().sort_index()
        axes[0, 1].plot(hourly_counts.index, hourly_counts.values, marker='o', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Anomalies')
        axes[0, 1].set_title('Anomalies by Hour of Day')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(range(0, 24, 2))
        
        # Daily distribution
        daily_counts = self.anomalies_df['day_of_week'].value_counts()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = daily_counts.reindex(day_order)
        axes[0, 2].bar(range(len(daily_counts)), daily_counts.values, color='orange', alpha=0.7)
        axes[0, 2].set_xlabel('Day of Week')
        axes[0, 2].set_ylabel('Number of Anomalies')
        axes[0, 2].set_title('Anomalies by Day of Week')
        axes[0, 2].set_xticks(range(len(daily_counts)))
        axes[0, 2].set_xticklabels([day[:3] for day in day_order], rotation=45)
        
        # Top cells
        top_cells = self.anomalies_df['cell_id'].value_counts().head(10)
        axes[1, 0].bar(range(len(top_cells)), top_cells.values, color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Cell Rank')
        axes[1, 0].set_ylabel('Number of Anomalies')
        axes[1, 0].set_title('Top 10 Cells by Anomaly Count')
        axes[1, 0].set_xticks(range(len(top_cells)))
        axes[1, 0].set_xticklabels([f'#{i+1}' for i in range(len(top_cells))])
        
        # Traffic correlation with severity
        axes[1, 1].scatter(self.anomalies_df['calls_total'], self.anomalies_df['severity_score'], 
                          alpha=0.6, s=20, color='blue')
        axes[1, 1].set_xlabel('Calls Total')
        axes[1, 1].set_ylabel('Severity Score (σ)')
        axes[1, 1].set_title('Calls vs Severity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Feature correlation heatmap
        feature_cols = ['sms_total', 'calls_total', 'internet_traffic', 'severity_score', 'anomaly_score']
        corr_matrix = self.anomalies_df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, ax=axes[1, 2], cbar_kws={'shrink': 0.8})
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Detailed Traffic Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Traffic Features vs Severity Analysis', fontsize=16, fontweight='bold')
        
        features = ['sms_total', 'calls_total', 'internet_traffic']
        feature_labels = ['SMS Total', 'Calls Total', 'Internet Traffic']
        
        for i, (feature, label) in enumerate(zip(features, feature_labels)):
            axes[i].scatter(self.anomalies_df[feature], self.anomalies_df['severity_score'], 
                           alpha=0.6, s=20, color=sns.color_palette()[i])
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('Severity Score (σ)')
            axes[i].set_title(f'Severity vs {label}')
            axes[i].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = self.anomalies_df[feature].corr(self.anomalies_df['severity_score'])
            axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'traffic_severity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def export_reports(self, summary_stats: dict, severe_anomalies: pd.DataFrame, 
                      temporal_patterns: dict, cell_patterns: pd.DataFrame):
        """Export comprehensive analysis reports."""
        print("Exporting analysis reports...")
        
        # 1. Summary report (text)
        with open(self.output_dir / 'anomaly_analysis_summary.txt', 'w') as f:
            f.write("CMMSE 2025: Anomaly Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERVIEW:\n")
            f.write(f"  Total anomalies: {summary_stats['overview']['total_anomalies']:,}\n")
            f.write(f"  Unique cells: {summary_stats['overview']['unique_cells']:,}\n")
            f.write(f"  Date range: {summary_stats['overview']['date_range'][0]} to {summary_stats['overview']['date_range'][1]}\n")
            f.write(f"  Avg anomalies per cell: {summary_stats['overview']['avg_anomalies_per_cell']:.1f}\n\n")
            
            f.write("SEVERITY STATISTICS:\n")
            for stat, value in summary_stats['severity'].items():
                f.write(f"  {stat.capitalize()}: {value:.2f}σ\n")
            f.write("\n")
            
            f.write("TEMPORAL PATTERNS:\n")
            f.write(f"  Peak hour: {summary_stats['temporal']['peak_hour']}:00\n")
            f.write(f"  Peak day: {summary_stats['temporal']['peak_day']}\n\n")
            
            f.write("TOP ANOMALOUS CELLS:\n")
            for i, (cell, count) in enumerate(list(summary_stats['cells']['top_10_cells'].items())[:5], 1):
                f.write(f"  {i}. Cell {cell}: {count} anomalies\n")
        
        # 2. Severe anomalies (CSV)
        severe_anomalies.to_csv(self.output_dir / 'severe_anomalies_detailed.csv', index=False)
        
        # 3. Cell-level analysis (CSV)
        cell_patterns.to_csv(self.output_dir / 'cell_anomaly_patterns.csv', index=False)
        
        # 4. Temporal analysis (CSV)
        temporal_patterns['hourly'].to_csv(self.output_dir / 'hourly_anomaly_patterns.csv')
        temporal_patterns['daily'].to_csv(self.output_dir / 'daily_anomaly_patterns.csv')
        
        print(f"Reports exported to {self.output_dir}")
    
    def run_complete_analysis(self, top_n_severe: int = 20):
        """Run complete anomaly analysis pipeline."""
        print("Starting complete anomaly analysis...")
        
        if len(self.anomalies_df) == 0:
            print("Warning: No anomalies to analyze")
            return
        
        # Generate all analyses
        summary_stats = self.generate_summary_statistics()
        severe_anomalies = self.identify_severe_anomalies(top_n_severe)
        temporal_patterns = self.analyze_temporal_patterns()
        cell_patterns = self.analyze_cell_patterns()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export reports
        self.export_reports(summary_stats, severe_anomalies, temporal_patterns, cell_patterns)
        
        # Print key findings
        self._print_key_findings(summary_stats, severe_anomalies)
        
        return {
            'summary': summary_stats,
            'severe_anomalies': severe_anomalies,
            'temporal_patterns': temporal_patterns,
            'cell_patterns': cell_patterns
        }
    
    def _print_key_findings(self, summary_stats: dict, severe_anomalies: pd.DataFrame):
        """Print key findings to console."""
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)
        
        print(f"📊 OVERVIEW:")
        print(f"   • {summary_stats['overview']['total_anomalies']:,} total anomalies detected")
        print(f"   • {summary_stats['overview']['unique_cells']:,} cells affected")
        print(f"   • {summary_stats['overview']['avg_anomalies_per_cell']:.1f} average anomalies per cell")
        
        print(f"\n🔥 SEVERITY:")
        print(f"   • Most severe: {summary_stats['severity']['max']:.1f}σ above normal")
        print(f"   • Average severity: {summary_stats['severity']['mean']:.1f}σ")
        print(f"   • 75th percentile: {summary_stats['severity']['75%']:.1f}σ")
        
        print(f"\n⏰ TEMPORAL PATTERNS:")
        print(f"   • Peak hour: {summary_stats['temporal']['peak_hour']}:00")
        print(f"   • Peak day: {summary_stats['temporal']['peak_day']}")
        
        print(f"\n📡 TOP ANOMALOUS CELLS:")
        for i, (cell, count) in enumerate(list(summary_stats['cells']['top_10_cells'].items())[:3], 1):
            print(f"   {i}. Cell {cell}: {count} anomalies")
        
        if len(severe_anomalies) > 0:
            print(f"\n⚠️  MOST SEVERE ANOMALY:")
            top_anomaly = severe_anomalies.iloc[0]
            print(f"   • Cell {top_anomaly['cell_id']} on {top_anomaly['timestamp_str']}")
            print(f"   • Severity: {top_anomaly['severity_score']:.1f}σ")
            print(f"   • Traffic: SMS={top_anomaly['sms_total']:.1f}, Calls={top_anomaly['calls_total']:.1f}, Internet={top_anomaly['internet_traffic']:.1f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Stage 5 - Anomaly Analysis")
    parser.add_argument("anomalies_path", help="Path to individual anomalies file from Stage 4")
    parser.add_argument("--output_dir", default="results/analysis/",
                       help="Output directory for analysis results")
    parser.add_argument("--top_n_severe", type=int, default=20,
                       help="Number of top severe anomalies to analyze")
    parser.add_argument("--preview", action="store_true", help="Show analysis preview")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CMMSE 2025: Mobile Network Anomaly Detection")
    print("Stage 5: Comprehensive Anomaly Analysis")
    print("="*60)
    
    start_time = time.perf_counter()
    
    try:
        # Load anomaly data
        print("Loading anomaly data...")
        
        if args.anomalies_path.endswith('.parquet'):
            anomalies_df = pd.read_parquet(args.anomalies_path)
        else:
            anomalies_df = pd.read_csv(args.anomalies_path)
        
        print(f"  Loaded {len(anomalies_df):,} individual anomaly records")
        
        if len(anomalies_df) == 0:
            print("No anomalies to analyze. Exiting.")
            return
        
        # Initialize analyzer
        analyzer = AnomalyAnalyzer(anomalies_df, args.output_dir)
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(args.top_n_severe)
        
        # Show preview if requested
        if args.preview:
            print("\n" + "="*40)
            print("ANALYSIS PREVIEW")
            print("="*40)
            anomalies_df.info()
            print("\nSeverity score statistics:")
            print(anomalies_df['severity_score'].describe())
            print("\nTop 5 cells with most anomalies:")
            print(anomalies_df['cell_id'].value_counts().head())
        
        # Performance summary
        total_time = time.perf_counter() - start_time
        
        print("\n" + "="*60)
        print("STAGE 5 SUMMARY")
        print("="*60)
        print(f"Anomalies analyzed: {len(anomalies_df):,}")
        print(f"Analysis time: {total_time:.2f} seconds")
        print(f"Output directory: {args.output_dir}")
        print(f"Reports generated: 6 files")
        print(f"Visualizations: 2 plots")
        print("Stage 5 completed successfully!")
        
    except Exception as e:
        print(f"Stage 5 failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
