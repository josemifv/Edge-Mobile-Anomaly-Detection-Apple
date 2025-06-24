#!/usr/bin/env python3
"""
analyze_severe_anomalies.py

CMMSE 2025: Severe Anomaly Analysis Tool
Analyzes individual anomaly data to identify the most significant anomalies

Usage:
    python scripts/analyze_severe_anomalies.py <individual_anomalies_path> [--top_n 10]

Example:
    python scripts/analyze_severe_anomalies.py results/individual_anomalies.parquet --top_n 20
"""

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


def analyze_severe_anomalies(individual_df: pd.DataFrame, top_n: int = 10) -> None:
    """Analyze and display the most severe anomalies."""
    
    if len(individual_df) == 0:
        print("No individual anomalies found in the dataset.")
        return
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(individual_df['timestamp']):
        individual_df['timestamp'] = pd.to_datetime(individual_df['timestamp'])
    
    # Sort by severity score (descending)
    top_anomalies = individual_df.nlargest(top_n, 'severity_score')
    
    print("="*80)
    print(f"TOP {top_n} MOST SEVERE ANOMALIES")
    print("="*80)
    
    for i, (_, anomaly) in enumerate(top_anomalies.iterrows(), 1):
        timestamp_str = anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{i:2d}. Cell {anomaly['cell_id']:4d} | {timestamp_str} | Severity: {anomaly['severity_score']:6.2f}σ")
        print(f"    Anomaly Score: {anomaly['anomaly_score']:.8f}")
        print(f"    Error Threshold: {anomaly['error_threshold']:.8f}")
        print(f"    Excess Factor: {anomaly['anomaly_score']/anomaly['error_threshold']:.2f}x above threshold")
        print(f"    Traffic Values:")
        print(f"      SMS Total: {anomaly['sms_total']:10.1f}")
        print(f"      Calls Total: {anomaly['calls_total']:10.1f}")
        print(f"      Internet Traffic: {anomaly['internet_traffic']:10.1f}")
    
    return top_anomalies


def generate_anomaly_statistics(individual_df: pd.DataFrame) -> None:
    """Generate comprehensive statistics about anomalies."""
    
    print("\n" + "="*80)
    print("ANOMALY ANALYSIS STATISTICS")
    print("="*80)
    
    # Basic statistics
    print(f"Total anomalies: {len(individual_df):,}")
    print(f"Unique cells with anomalies: {individual_df['cell_id'].nunique():,}")
    print(f"Time range: {individual_df['timestamp'].min()} to {individual_df['timestamp'].max()}")
    
    # Severity score statistics
    print(f"\nSeverity Score Statistics:")
    severity_stats = individual_df['severity_score'].describe()
    for stat, value in severity_stats.items():
        print(f"  {stat:>10s}: {value:8.2f}σ")
    
    # Anomaly score statistics
    print(f"\nAnomaly Score Statistics:")
    score_stats = individual_df['anomaly_score'].describe()
    for stat, value in score_stats.items():
        print(f"  {stat:>10s}: {value:10.6f}")
    
    # Top cells with most anomalies
    print(f"\nCells with Most Anomalies (Top 10):")
    cell_counts = individual_df['cell_id'].value_counts().head(10)
    for cell_id, count in cell_counts.items():
        cell_rate = (count / len(individual_df)) * 100
        print(f"  Cell {cell_id:4d}: {count:4d} anomalies ({cell_rate:5.1f}%)")
    
    # Traffic pattern analysis
    print(f"\nTraffic Pattern Analysis:")
    feature_stats = individual_df[['sms_total', 'calls_total', 'internet_traffic']].describe()
    print(feature_stats)
    
    # Temporal analysis
    individual_df['hour'] = individual_df['timestamp'].dt.hour
    individual_df['day_of_week'] = individual_df['timestamp'].dt.day_name()
    
    print(f"\nTemporal Patterns:")
    print(f"  Peak anomaly hour: {individual_df['hour'].mode().iloc[0]:02d}:00")
    print(f"  Peak anomaly day: {individual_df['day_of_week'].mode().iloc[0]}")
    
    hourly_counts = individual_df['hour'].value_counts().sort_index()
    daily_counts = individual_df['day_of_week'].value_counts()
    
    print(f"\nHourly Distribution (Top 5):")
    for hour in hourly_counts.head().index:
        count = hourly_counts[hour]
        percentage = (count / len(individual_df)) * 100
        print(f"  {hour:02d}:00 - {(hour+1):02d}:00: {count:4d} anomalies ({percentage:5.1f}%)")
    
    print(f"\nDaily Distribution:")
    for day in daily_counts.index:
        count = daily_counts[day]
        percentage = (count / len(individual_df)) * 100
        print(f"  {day:>9s}: {count:4d} anomalies ({percentage:5.1f}%)")


def plot_anomaly_distributions(individual_df: pd.DataFrame, output_dir: str = "results/figures/") -> None:
    """Generate visualization plots for anomaly analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Severity Score Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Anomaly Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Severity score histogram
    axes[0, 0].hist(individual_df['severity_score'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Severity Score (σ)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Severity Scores')
    axes[0, 0].axvline(individual_df['severity_score'].mean(), color='blue', linestyle='--', 
                      label=f'Mean: {individual_df["severity_score"].mean():.2f}σ')
    axes[0, 0].legend()
    
    # Top cells with anomalies
    top_cells = individual_df['cell_id'].value_counts().head(10)
    axes[0, 1].bar(range(len(top_cells)), top_cells.values, color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Cell Rank')
    axes[0, 1].set_ylabel('Number of Anomalies')
    axes[0, 1].set_title('Top 10 Cells by Anomaly Count')
    axes[0, 1].set_xticks(range(len(top_cells)))
    axes[0, 1].set_xticklabels([f'#{i+1}' for i in range(len(top_cells))])
    
    # Hourly distribution
    individual_df['hour'] = individual_df['timestamp'].dt.hour
    hourly_counts = individual_df['hour'].value_counts().sort_index()
    axes[1, 0].plot(hourly_counts.index, hourly_counts.values, marker='o', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Number of Anomalies')
    axes[1, 0].set_title('Anomalies by Hour of Day')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # Feature correlation
    feature_cols = ['sms_total', 'calls_total', 'internet_traffic', 'severity_score']
    corr_matrix = individual_df[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path / 'anomaly_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved dashboard plot: {output_path / 'anomaly_analysis_dashboard.png'}")
    
    # Figure 2: Severity vs Features
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Severity Score vs Traffic Features', fontsize=16, fontweight='bold')
    
    features = ['sms_total', 'calls_total', 'internet_traffic']
    feature_labels = ['SMS Total', 'Calls Total', 'Internet Traffic']
    
    for i, (feature, label) in enumerate(zip(features, feature_labels)):
        axes[i].scatter(individual_df[feature], individual_df['severity_score'], 
                       alpha=0.6, s=20, color=sns.color_palette()[i])
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Severity Score (σ)')
        axes[i].set_title(f'Severity vs {label}')
        axes[i].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = individual_df[feature].corr(individual_df['severity_score'])
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'severity_vs_features.png', dpi=300, bbox_inches='tight')
    print(f"Saved feature correlation plot: {output_path / 'severity_vs_features.png'}")
    
    plt.close('all')


def export_severe_anomalies(top_anomalies: pd.DataFrame, output_path: str = "results/severe_anomalies_report.csv") -> None:
    """Export the most severe anomalies to a detailed report."""
    
    # Create a comprehensive report
    report_df = top_anomalies.copy()
    
    # Add additional analysis columns
    report_df['timestamp_str'] = report_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    report_df['excess_factor'] = report_df['anomaly_score'] / report_df['error_threshold']
    report_df['hour'] = report_df['timestamp'].dt.hour
    report_df['day_of_week'] = report_df['timestamp'].dt.day_name()
    report_df['date'] = report_df['timestamp'].dt.date
    
    # Reorder columns for better readability
    column_order = [
        'cell_id', 'timestamp_str', 'date', 'hour', 'day_of_week',
        'severity_score', 'anomaly_score', 'error_threshold', 'excess_factor',
        'sms_total', 'calls_total', 'internet_traffic'
    ]
    
    report_df = report_df[column_order]
    
    # Save the report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    
    print(f"\nSaved detailed report: {output_path}")
    print(f"Report contains {len(report_df)} severe anomalies with complete analysis")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Severe Anomaly Analysis")
    parser.add_argument("individual_anomalies_path", help="Path to individual anomalies file")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top anomalies to analyze")
    parser.add_argument("--output_dir", default="results/", help="Output directory for reports and plots")
    parser.add_argument("--generate_plots", action="store_true", help="Generate visualization plots")
    parser.add_argument("--export_report", action="store_true", help="Export detailed report")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CMMSE 2025: Severe Anomaly Analysis")
    print("="*80)
    
    try:
        # Load individual anomalies data
        print(f"Loading individual anomalies from: {args.individual_anomalies_path}")
        
        if args.individual_anomalies_path.endswith('.parquet'):
            individual_df = pd.read_parquet(args.individual_anomalies_path)
        else:
            individual_df = pd.read_csv(args.individual_anomalies_path)
        
        print(f"Loaded {len(individual_df):,} individual anomaly records")
        
        if len(individual_df) == 0:
            print("No anomalies found in the dataset. Exiting.")
            return
        
        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(individual_df['timestamp']):
            individual_df['timestamp'] = pd.to_datetime(individual_df['timestamp'])
        
        # Analyze severe anomalies
        top_anomalies = analyze_severe_anomalies(individual_df, args.top_n)
        
        # Generate comprehensive statistics
        generate_anomaly_statistics(individual_df)
        
        # Generate plots if requested
        if args.generate_plots:
            plot_anomaly_distributions(individual_df, f"{args.output_dir}/figures/")
        
        # Export detailed report if requested
        if args.export_report:
            report_path = f"{args.output_dir}/severe_anomalies_top_{args.top_n}.csv"
            export_severe_anomalies(top_anomalies, report_path)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Analyzed {len(individual_df):,} anomalies")
        print(f"Identified top {args.top_n} most severe anomalies")
        
        if args.generate_plots:
            print(f"Generated visualization plots in: {args.output_dir}/figures/")
        
        if args.export_report:
            print(f"Exported detailed report: {args.output_dir}/severe_anomalies_top_{args.top_n}.csv")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
