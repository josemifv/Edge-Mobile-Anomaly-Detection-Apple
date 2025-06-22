#!/usr/bin/env python3
"""
07_plot_extreme_cell_timeline.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 7: Extreme Cell Timeline Visualization

Creates a time series plot showing datetime vs severity for the cell with 
the most extreme anomaly activity.

Usage:
    python scripts/07_plot_extreme_cell_timeline.py <individual_anomalies_file> [--cell_id <id>] [--output_dir <dir>]

Example:
    python scripts/07_plot_extreme_cell_timeline.py results/full_individual_anomalies.parquet --output_dir results/timelines/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from pathlib import Path
from datetime import datetime
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class CellTimelineAnalyzer:
    """Timeline analysis tool for individual cell anomaly patterns."""
    
    def __init__(self, output_dir: str = "results/timelines/"):
        """
        Initialize the timeline analyzer.
        
        Args:
            output_dir: Output directory for generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_individual_anomalies(self, anomalies_path: str) -> pd.DataFrame:
        """Load individual anomaly records."""
        print("Loading individual anomaly data...")
        
        if anomalies_path.endswith('.parquet'):
            df = pd.read_parquet(anomalies_path)
        else:
            df = pd.read_csv(anomalies_path)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"  Loaded {len(df):,} individual anomaly records")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Unique cells: {df['cell_id'].nunique():,}")
        
        return df
    
    def find_most_extreme_cell(self, df: pd.DataFrame) -> int:
        """Find the cell with the most anomalies."""
        cell_counts = df['cell_id'].value_counts()
        most_extreme_cell = cell_counts.idxmax()
        max_count = cell_counts.max()
        
        print(f"  Most extreme cell: {most_extreme_cell} with {max_count:,} anomalies")
        return most_extreme_cell
    
    def analyze_cell_timeline(self, df: pd.DataFrame, cell_id: int) -> pd.DataFrame:
        """Extract and analyze timeline data for a specific cell."""
        print(f"Analyzing timeline for cell {cell_id}...")
        
        # Filter data for the specific cell
        cell_data = df[df['cell_id'] == cell_id].copy()
        
        if len(cell_data) == 0:
            raise ValueError(f"No anomaly data found for cell {cell_id}")
        
        # Sort by timestamp
        cell_data = cell_data.sort_values('timestamp')
        
        # Add temporal features for analysis
        cell_data['hour'] = cell_data['timestamp'].dt.hour
        cell_data['date'] = cell_data['timestamp'].dt.date
        cell_data['day_of_week'] = cell_data['timestamp'].dt.day_name()
        cell_data['week'] = cell_data['timestamp'].dt.isocalendar().week
        
        print(f"  Cell {cell_id} timeline:")
        print(f"    Total anomalies: {len(cell_data):,}")
        print(f"    Date range: {cell_data['timestamp'].min()} to {cell_data['timestamp'].max()}")
        print(f"    Severity range: {cell_data['severity_score'].min():.2f}σ to {cell_data['severity_score'].max():.2f}σ")
        print(f"    Average severity: {cell_data['severity_score'].mean():.2f}σ")
        
        return cell_data
    
    def create_comprehensive_timeline_plot(self, cell_data: pd.DataFrame, cell_id: int):
        """Create a comprehensive timeline visualization."""
        print(f"Creating comprehensive timeline plot for cell {cell_id}...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.5, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        # Main timeline plot (top, spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        
        # Create the main time series plot
        scatter = ax_main.scatter(cell_data['timestamp'], cell_data['severity_score'], 
                                 c=cell_data['severity_score'], cmap='plasma', 
                                 alpha=0.7, s=15, edgecolors='black', linewidth=0.1)
        
        ax_main.set_xlabel('Date and Time', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('Severity Score (σ)', fontsize=12, fontweight='bold')
        ax_main.set_title(f'Cell {cell_id} - Anomaly Severity Timeline\n'
                         f'Total Anomalies: {len(cell_data):,} | '
                         f'Avg Severity: {cell_data["severity_score"].mean():.2f}σ | '
                         f'Max Severity: {cell_data["severity_score"].max():.2f}σ', 
                         fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8)
        cbar.set_label('Severity Score (σ)', fontsize=10)
        
        # Add grid and formatting
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(axis='x', rotation=45)
        
        # Add severity threshold lines
        high_severity_threshold = cell_data['severity_score'].quantile(0.95)
        ax_main.axhline(y=high_severity_threshold, color='red', linestyle='--', alpha=0.7,
                       label=f'95th Percentile: {high_severity_threshold:.1f}σ')
        ax_main.axhline(y=cell_data['severity_score'].mean(), color='blue', linestyle='--', alpha=0.7,
                       label=f'Average: {cell_data["severity_score"].mean():.1f}σ')
        ax_main.legend(loc='upper right')
        
        # Subplot 1: Daily aggregation
        ax1 = fig.add_subplot(gs[1, 0])
        daily_stats = cell_data.groupby('date').agg({
            'severity_score': ['count', 'mean', 'max']
        }).round(2)
        daily_stats.columns = ['count', 'mean_severity', 'max_severity']
        daily_stats = daily_stats.reset_index()
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])
        
        ax1.bar(daily_stats['date'], daily_stats['count'], alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Daily Anomaly Count', fontsize=10)
        ax1.set_title('Daily Anomaly Count', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Hourly pattern
        ax2 = fig.add_subplot(gs[1, 1])
        hourly_stats = cell_data.groupby('hour').agg({
            'severity_score': ['count', 'mean']
        }).round(2)
        hourly_stats.columns = ['count', 'mean_severity']
        
        bars = ax2.bar(hourly_stats.index, hourly_stats['count'], 
                      alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Hour of Day', fontsize=10)
        ax2.set_ylabel('Anomaly Count', fontsize=10)
        ax2.set_title('Hourly Anomaly Pattern', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars for peak hours
        peak_hours = hourly_stats['count'].nlargest(3).index
        for hour in peak_hours:
            count = hourly_stats.loc[hour, 'count']
            ax2.text(hour, count + max(hourly_stats['count']) * 0.01, str(count), 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Subplot 3: Weekly pattern
        ax3 = fig.add_subplot(gs[2, 0])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_stats = cell_data.groupby('day_of_week').agg({
            'severity_score': ['count', 'mean']
        }).round(2)
        weekly_stats.columns = ['count', 'mean_severity']
        weekly_stats = weekly_stats.reindex(day_order)
        
        ax3.bar(range(len(weekly_stats)), weekly_stats['count'], 
               alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Day of Week', fontsize=10)
        ax3.set_ylabel('Anomaly Count', fontsize=10)
        ax3.set_title('Weekly Anomaly Pattern', fontsize=11, fontweight='bold')
        ax3.set_xticks(range(len(day_order)))
        ax3.set_xticklabels([day[:3] for day in day_order])
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Severity distribution
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(cell_data['severity_score'], bins=50, alpha=0.7, color='gold', 
                edgecolor='black', density=True)
        ax4.axvline(cell_data['severity_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {cell_data["severity_score"].mean():.1f}σ')
        ax4.axvline(cell_data['severity_score'].median(), color='blue', linestyle='--', 
                   label=f'Median: {cell_data["severity_score"].median():.1f}σ')
        ax4.set_xlabel('Severity Score (σ)', fontsize=10)
        ax4.set_ylabel('Density', fontsize=10)
        ax4.set_title('Severity Distribution', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Traffic patterns (bottom row, spanning both columns)
        ax5 = fig.add_subplot(gs[3, :])
        
        # Create secondary y-axis for traffic data
        ax5_twin = ax5.twinx()
        
        # Plot severity as line
        line1 = ax5.plot(cell_data['timestamp'], cell_data['severity_score'], 
                        alpha=0.6, color='red', linewidth=0.5, label='Severity Score')
        
        # Plot traffic data as filled areas
        ax5_twin.fill_between(cell_data['timestamp'], 0, cell_data['sms_total'], 
                             alpha=0.3, color='blue', label='SMS Total')
        ax5_twin.fill_between(cell_data['timestamp'], cell_data['sms_total'], 
                             cell_data['sms_total'] + cell_data['calls_total'], 
                             alpha=0.3, color='green', label='Calls Total')
        
        ax5.set_xlabel('Date and Time', fontsize=10)
        ax5.set_ylabel('Severity Score (σ)', fontsize=10, color='red')
        ax5_twin.set_ylabel('Traffic Volume', fontsize=10, color='blue')
        ax5.set_title('Severity vs Traffic Patterns', fontsize=11, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / f'cell_{cell_id}_comprehensive_timeline.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved comprehensive timeline: {output_path}")
        plt.close()
        
        return output_path
    
    def create_detailed_statistics_report(self, cell_data: pd.DataFrame, cell_id: int):
        """Create detailed statistics report for the cell."""
        print(f"Generating detailed statistics report for cell {cell_id}...")
        
        # Calculate comprehensive statistics
        stats = {
            'overview': {
                'cell_id': cell_id,
                'total_anomalies': len(cell_data),
                'date_range': (cell_data['timestamp'].min(), cell_data['timestamp'].max()),
                'days_active': cell_data['date'].nunique(),
                'avg_anomalies_per_day': len(cell_data) / cell_data['date'].nunique()
            },
            'severity': {
                'min': cell_data['severity_score'].min(),
                'max': cell_data['severity_score'].max(),
                'mean': cell_data['severity_score'].mean(),
                'median': cell_data['severity_score'].median(),
                'std': cell_data['severity_score'].std(),
                'p95': cell_data['severity_score'].quantile(0.95),
                'p99': cell_data['severity_score'].quantile(0.99)
            },
            'temporal': {
                'peak_hour': cell_data['hour'].value_counts().idxmax(),
                'peak_day': cell_data['day_of_week'].value_counts().idxmax(),
                'peak_date': cell_data['date'].value_counts().idxmax(),
                'max_daily_anomalies': cell_data['date'].value_counts().max()
            },
            'traffic': {
                'avg_sms': cell_data['sms_total'].mean(),
                'avg_calls': cell_data['calls_total'].mean(),
                'avg_internet': cell_data['internet_traffic'].mean(),
                'max_sms': cell_data['sms_total'].max(),
                'max_calls': cell_data['calls_total'].max(),
                'max_internet': cell_data['internet_traffic'].max()
            }
        }
        
        # Save detailed report
        report_path = self.output_dir / f'cell_{cell_id}_detailed_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"DETAILED ANALYSIS REPORT - CELL {cell_id}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERVIEW:\n")
            f.write(f"  Cell ID: {stats['overview']['cell_id']}\n")
            f.write(f"  Total anomalies: {stats['overview']['total_anomalies']:,}\n")
            f.write(f"  Date range: {stats['overview']['date_range'][0]} to {stats['overview']['date_range'][1]}\n")
            f.write(f"  Days active: {stats['overview']['days_active']}\n")
            f.write(f"  Average anomalies per day: {stats['overview']['avg_anomalies_per_day']:.1f}\n\n")
            
            f.write("SEVERITY STATISTICS:\n")
            f.write(f"  Minimum severity: {stats['severity']['min']:.2f}σ\n")
            f.write(f"  Maximum severity: {stats['severity']['max']:.2f}σ\n")
            f.write(f"  Average severity: {stats['severity']['mean']:.2f}σ\n")
            f.write(f"  Median severity: {stats['severity']['median']:.2f}σ\n")
            f.write(f"  Standard deviation: {stats['severity']['std']:.2f}σ\n")
            f.write(f"  95th percentile: {stats['severity']['p95']:.2f}σ\n")
            f.write(f"  99th percentile: {stats['severity']['p99']:.2f}σ\n\n")
            
            f.write("TEMPORAL PATTERNS:\n")
            f.write(f"  Peak hour: {stats['temporal']['peak_hour']}:00\n")
            f.write(f"  Peak day of week: {stats['temporal']['peak_day']}\n")
            f.write(f"  Most active date: {stats['temporal']['peak_date']}\n")
            f.write(f"  Max anomalies in single day: {stats['temporal']['max_daily_anomalies']}\n\n")
            
            f.write("TRAFFIC PATTERNS:\n")
            f.write(f"  Average SMS: {stats['traffic']['avg_sms']:.1f}\n")
            f.write(f"  Average Calls: {stats['traffic']['avg_calls']:.1f}\n")
            f.write(f"  Average Internet: {stats['traffic']['avg_internet']:.1f}\n")
            f.write(f"  Maximum SMS: {stats['traffic']['max_sms']:.1f}\n")
            f.write(f"  Maximum Calls: {stats['traffic']['max_calls']:.1f}\n")
            f.write(f"  Maximum Internet: {stats['traffic']['max_internet']:.1f}\n\n")
            
            # Top 10 most severe anomalies
            f.write("TOP 10 MOST SEVERE ANOMALIES:\n")
            top_anomalies = cell_data.nlargest(10, 'severity_score')[['timestamp', 'severity_score', 'sms_total', 'calls_total', 'internet_traffic']]
            for i, (_, row) in enumerate(top_anomalies.iterrows(), 1):
                f.write(f"  {i}. {row['timestamp']}: {row['severity_score']:.2f}σ "
                       f"(SMS:{row['sms_total']:.1f}, Calls:{row['calls_total']:.1f}, Internet:{row['internet_traffic']:.1f})\n")
        
        print(f"  Saved detailed report: {report_path}")
        return report_path, stats
    
    def analyze_extreme_cell(self, anomalies_path: str, cell_id: int = None):
        """Complete analysis of the most extreme cell."""
        print("=" * 60)
        print("CMMSE 2025: Extreme Cell Timeline Analysis")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Load data
        df = self.load_individual_anomalies(anomalies_path)
        
        # Find most extreme cell if not specified
        if cell_id is None:
            cell_id = self.find_most_extreme_cell(df)
        
        # Analyze specific cell
        cell_data = self.analyze_cell_timeline(df, cell_id)
        
        # Create visualizations
        plot_path = self.create_comprehensive_timeline_plot(cell_data, cell_id)
        
        # Generate detailed report
        report_path, stats = self.create_detailed_statistics_report(cell_data, cell_id)
        
        # Save cell data for further analysis
        csv_path = self.output_dir / f'cell_{cell_id}_anomaly_data.csv'
        cell_data.to_csv(csv_path, index=False)
        print(f"  Saved cell data: {csv_path}")
        
        total_time = time.perf_counter() - start_time
        
        print("\n" + "=" * 60)
        print("EXTREME CELL ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Cell analyzed: {cell_id}")
        print(f"Total anomalies: {len(cell_data):,}")
        print(f"Severity range: {stats['severity']['min']:.1f}σ to {stats['severity']['max']:.1f}σ")
        print(f"Peak activity: {stats['temporal']['peak_day']} at {stats['temporal']['peak_hour']}:00")
        print(f"Analysis time: {total_time:.2f} seconds")
        print(f"Output directory: {self.output_dir}")
        print("Analysis completed successfully!")
        
        return {
            'cell_id': cell_id,
            'cell_data': cell_data,
            'statistics': stats,
            'plot_path': plot_path,
            'report_path': report_path,
            'csv_path': csv_path
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Extreme Cell Timeline Analysis")
    parser.add_argument("anomalies_path", help="Path to individual anomalies file")
    parser.add_argument("--cell_id", type=int, help="Specific cell ID to analyze (default: most extreme)")
    parser.add_argument("--output_dir", default="results/timelines/",
                       help="Output directory for timeline analysis")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = CellTimelineAnalyzer(args.output_dir)
        
        # Run analysis
        results = analyzer.analyze_extreme_cell(args.anomalies_path, args.cell_id)
        
    except Exception as e:
        print(f"Timeline analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
