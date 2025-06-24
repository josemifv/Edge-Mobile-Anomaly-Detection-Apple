#!/usr/bin/env python3
"""
Pipeline Benchmark Visualization Generator
==========================================

Generates comprehensive visualizations for pipeline benchmark results using:
- matplotlib/seaborn for static PNG visualizations
- plotly for interactive HTML visualizations

Creates:
• Line plot of total pipeline time vs run index
• Boxplots for each stage's timing  
• Heatmap of CPU %, memory and temperature over time for each run
• Histogram of throughput distribution

Usage:
    python scripts/utils/generate_visualizations.py benchmark_dir --output_dir summary_folder
    
Example:
    python scripts/utils/generate_visualizations.py outputs/benchmarks/sample_benchmark_20250624_131351/ --output_dir outputs/benchmarks/sample_benchmark_20250624_131351/summary/

Academic Research Context:
- Edge Mobile Anomaly Detection Pipeline Optimization for Apple Silicon
- CMMSE 2025 Conference Submission
- Performance analysis and visualization for academic publication

Author: José Miguel Franco-Valiente
Institution: Universidad Politécnica de Madrid (UPM)
Project: Edge-Mobile-Anomaly-Detection-Apple
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import sys
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Configure seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkVisualizationGenerator:
    """
    Generates comprehensive visualizations for pipeline benchmark results.
    
    This class creates both static (PNG) and interactive (HTML) visualizations
    for academic research publication and performance analysis.
    """
    
    def __init__(self, benchmark_dir: str, output_dir: str):
        """
        Initialize the visualization generator.
        
        Args:
            benchmark_dir: Path to benchmark results directory
            output_dir: Path to output directory for visualizations
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark data
        self.summary_data = self._load_summary_data()
        self.individual_runs = self._extract_individual_runs()
        self.timing_data = self._extract_timing_data()
        self.resource_data = self._load_resource_monitoring_data()
        
        logger.info(f"Loaded data for {len(self.individual_runs)} runs")
        
    def _load_summary_data(self) -> Dict:
        """Load benchmark summary JSON data."""
        summary_file = self.benchmark_dir / "summary" / "benchmark_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")
            
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    def _extract_individual_runs(self) -> pd.DataFrame:
        """Extract individual run data into a pandas DataFrame."""
        runs_data = []
        
        for run in self.summary_data.get('individual_runs', []):
            if not run.get('success', False):
                continue  # Skip failed runs
                
            run_data = {
                'run_id': run['run_id'],
                'execution_time_seconds': run['execution_time_seconds'],
                'execution_time_minutes': run['execution_time_minutes'],
                'mean_cpu_utilization_percent': run.get('mean_cpu_utilization_percent', 0),
                'peak_cpu_utilization_percent': run.get('peak_cpu_utilization_percent', 0),
                'max_temperature_celsius': run.get('max_temperature_celsius', 0),
                'memory_delta_mb': run['system_metrics']['memory_delta_mb'],
                'initial_memory_mb': run['system_metrics']['initial_memory_mb'],
                'final_memory_mb': run['system_metrics']['final_memory_mb'],
            }
            
            # Add stage timings
            for stage, time_val in run.get('stage_timings', {}).items():
                run_data[stage] = time_val
                
            # Add throughput metrics
            throughput_keys = [k for k in run.keys() if 'rows_per_second' in k]
            for key in throughput_keys:
                run_data[key] = run[key]
                
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def _extract_timing_data(self) -> pd.DataFrame:
        """Extract stage timing data for boxplot analysis."""
        timing_data = []
        
        stage_columns = [col for col in self.individual_runs.columns if col.startswith('stage') and col.endswith('_time')]
        
        for _, run in self.individual_runs.iterrows():
            for stage_col in stage_columns:
                stage_name = stage_col.replace('_time', '').replace('stage', 'Stage ')
                timing_data.append({
                    'run_id': run['run_id'],
                    'stage': stage_name,
                    'time_seconds': run[stage_col]
                })
        
        return pd.DataFrame(timing_data)
    
    def _load_resource_monitoring_data(self) -> Optional[pd.DataFrame]:
        """Load resource monitoring data from individual run folders."""
        resource_files = []
        
        # Look for resource monitoring CSV files in run folders
        for run_dir in self.benchmark_dir.glob("run_*"):
            resource_file = run_dir / "resource_monitor.csv"
            if resource_file.exists():
                try:
                    df = pd.read_csv(resource_file)
                    df['run_id'] = int(run_dir.name.split('_')[1])
                    resource_files.append(df)
                except Exception as e:
                    logger.warning(f"Could not load resource file {resource_file}: {e}")
        
        if resource_files:
            combined_df = pd.concat(resource_files, ignore_index=True)
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            return combined_df
        else:
            logger.warning("No resource monitoring files found")
            return None
    
    def generate_pipeline_time_plot(self) -> str:
        """
        Generate line plot of total pipeline time vs run index.
        
        Returns:
            Path to saved PNG file
        """
        plt.figure(figsize=(12, 8))
        
        # Create line plot with markers
        plt.plot(self.individual_runs['run_id'], 
                self.individual_runs['execution_time_minutes'], 
                marker='o', linewidth=2, markersize=8, 
                color='#2E86AB', markerfacecolor='#A23B72')
        
        # Add trend line
        z = np.polyfit(self.individual_runs['run_id'], self.individual_runs['execution_time_minutes'], 1)
        p = np.poly1d(z)
        plt.plot(self.individual_runs['run_id'], p(self.individual_runs['run_id']), 
                "r--", alpha=0.8, linewidth=1, label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
        
        # Customize plot
        plt.title('Pipeline Execution Time vs Run Index', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Run Index', fontsize=14)
        plt.ylabel('Execution Time (minutes)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics annotation
        mean_time = self.individual_runs['execution_time_minutes'].mean()
        std_time = self.individual_runs['execution_time_minutes'].std()
        plt.text(0.02, 0.98, f'Mean: {mean_time:.2f} ± {std_time:.2f} min', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        output_file = self.output_dir / "pipeline_time_vs_run_index.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated pipeline time plot: {output_file}")
        return str(output_file)
    
    def generate_stage_timing_boxplots(self) -> str:
        """
        Generate boxplots for each stage's timing.
        
        Returns:
            Path to saved PNG file
        """
        plt.figure(figsize=(14, 10))
        
        # Create boxplot
        sns.boxplot(data=self.timing_data, x='stage', y='time_seconds', palette='Set2')
        
        # Customize plot
        plt.title('Stage Execution Time Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Pipeline Stage', fontsize=14)
        plt.ylabel('Execution Time (seconds)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add mean values as text
        for i, stage in enumerate(self.timing_data['stage'].unique()):
            stage_data = self.timing_data[self.timing_data['stage'] == stage]['time_seconds']
            mean_val = stage_data.mean()
            plt.text(i, mean_val, f'{mean_val:.1f}s', ha='center', va='bottom', 
                    fontweight='bold', color='red')
        
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_file = self.output_dir / "stage_timing_boxplots.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated stage timing boxplots: {output_file}")
        return str(output_file)
    
    def generate_resource_heatmap(self) -> str:
        """
        Generate interactive heatmap of CPU %, memory and temperature over time for each run.
        
        Returns:
            Path to saved HTML file
        """
        if self.resource_data is None:
            logger.warning("No resource monitoring data available for heatmap")
            # Create a fallback static heatmap with available data
            return self._generate_static_resource_heatmap()
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Utilization (%)', 'Memory Usage (MB)', 'Temperature (°C)'),
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Color scale for runs
        colors = px.colors.qualitative.Set3
        
        for i, run_id in enumerate(sorted(self.resource_data['run_id'].unique())):
            run_data = self.resource_data[self.resource_data['run_id'] == run_id]
            color = colors[i % len(colors)]
            
            if len(run_data) == 0:
                continue
            
            # CPU utilization
            if 'cpu_percent' in run_data.columns:
                fig.add_trace(
                    go.Scatter(x=run_data.index, y=run_data['cpu_percent'],
                            name=f'Run {run_id} CPU', line=dict(color=color),
                            showlegend=True if i == 0 else False),
                    row=1, col=1
                )
            
            # Memory usage
            memory_cols = [col for col in run_data.columns if 'memory' in col.lower() and 'mb' in col.lower()]
            if memory_cols:
                memory_col = memory_cols[0]  # Use first available memory column
                fig.add_trace(
                    go.Scatter(x=run_data.index, y=run_data[memory_col],
                            name=f'Run {run_id} Memory', line=dict(color=color, dash='dash'),
                            showlegend=True if i == 0 else False),
                    row=2, col=1
                )
            
            # Temperature
            temp_cols = [col for col in run_data.columns if 'temperature' in col.lower()]
            if temp_cols:
                temp_col = temp_cols[0]  # Use first available temperature column
                fig.add_trace(
                    go.Scatter(x=run_data.index, y=run_data[temp_col],
                            name=f'Run {run_id} Temp', line=dict(color=color, dash='dot'),
                            showlegend=True if i == 0 else False),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="Resource Utilization Over Time (Interactive)",
            title_x=0.5,
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="CPU %", row=1, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Time Points", row=3, col=1)
        
        # Save interactive HTML
        output_file = self.output_dir / "resource_utilization_heatmap.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Generated interactive resource heatmap: {output_file}")
        return str(output_file)
    
    def _generate_static_resource_heatmap(self) -> str:
        """Generate static resource heatmap when monitoring data is not available."""
        # Use summary data to create a static heatmap
        runs_data = []
        for _, run in self.individual_runs.iterrows():
            runs_data.append([
                run.get('mean_cpu_utilization_percent', 0),
                run.get('peak_cpu_utilization_percent', 0),
                run.get('memory_delta_mb', 0) / 1000,  # Convert to GB
                run.get('max_temperature_celsius', 0)
            ])
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        heatmap_data = np.array(runs_data).T
        labels = ['Mean CPU %', 'Peak CPU %', 'Memory (GB)', 'Max Temp (°C)']
        run_labels = [f'Run {int(run_id)}' for run_id in self.individual_runs['run_id']]
        
        sns.heatmap(heatmap_data, 
                   xticklabels=run_labels,
                   yticklabels=labels,
                   annot=True, fmt='.1f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Resource Usage'})
        
        plt.title('Resource Utilization Heatmap (Summary)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Pipeline Runs', fontsize=14)
        plt.ylabel('Resource Metrics', fontsize=14)
        
        # Save plot
        output_file = self.output_dir / "resource_utilization_heatmap.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated static resource heatmap: {output_file}")
        return str(output_file)
    
    def generate_throughput_histogram(self) -> str:
        """
        Generate histogram of throughput distribution.
        
        Returns:
            Path to saved PNG file
        """
        # Collect all throughput data
        throughput_data = []
        throughput_labels = []
        
        throughput_cols = [col for col in self.individual_runs.columns if 'rows_per_second' in col]
        
        for col in throughput_cols:
            stage_name = col.replace('_rows_per_second', '').replace('stage', 'Stage ')
            values = self.individual_runs[col].dropna()
            
            for val in values:
                throughput_data.append(val)
                throughput_labels.append(stage_name)
        
        # Create DataFrame for plotting
        throughput_df = pd.DataFrame({
            'throughput': throughput_data,
            'stage': throughput_labels
        })
        
        # Create histogram with subplots for each stage
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        stages = throughput_df['stage'].unique()
        
        for i, stage in enumerate(stages):
            if i >= len(axes):
                break
                
            stage_data = throughput_df[throughput_df['stage'] == stage]['throughput']
            
            axes[i].hist(stage_data, bins=max(3, len(stage_data)//2), 
                        alpha=0.7, color=sns.color_palette("husl", len(stages))[i], 
                        edgecolor='black')
            
            axes[i].set_title(f'{stage} Throughput Distribution', fontweight='bold')
            axes[i].set_xlabel('Throughput (rows/second)')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = stage_data.mean()
            std_val = stage_data.std()
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_val:.0f}')
            axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(stages), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Pipeline Stage Throughput Distribution', fontsize=16, fontweight='bold')
        
        # Save plot
        output_file = self.output_dir / "throughput_distribution_histogram.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated throughput histogram: {output_file}")
        return str(output_file)
    
    def generate_interactive_throughput_plot(self) -> str:
        """
        Generate interactive throughput comparison plot using Plotly.
        
        Returns:
            Path to saved HTML file
        """
        throughput_cols = [col for col in self.individual_runs.columns if 'rows_per_second' in col]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(throughput_cols):
            stage_name = col.replace('_rows_per_second', '').replace('stage', 'Stage ')
            
            fig.add_trace(go.Scatter(
                x=self.individual_runs['run_id'],
                y=self.individual_runs[col],
                mode='lines+markers',
                name=stage_name,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Pipeline Stage Throughput Comparison (Interactive)",
            title_x=0.5,
            xaxis_title="Run Index",
            yaxis_title="Throughput (rows/second)",
            yaxis_type="log",  # Log scale for better visualization
            height=600,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1 run", step="all", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        # Save interactive HTML
        output_file = self.output_dir / "throughput_comparison_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Generated interactive throughput plot: {output_file}")
        return str(output_file)
    
    def generate_comprehensive_dashboard(self) -> str:
        """
        Generate comprehensive interactive dashboard with all metrics.
        
        Returns:
            Path to saved HTML file
        """
        # Create subplots for dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Pipeline Execution Time',
                'CPU Utilization by Run',
                'Memory Usage by Run', 
                'Temperature by Run'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Pipeline execution time
        fig.add_trace(
            go.Scatter(x=self.individual_runs['run_id'], 
                      y=self.individual_runs['execution_time_minutes'],
                      mode='lines+markers', name='Execution Time',
                      line=dict(color='#2E86AB', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        # 2. CPU utilization
        fig.add_trace(
            go.Bar(x=self.individual_runs['run_id'],
                   y=self.individual_runs['mean_cpu_utilization_percent'],
                   name='Mean CPU %', marker_color='lightblue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=self.individual_runs['run_id'],
                   y=self.individual_runs['peak_cpu_utilization_percent'],
                   name='Peak CPU %', marker_color='darkblue'),
            row=1, col=2
        )
        
        # 3. Memory usage
        fig.add_trace(
            go.Scatter(x=self.individual_runs['run_id'],
                      y=self.individual_runs['memory_delta_mb'],
                      mode='lines+markers', name='Memory Delta (MB)',
                      line=dict(color='green', width=3),
                      marker=dict(size=8)),
            row=2, col=1
        )
        
        # 4. Temperature
        fig.add_trace(
            go.Scatter(x=self.individual_runs['run_id'],
                      y=self.individual_runs['max_temperature_celsius'],
                      mode='lines+markers', name='Max Temperature (°C)',
                      line=dict(color='red', width=3),
                      marker=dict(size=8)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Pipeline Performance Dashboard (Interactive)",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Run Index", row=1, col=1)
        fig.update_xaxes(title_text="Run Index", row=1, col=2)
        fig.update_xaxes(title_text="Run Index", row=2, col=1)
        fig.update_xaxes(title_text="Run Index", row=2, col=2)
        
        fig.update_yaxes(title_text="Time (minutes)", row=1, col=1)
        fig.update_yaxes(title_text="CPU %", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)
        
        # Save interactive HTML
        output_file = self.output_dir / "performance_dashboard_interactive.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Generated comprehensive dashboard: {output_file}")
        return str(output_file)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generate all visualizations and return paths to created files.
        
        Returns:
            Dictionary mapping visualization type to file path
        """
        logger.info("Starting comprehensive visualization generation...")
        
        results = {}
        
        try:
            # Static PNG visualizations
            results['pipeline_time_plot'] = self.generate_pipeline_time_plot()
            results['stage_timing_boxplots'] = self.generate_stage_timing_boxplots()
            results['throughput_histogram'] = self.generate_throughput_histogram()
            results['resource_heatmap'] = self.generate_resource_heatmap()
            
            # Interactive HTML visualizations
            results['interactive_throughput'] = self.generate_interactive_throughput_plot()
            results['interactive_dashboard'] = self.generate_comprehensive_dashboard()
            
            logger.info(f"Successfully generated {len(results)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive visualizations for pipeline benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default output to summary folder
    python scripts/utils/generate_visualizations.py outputs/benchmarks/sample_benchmark_20250624_131351/
    
    # Custom output directory
    python scripts/utils/generate_visualizations.py outputs/benchmarks/sample_benchmark_20250624_131351/ --output_dir visualizations/
    
    # Verbose output
    python scripts/utils/generate_visualizations.py outputs/benchmarks/sample_benchmark_20250624_131351/ --verbose
        """
    )
    
    parser.add_argument(
        "benchmark_dir",
        help="Path to benchmark results directory"
    )
    
    parser.add_argument(
        "--output_dir",
        help="Output directory for visualizations (default: benchmark_dir/summary/)",
        default=None
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.benchmark_dir) / "summary")
    
    try:
        # Generate visualizations
        generator = BenchmarkVisualizationGenerator(args.benchmark_dir, args.output_dir)
        results = generator.generate_all_visualizations()
        
        # Print results
        print("\n" + "="*80)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*80)
        print(f"Generated {len(results)} visualizations:")
        print()
        
        for viz_type, file_path in results.items():
            print(f"  ✅ {viz_type:25} → {file_path}")
        
        print(f"\nAll files saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
