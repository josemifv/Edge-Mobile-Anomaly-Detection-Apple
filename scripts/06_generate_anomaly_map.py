#!/usr/bin/env python3
"""
06_generate_anomaly_map.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 6: Geographic Anomaly Visualization

Generates interactive maps showing cell classifications based on their anomaly patterns
using percentiles for classification. Processes individual anomaly records from Stage 4
output and aggregates them into cell-level statistics for geographic visualization.

Data Flow:
- Input: Individual anomaly records (parquet) from Stage 4 
- Processing: Cell-level aggregation with statistical analysis
- Output: Interactive Folium and Plotly maps with percentile-based classification

Features:
- Handles both array and scalar cell_id formats from Stage 4 output
- Percentile-based classification for meaningful geographic patterns
- Dual visualization (Folium interactive + Plotly choropleth)
- Comprehensive statistics and category analysis
- Milano grid integration with full geographic coverage

Usage:
    python scripts/06_generate_anomaly_map.py \u003canomaly_parquet_path\u003e [options]

Required Arguments:
    anomaly_parquet_path    Path to individual anomalies parquet file from Stage 4

Optional Arguments:
    --output_dir DIR        Output directory for maps (default: results/maps/)
    --geojson_path PATH     Milano grid GeoJSON path (default: data/raw/milano-grid.geojson)
    --metrics METRICS       Classification metrics (default: anomaly_count avg_severity)

Examples:
    # Basic usage with Stage 4 output
    python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet --output_dir results/maps/
    
    # With custom metrics and GeoJSON
    python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet \
        --output_dir maps/ --metrics anomaly_count max_severity --geojson_path data/raw/milano-grid.geojson
    
    # Complete pipeline integration
    python scripts/run_pipeline.py data/raw/ --output_dir results/
    python scripts/06_generate_anomaly_map.py results/individual_anomalies.parquet --output_dir results/maps/
"""

import pandas as pd
import numpy as np
import json
import argparse
import time
from pathlib import Path
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import warnings
# Suppress specific known warnings while preserving error visibility
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class AnomalyMapGenerator:
    """Geographic visualization tool for cell anomaly patterns."""
    
    def __init__(self, geojson_path: str = "data/raw/milano-grid.geojson", 
                 output_dir: str = "results/maps/"):
        """
        Initialize the map generator.
        
        Args:
            geojson_path: Path to Milano grid GeoJSON file
            output_dir: Output directory for generated maps
        """
        self.geojson_path = Path(geojson_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load geographic data
        print("Loading geographic data...")
        with open(self.geojson_path, 'r') as f:
            self.geojson_data = json.load(f)
        
        print(f"  Loaded {len(self.geojson_data['features'])} cell geometries")
        
        # Milano center coordinates for map centering
        self.milano_center = [45.4642, 9.1900]
        
    def load_anomaly_data(self, anomaly_parquet_path: str) -> pd.DataFrame:
        """Load and aggregate individual anomaly records from Stage 4 output.
        
        Data Flow:
        1. Load individual anomaly records (parquet format from Stage 4)
        2. Handle cell_id format (convert arrays to scalars if needed)
        3. Aggregate records by cell_id to get cell-level statistics
        4. Transform into format suitable for geographic visualization
        
        Returns:
            DataFrame with cell-level aggregated statistics
        """
        print("Loading individual anomaly records...")
        
        # Step 1: Read parquet file containing individual anomaly records
        # Expected format: cell_id, timestamp, anomaly_score, severity_score, traffic features
        anomalies_df = pd.read_parquet(anomaly_parquet_path)
        print(f"  Loaded {len(anomalies_df):,} individual anomaly records")
        
        # Step 2: Handle cell_id format inconsistencies
        # Stage 4 output may contain cell_id as arrays [cellId] rather than scalars
        first_cell_id = anomalies_df['cell_id'].iloc[0]
        if isinstance(first_cell_id, (list, np.ndarray)):
            print("  Converting cell_id arrays to integers...")
            # Extract first element from array format: [4472] -> 4472
            anomalies_df['cell_id'] = anomalies_df['cell_id'].apply(
                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
            )
        
        # Step 3: Aggregate individual anomaly records into cell-level statistics
        # Transform from individual anomaly records to per-cell summary metrics
        cell_stats = anomalies_df.groupby('cell_id').agg({
            'severity_score': ['count', 'mean', 'max', 'std'],  # Anomaly frequency and severity
            'sms_total': 'mean',           # Average SMS traffic during anomalies
            'calls_total': 'mean',         # Average call traffic during anomalies
            'internet_traffic': 'mean'     # Average internet traffic during anomalies
        }).reset_index()
        
        # Step 4: Flatten hierarchical column names and standardize format
        # Convert from MultiIndex columns to flat structure for easier processing
        cell_stats.columns = [
            'cell_id', 'anomaly_count', 'avg_severity',
            'max_severity', 'severity_std',
            'sms_total_mean', 'calls_total_mean', 'internet_traffic_mean'
        ]
        
        # Calculate weighted severity metric that considers both frequency and severity
        # This gives higher priority to cells with severe anomalies, even if less frequent
        cell_stats['weighted_severity'] = (
            cell_stats['anomaly_count'] * cell_stats['avg_severity']
        )
        
        # Calculate severity impact factor (combines frequency, severity, and maximum impact)
        # This metric balances:
        # - Frequency: how often anomalies occur
        # - Average impact: typical severity level
        # - Peak impact: worst case severity
        cell_stats['severity_impact'] = (
            0.5 * cell_stats['anomaly_count'] * cell_stats['avg_severity'] +  # Weighted frequency
            0.3 * cell_stats['max_severity'] +                                # Peak severity
            0.2 * cell_stats['anomaly_count']                                 # Raw frequency
        )
        
        # Calculate extreme severity filter (p99 of individual anomalies)
        # This identifies cells that have experienced truly extreme incidents
        print("  Calculating extreme severity threshold (p99)...")
        p99_severity = np.percentile(anomalies_df['severity_score'], 99)
        print(f"  P99 severity threshold: {p99_severity:.2f}σ")
        
        # Mark cells that have anomalies above p99 threshold
        extreme_anomalies = anomalies_df[anomalies_df['severity_score'] >= p99_severity]
        cells_with_extreme = set(extreme_anomalies['cell_id'])
        
        cell_stats['has_extreme_anomalies'] = cell_stats['cell_id'].isin(cells_with_extreme)
        cell_stats['extreme_anomaly_count'] = cell_stats['cell_id'].map(
            extreme_anomalies['cell_id'].value_counts()
        ).fillna(0).astype(int)
        
        print(f"  Cells with p99+ severity anomalies: {len(cells_with_extreme):,} ({len(cells_with_extreme)/len(cell_stats)*100:.1f}%)")
        print(f"  Total p99+ severity anomalies: {len(extreme_anomalies):,}")
        
        print(f"  Aggregated stats for {len(cell_stats)} cells")
        print(f"  Cell-level metrics: anomaly_count, severity statistics, traffic averages")
        print(f"  Enhanced metrics: weighted_severity, severity_impact")
        return cell_stats
    
    def classify_cells_by_percentiles(self, anomaly_df: pd.DataFrame, 
                                    classification_metric: str = 'anomaly_count') -> pd.DataFrame:
        """
        Classify cells into categories based on percentiles of the classification metric.
        
        Args:
            anomaly_df: DataFrame with anomaly statistics
            classification_metric: Column to use for classification
            
        Returns:
            DataFrame with classification categories added
        """
        print(f"Classifying cells based on {classification_metric}...")
        
        # Add cells with zero anomalies (missing from analysis)
        # Extract all cell IDs from the GeoJSON data
        all_cell_ids = {f['properties']['cellId'] for f in self.geojson_data['features']}
        analyzed_cell_ids = set(anomaly_df['cell_id'])
        missing_cell_ids = all_cell_ids - analyzed_cell_ids
        
        if missing_cell_ids:
            # Create entries for cells with no anomalies
            zero_anomaly_rows = []
            for cell_id in missing_cell_ids:
                zero_anomaly_rows.append({
                    'cell_id': cell_id,
                    'anomaly_count': 0,
                    'avg_severity': 0,
                    'max_severity': 0,
                    'severity_std': 0,
                    'anomaly_score_mean': 0,
                    'sms_total_mean': 0,
                    'calls_total_mean': 0,
                    'internet_traffic_mean': 0
                })
            
            zero_df = pd.DataFrame(zero_anomaly_rows)
            anomaly_df = pd.concat([anomaly_df, zero_df], ignore_index=True)
            print(f"  Added {len(missing_cell_ids)} cells with zero anomalies")
        
        # Calculate percentiles for classification
        values = anomaly_df[classification_metric]
        
        # Define percentile-based categories
        if classification_metric == 'anomaly_count':
            # Special handling for anomaly count (many zeros)
            zero_mask = values == 0
            non_zero_values = values[~zero_mask]
            
            if len(non_zero_values) > 0:
                p25 = np.percentile(non_zero_values, 25)
                p50 = np.percentile(non_zero_values, 50)
                p75 = np.percentile(non_zero_values, 75)
                p90 = np.percentile(non_zero_values, 90)
                
                def classify_anomaly_count(count):
                    if count == 0:
                        return "No Anomalies"
                    elif count <= p25:
                        return "Low Activity"
                    elif count <= p50:
                        return "Moderate Activity"
                    elif count <= p75:
                        return "High Activity"
                    elif count <= p90:
                        return "Very High Activity"
                    else:
                        return "Extreme Activity"
                
                anomaly_df['category'] = anomaly_df[classification_metric].apply(classify_anomaly_count)
                
                print(f"  Classification thresholds:")
                print(f"    No anomalies: {zero_mask.sum()} cells")
                print(f"    Low (≤{p25:.0f}): up to 25th percentile")
                print(f"    Moderate (≤{p50:.0f}): 25th-50th percentile")
                print(f"    High (≤{p75:.0f}): 50th-75th percentile")
                print(f"    Very High (≤{p90:.0f}): 75th-90th percentile")
                print(f"    Extreme (>{p90:.0f}): >90th percentile")
                
        else:
            # Standard percentile classification for other metrics
            p20 = np.percentile(values, 20)
            p40 = np.percentile(values, 40)
            p60 = np.percentile(values, 60)
            p80 = np.percentile(values, 80)
            
            def classify_standard(value):
                if value <= p20:
                    return "Very Low"
                elif value <= p40:
                    return "Low"
                elif value <= p60:
                    return "Moderate"
                elif value <= p80:
                    return "High"
                else:
                    return "Very High"
            
            anomaly_df['category'] = anomaly_df[classification_metric].apply(classify_standard)
            
            print(f"  Classification thresholds (percentiles):")
            print(f"    Very Low (≤{p20:.2f}): 0-20th percentile")
            print(f"    Low (≤{p40:.2f}): 20th-40th percentile")
            print(f"    Moderate (≤{p60:.2f}): 40th-60th percentile")
            print(f"    High (≤{p80:.2f}): 60th-80th percentile")
            print(f"    Very High (>{p80:.2f}): >80th percentile")
        
        # Print category distribution
        category_counts = anomaly_df['category'].value_counts()
        print(f"  Category distribution:")
        for cat, count in category_counts.items():
            print(f"    {cat}: {count} cells ({count/len(anomaly_df)*100:.1f}%)")
        
        return anomaly_df
    
    def create_folium_map(self, anomaly_df: pd.DataFrame, 
                         classification_metric: str = 'anomaly_count') -> folium.Map:
        """Create interactive Folium map with anomaly classifications."""
        print("Creating Folium interactive map...")
        
        # Create base map
        m = folium.Map(
            location=self.milano_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Define color scheme for categories
        if classification_metric == 'anomaly_count':
            color_map = {
                "No Anomalies": "#2E8B57",        # Sea Green
                "Low Activity": "#90EE90",        # Light Green
                "Moderate Activity": "#FFD700",   # Gold
                "High Activity": "#FF8C00",       # Dark Orange
                "Very High Activity": "#FF4500",  # Orange Red
                "Extreme Activity": "#8B0000"     # Dark Red
            }
        else:
            color_map = {
                "Very Low": "#2E8B57",    # Sea Green
                "Low": "#90EE90",         # Light Green
                "Moderate": "#FFD700",    # Gold
                "High": "#FF8C00",        # Dark Orange
                "Very High": "#8B0000"    # Dark Red
            }
        
        # Create cell dictionary for quick lookup
        cell_dict = {row['cell_id']: row for _, row in anomaly_df.iterrows()}
        
        # Add polygons to map
        for feature in self.geojson_data['features']:
            cell_id = feature['properties']['cellId']
            cell_data = cell_dict.get(cell_id, {})
            
            category = cell_data.get('category', 'Unknown')
            color = color_map.get(category, '#CCCCCC')
            
            # Create popup text with cell information
            has_extreme = cell_data.get('has_extreme_anomalies', False)
            extreme_count = cell_data.get('extreme_anomaly_count', 0)
            extreme_indicator = "🔥 YES" if has_extreme else "No"
            
            popup_text = f"""
            <b>Cell ID:</b> {cell_id}<br>
            <b>Category:</b> {category}<br>
            <b>Anomaly Count:</b> {cell_data.get('anomaly_count', 0)}<br>
            <b>Avg Severity:</b> {cell_data.get('avg_severity', 0):.2f}σ<br>
            <b>Max Severity:</b> {cell_data.get('max_severity', 0):.2f}σ<br>
            <b>P99+ Extreme Anomalies:</b> {extreme_indicator} ({extreme_count})<br>
            <b>Weighted Severity:</b> {cell_data.get('weighted_severity', 0):.1f}<br>
            <b>Severity Impact:</b> {cell_data.get('severity_impact', 0):.1f}<br>
            <b>Avg SMS:</b> {cell_data.get('sms_total_mean', 0):.1f}<br>
            <b>Avg Calls:</b> {cell_data.get('calls_total_mean', 0):.1f}<br>
            <b>Avg Internet:</b> {cell_data.get('internet_traffic_mean', 0):.1f}
            """
            
            folium.GeoJson(
                feature,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Cell {cell_id}: {category}"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Anomaly Classification</b></p>
        '''
        
        for category, color in color_map.items():
            count = len(anomaly_df[anomaly_df['category'] == category])
            legend_html += f'<p><i class="fa fa-square" style="color:{color}"></i> {category} ({count})</p>'
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_plotly_map(self, anomaly_df: pd.DataFrame, 
                         classification_metric: str = 'anomaly_count') -> go.Figure:
        """Create Plotly choropleth map."""
        print("Creating Plotly choropleth map...")
        
        # Prepare data for Plotly
        # Extract cell coordinates from GeoJSON for scatter plot alternative
        cell_coords = []
        for feature in self.geojson_data['features']:
            cell_id = feature['properties']['cellId']
            
            # Get centroid of polygon
            coords = feature['geometry']['coordinates'][0]
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]
            centroid_lon = sum(lons) / len(lons)
            centroid_lat = sum(lats) / len(lats)
            
            cell_coords.append({
                'cell_id': cell_id,
                'lon': centroid_lon,
                'lat': centroid_lat
            })
        
        coord_df = pd.DataFrame(cell_coords)
        plot_df = pd.merge(coord_df, anomaly_df, on='cell_id')
        
        # Create color scale
        if classification_metric == 'anomaly_count':
            # Use actual values for continuous color scale
            fig = go.Figure(data=go.Scattermapbox(
                lat=plot_df['lat'],
                lon=plot_df['lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=plot_df['anomaly_count'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Anomaly Count"),
                    opacity=0.8
                ),
                text=plot_df.apply(lambda row: 
                    f"Cell {row['cell_id']}<br>" +
                    f"Category: {row['category']}<br>" +
                    f"Anomalies: {row['anomaly_count']}<br>" +
                    f"Avg Severity: {row['avg_severity']:.2f}σ<br>" +
                    f"P99+ Extreme: {'🔥 YES' if row.get('has_extreme_anomalies', False) else 'No'} ({row.get('extreme_anomaly_count', 0)})<br>" +
                    f"Weighted Severity: {row.get('weighted_severity', 0):.1f}<br>" +
                    f"Severity Impact: {row.get('severity_impact', 0):.1f}", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
        else:
            # Use categorical colors
            fig = px.scatter_mapbox(
                plot_df, 
                lat='lat', 
                lon='lon',
                color='category',
                hover_data=['cell_id', 'anomaly_count', 'avg_severity'],
                zoom=10,
                height=800
            )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=self.milano_center[0], lon=self.milano_center[1]),
                zoom=10
            ),
            title=f"Milano Cell Network - Anomaly Classification by {classification_metric.replace('_', ' ').title()}",
            height=800,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    def generate_summary_statistics(self, anomaly_df: pd.DataFrame) -> dict:
        """Generate summary statistics for the map."""
        total_cells = len(anomaly_df)
        cells_with_anomalies = len(anomaly_df[anomaly_df['anomaly_count'] > 0])
        total_anomalies = anomaly_df['anomaly_count'].sum()
        
        category_stats = anomaly_df.groupby('category').agg({
            'anomaly_count': ['count', 'sum', 'mean'],
            'avg_severity': 'mean',
            'max_severity': 'max'
        }).round(2)
        
        return {
            'total_cells': total_cells,
            'cells_with_anomalies': cells_with_anomalies,
            'cells_without_anomalies': total_cells - cells_with_anomalies,
            'total_anomalies': total_anomalies,
            'avg_anomalies_per_cell': total_anomalies / total_cells,
            'category_stats': category_stats
        }
    
    def generate_maps(self, anomaly_parquet_path: str, 
                     classification_metrics: list = None):
        """Generate all maps and save to output directory."""
        if classification_metrics is None:
            classification_metrics = ['anomaly_count', 'avg_severity']
        
        print("=" * 60)
        print("CMMSE 2025: Geographic Anomaly Visualization")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Load anomaly data
        anomaly_df = self.load_anomaly_data(anomaly_parquet_path)
        
        for metric in classification_metrics:
            if metric not in anomaly_df.columns:
                print(f"Warning: Metric '{metric}' not found in data. Skipping.")
                continue
                
            print(f"\n--- Generating maps for {metric} ---")
            
            # Classify cells
            classified_df = self.classify_cells_by_percentiles(anomaly_df, metric)
            
            # Generate summary
            stats = self.generate_summary_statistics(classified_df)
            
            # Create Folium map
            folium_map = self.create_folium_map(classified_df, metric)
            folium_path = self.output_dir / f"milano_anomaly_map_{metric}_folium.html"
            folium_map.save(str(folium_path))
            print(f"  Saved Folium map: {folium_path}")
            
            # Create Plotly map
            plotly_fig = self.create_plotly_map(classified_df, metric)
            plotly_path = self.output_dir / f"milano_anomaly_map_{metric}_plotly.html"
            plot(plotly_fig, filename=str(plotly_path), auto_open=False)
            print(f"  Saved Plotly map: {plotly_path}")
            
            # Save classification data
            csv_path = self.output_dir / f"cell_classification_{metric}.csv"
            classified_df.to_csv(csv_path, index=False)
            print(f"  Saved classification data: {csv_path}")
            
            # Save summary statistics
            summary_path = self.output_dir / f"classification_summary_{metric}.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Milano Cell Network - Anomaly Classification Summary\n")
                f.write(f"Classification Metric: {metric}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"OVERVIEW:\n")
                f.write(f"  Total cells: {stats['total_cells']:,}\n")
                f.write(f"  Cells with anomalies: {stats['cells_with_anomalies']:,}\n")
                f.write(f"  Cells without anomalies: {stats['cells_without_anomalies']:,}\n")
                f.write(f"  Total anomalies: {stats['total_anomalies']:,}\n")
                f.write(f"  Average anomalies per cell: {stats['avg_anomalies_per_cell']:.1f}\n\n")
                
                f.write("CATEGORY BREAKDOWN:\n")
                for category in classified_df['category'].unique():
                    cat_data = classified_df[classified_df['category'] == category]
                    f.write(f"  {category}:\n")
                    f.write(f"    Cells: {len(cat_data)} ({len(cat_data)/len(classified_df)*100:.1f}%)\n")
                    f.write(f"    Total anomalies: {cat_data['anomaly_count'].sum():,}\n")
                    f.write(f"    Avg anomalies per cell: {cat_data['anomaly_count'].mean():.1f}\n")
                    if cat_data['anomaly_count'].sum() > 0:
                        f.write(f"    Avg severity: {cat_data['avg_severity'].mean():.2f}σ\n")
                    f.write("\n")
            
            print(f"  Saved summary: {summary_path}")
        
        total_time = time.perf_counter() - start_time
        
        print("\n" + "=" * 60)
        print("MAP GENERATION SUMMARY")
        print("=" * 60)
        print(f"Maps generated: {len(classification_metrics)} metrics")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created: {len(classification_metrics) * 4} files")
        print(f"Generation time: {total_time:.2f} seconds")
        print("Map generation completed successfully!")
    
    def generate_extreme_anomaly_map(self, anomaly_parquet_path: str):
        """Generate special map showing only cells with p99+ extreme severity anomalies."""
        print("Generating EXTREME SEVERITY FILTER MAP...")
        
        # Load data and get extreme anomaly information
        anomaly_df = self.load_anomaly_data(anomaly_parquet_path)
        
        # Filter to only cells with extreme anomalies
        extreme_cells = anomaly_df[anomaly_df['has_extreme_anomalies'] == True].copy()
        
        if len(extreme_cells) == 0:
            print("  No cells with p99+ extreme anomalies found.")
            return
        
        print(f"  Found {len(extreme_cells)} cells with extreme anomalies (p99+ severity)")
        
        # Create special classification based on extreme anomaly count
        extreme_counts = extreme_cells['extreme_anomaly_count']
        p25_ext = np.percentile(extreme_counts, 25)
        p50_ext = np.percentile(extreme_counts, 50) 
        p75_ext = np.percentile(extreme_counts, 75)
        
        def classify_extreme_count(count):
            if count <= p25_ext:
                return "Low Extreme"
            elif count <= p50_ext:
                return "Moderate Extreme"
            elif count <= p75_ext:
                return "High Extreme"
            else:
                return "Critical Extreme"
        
        extreme_cells['category'] = extreme_cells['extreme_anomaly_count'].apply(classify_extreme_count)
        
        print(f"  Extreme anomaly classification thresholds:")
        print(f"    Low Extreme (≤{p25_ext:.0f}): up to 25th percentile")
        print(f"    Moderate Extreme (≤{p50_ext:.0f}): 25th-50th percentile")
        print(f"    High Extreme (≤{p75_ext:.0f}): 50th-75th percentile")
        print(f"    Critical Extreme (>{p75_ext:.0f}): >75th percentile")
        
        # Print category distribution
        category_counts = extreme_cells['category'].value_counts()
        print(f"  Extreme category distribution:")
        for cat, count in category_counts.items():
            print(f"    {cat}: {count} cells ({count/len(extreme_cells)*100:.1f}%)")
        
        # Create Folium map for extreme cells only
        print("Creating extreme severity Folium map...")
        
        m = folium.Map(
            location=self.milano_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Define color scheme for extreme classifications
        extreme_color_map = {
            "Low Extreme": "#FFA500",      # Orange
            "Moderate Extreme": "#FF6347", # Tomato  
            "High Extreme": "#DC143C",     # Crimson
            "Critical Extreme": "#8B0000"  # Dark Red
        }
        
        # Create cell dictionary for quick lookup
        extreme_cell_dict = {row['cell_id']: row for _, row in extreme_cells.iterrows()}
        
        # Add only extreme cells to map
        for feature in self.geojson_data['features']:
            cell_id = feature['properties']['cellId']
            
            if cell_id in extreme_cell_dict:
                cell_data = extreme_cell_dict[cell_id]
                category = cell_data['category']
                color = extreme_color_map[category]
                
                # Enhanced popup for extreme cells
                popup_text = f"""
                <b>🔥 EXTREME SEVERITY CELL</b><br>
                <b>Cell ID:</b> {cell_id}<br>
                <b>Extreme Category:</b> {category}<br>
                <b>P99+ Extreme Anomalies:</b> {cell_data['extreme_anomaly_count']}<br>
                <b>Total Anomalies:</b> {cell_data['anomaly_count']}<br>
                <b>Max Severity:</b> {cell_data['max_severity']:.2f}σ<br>
                <b>Avg Severity:</b> {cell_data['avg_severity']:.2f}σ<br>
                <b>Weighted Severity:</b> {cell_data['weighted_severity']:.1f}<br>
                <b>Severity Impact:</b> {cell_data['severity_impact']:.1f}<br>
                <b>Avg SMS:</b> {cell_data['sms_total_mean']:.1f}<br>
                <b>Avg Calls:</b> {cell_data['calls_total_mean']:.1f}<br>
                <b>Avg Internet:</b> {cell_data['internet_traffic_mean']:.1f}
                """
                
                folium.GeoJson(
                    feature,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1.5,
                        'fillOpacity': 0.8
                    },
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"🔥 Cell {cell_id}: {category} ({cell_data['extreme_anomaly_count']} extreme)"
                ).add_to(m)
            else:
                # Add other cells as light gray background
                folium.GeoJson(
                    feature,
                    style_function=lambda x: {
                        'fillColor': '#E8E8E8',
                        'color': '#CCCCCC',
                        'weight': 0.2,
                        'fillOpacity': 0.3
                    },
                    tooltip=f"Cell {cell_id}: No extreme anomalies"
                ).add_to(m)
        
        # Add legend for extreme map
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>🔥 EXTREME SEVERITY CLASSIFICATION</b></p>
        <p><i>Cells with P99+ Severity Anomalies Only</i></p>
        '''
        
        for category, color in extreme_color_map.items():
            count = len(extreme_cells[extreme_cells['category'] == category])
            legend_html += f'<p><i class="fa fa-square" style="color:{color}"></i> {category} ({count})</p>'
        
        other_cells = len(self.geojson_data['features']) - len(extreme_cells)
        legend_html += f'<p><i class="fa fa-square" style="color:#E8E8E8"></i> No Extreme Anomalies ({other_cells})</p>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save extreme severity map
        extreme_path = self.output_dir / "milano_extreme_severity_filter.html"
        m.save(str(extreme_path))
        print(f"  Saved extreme severity map: {extreme_path}")
        
        # Save extreme cells data
        extreme_csv_path = self.output_dir / "extreme_severity_cells.csv"
        extreme_cells.to_csv(extreme_csv_path, index=False)
        print(f"  Saved extreme cells data: {extreme_csv_path}")
        
        # Save extreme summary
        extreme_summary_path = self.output_dir / "extreme_severity_summary.txt"
        with open(extreme_summary_path, 'w') as f:
            f.write("Milano Cell Network - EXTREME SEVERITY ANALYSIS\n")
            f.write("P99+ Severity Anomalies Only\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"EXTREME ANOMALY OVERVIEW:\n")
            f.write(f"  Total cells analyzed: {len(anomaly_df):,}\n")
            f.write(f"  Cells with extreme anomalies: {len(extreme_cells):,} ({len(extreme_cells)/len(anomaly_df)*100:.1f}%)\n")
            f.write(f"  Total extreme anomalies: {extreme_cells['extreme_anomaly_count'].sum():,}\n")
            f.write(f"  Average extreme anomalies per extreme cell: {extreme_cells['extreme_anomaly_count'].mean():.1f}\n\n")
            
            f.write("EXTREME CATEGORY BREAKDOWN:\n")
            for category in extreme_cells['category'].unique():
                cat_data = extreme_cells[extreme_cells['category'] == category]
                f.write(f"  {category}:\n")
                f.write(f"    Cells: {len(cat_data)} ({len(cat_data)/len(extreme_cells)*100:.1f}%)\n")
                f.write(f"    Extreme anomalies: {cat_data['extreme_anomaly_count'].sum():,}\n")
                f.write(f"    Avg extreme per cell: {cat_data['extreme_anomaly_count'].mean():.1f}\n")
                f.write(f"    Max severity: {cat_data['max_severity'].max():.2f}σ\n")
                f.write(f"    Avg severity: {cat_data['avg_severity'].mean():.2f}σ\n")
                f.write("\n")
        
        print(f"  Saved extreme summary: {extreme_summary_path}")
        print("✅ Extreme severity filter map generation completed!")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Geographic Anomaly Visualization")
    parser.add_argument("anomaly_parquet_path", type=Path,
                       help="Path to individual anomalies parquet file from Stage 4")
    parser.add_argument("--output_dir", default="results/maps/",
                       help="Output directory for generated maps")
    parser.add_argument("--geojson_path", default="data/raw/milano-grid.geojson",
                       help="Path to Milano grid GeoJSON file")
    parser.add_argument("--metrics", nargs='+', 
                       default=['anomaly_count', 'weighted_severity'],
                       help="Classification metrics to use (options: anomaly_count, avg_severity, weighted_severity, severity_impact)")
    parser.add_argument("--extreme_only", action="store_true",
                       help="Generate additional map showing only cells with p99+ extreme severity anomalies")
    
    args = parser.parse_args()
    
    try:
        # Initialize map generator
        generator = AnomalyMapGenerator(args.geojson_path, args.output_dir)
        
        # Generate maps
        generator.generate_maps(args.anomaly_parquet_path, args.metrics)
        
        # Generate extreme anomaly filter map if requested
        if args.extreme_only:
            print("\n--- Generating EXTREME SEVERITY FILTER MAP ---")
            generator.generate_extreme_anomaly_map(args.anomaly_parquet_path)
        
    except Exception as e:
        print(f"Map generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
