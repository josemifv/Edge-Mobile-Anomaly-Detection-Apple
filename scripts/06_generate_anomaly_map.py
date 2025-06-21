#!/usr/bin/env python3
"""
06_generate_anomaly_map.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Stage 6: Geographic Anomaly Visualization

Generates an interactive map showing cell classifications based on their anomaly patterns
using percentiles for classification. Combines geospatial data with anomaly analysis results.

Usage:
    python scripts/06_generate_anomaly_map.py <anomaly_analysis_csv> [--output_dir <dir>]

Example:
    python scripts/06_generate_anomaly_map.py results/test_analysis/cell_anomaly_patterns.csv --output_dir results/maps/
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
warnings.filterwarnings('ignore')


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
        
    def load_anomaly_data(self, anomaly_csv_path: str) -> pd.DataFrame:
        """Load and process anomaly analysis results."""
        print("Loading anomaly analysis data...")
        
        anomaly_df = pd.read_csv(anomaly_csv_path)
        print(f"  Loaded anomaly data for {len(anomaly_df)} cells")
        
        # Extract relevant columns and rename for clarity
        anomaly_df = anomaly_df.rename(columns={
            'severity_score_count': 'anomaly_count',
            'severity_score_mean': 'avg_severity',
            'severity_score_max': 'max_severity',
            'severity_score_std': 'severity_std'
        })
        
        return anomaly_df
    
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
        all_cell_ids = set(range(1, 10001))
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
            popup_text = f"""
            <b>Cell ID:</b> {cell_id}<br>
            <b>Category:</b> {category}<br>
            <b>Anomaly Count:</b> {cell_data.get('anomaly_count', 0)}<br>
            <b>Avg Severity:</b> {cell_data.get('avg_severity', 0):.2f}σ<br>
            <b>Max Severity:</b> {cell_data.get('max_severity', 0):.2f}σ<br>
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
                    f"Avg Severity: {row['avg_severity']:.2f}σ", axis=1),
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
    
    def generate_maps(self, anomaly_csv_path: str, 
                     classification_metrics: list = None):
        """Generate all maps and save to output directory."""
        if classification_metrics is None:
            classification_metrics = ['anomaly_count', 'avg_severity']
        
        print("=" * 60)
        print("CMMSE 2025: Geographic Anomaly Visualization")
        print("=" * 60)
        
        start_time = time.perf_counter()
        
        # Load anomaly data
        anomaly_df = self.load_anomaly_data(anomaly_csv_path)
        
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


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="CMMSE 2025: Geographic Anomaly Visualization")
    parser.add_argument("anomaly_csv_path", help="Path to cell anomaly patterns CSV from Stage 5")
    parser.add_argument("--output_dir", default="results/maps/",
                       help="Output directory for generated maps")
    parser.add_argument("--geojson_path", default="data/raw/milano-grid.geojson",
                       help="Path to Milano grid GeoJSON file")
    parser.add_argument("--metrics", nargs='+', 
                       default=['anomaly_count', 'avg_severity'],
                       help="Classification metrics to use")
    
    args = parser.parse_args()
    
    try:
        # Initialize map generator
        generator = AnomalyMapGenerator(args.geojson_path, args.output_dir)
        
        # Generate maps
        generator.generate_maps(args.anomaly_csv_path, args.metrics)
        
    except Exception as e:
        print(f"Map generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
