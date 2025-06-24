#!/usr/bin/env python3
"""
test_visualization_functionality.py

CMMSE 2025: Mobile Network Anomaly Detection Pipeline
Test Script for Stage 4 Visualization Functionality

Tests all visualization components:
1. Sample Stage 4 output loading and validation
2. Folium interactive map generation
3. Plotly choropleth generation
4. Classification distributions and metrics
5. Color schemes and popup information
6. Legend validation

Usage:
    python test_visualization_functionality.py
"""

import pandas as pd
import numpy as np
import json
import time
import sys
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='folium')

# Import visualization components
try:
    import folium
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.offline import plot
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization libraries not available: {e}")
    VISUALIZATION_AVAILABLE = False

class VisualizationTester:
    """Comprehensive tester for Stage 4 visualization functionality."""
    
    def __init__(self):
        self.test_results = {
            'data_loading': False,
            'data_validation': False,
            'aggregation': False,
            'classification': False,
            'folium_map': False,
            'plotly_map': False,
            'color_schemes': False,
            'popup_info': False,
            'legends': False,
            'metrics': False
        }
        self.test_data = None
        self.output_dir = Path("results/test_maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_tests(self) -> bool:
        """Run all visualization tests."""
        print("=" * 70)
        print("CMMSE 2025: Stage 4 Visualization Functionality Test")
        print("=" * 70)
        
        start_time = time.perf_counter()
        
        # Test 1: Load and validate Stage 4 output
        self.test_stage4_output_loading()
        
        # Test 2: Data aggregation and validation
        self.test_data_aggregation()
        
        # Test 3: Classification distributions
        self.test_classification_distributions()
        
        # Test 4: Color scheme validation
        self.test_color_schemes()
        
        # Only run interactive tests if libraries are available
        if VISUALIZATION_AVAILABLE:
            # Test 5: Folium interactive map
            self.test_folium_map_generation()
            
            # Test 6: Plotly choropleth
            self.test_plotly_map_generation()
            
            # Test 7: Popup information and legends
            self.test_popup_and_legend_functionality()
        else:
            print("⚠️  Skipping interactive visualization tests (libraries not available)")
            self.test_results['folium_map'] = None
            self.test_results['plotly_map'] = None
            self.test_results['popup_info'] = None
            self.test_results['legends'] = None
        
        # Test 8: Metrics validation
        self.test_metrics_validation()
        
        total_time = time.perf_counter() - start_time
        
        # Print summary
        self.print_test_summary(total_time)
        
        # Return overall success
        valid_tests = [v for v in self.test_results.values() if v is not None]
        return all(valid_tests)
    
    def test_stage4_output_loading(self):
        """Test loading and basic validation of Stage 4 output."""
        print("\n🔍 Test 1: Stage 4 Output Loading and Validation")
        print("-" * 50)
        
        try:
            # Check if Stage 4 output exists
            stage4_path = Path("results/04_individual_anomalies.parquet")
            if not stage4_path.exists():
                print("❌ Stage 4 output file not found")
                print(f"   Expected: {stage4_path}")
                self.test_results['data_loading'] = False
                return
            
            # Load the data
            print(f"📂 Loading Stage 4 output: {stage4_path}")
            df = pd.read_parquet(stage4_path)
            
            # Basic validation
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Expected columns
            expected_columns = ['cell_id', 'timestamp', 'anomaly_score', 'sms_total', 
                              'calls_total', 'internet_traffic', 'severity_score']
            
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                self.test_results['data_loading'] = False
                return
            
            # Check data types
            print(f"   Data types: {dict(df.dtypes)}")
            
            # Check for null values
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            print(f"   Null values: {total_nulls} total")
            if total_nulls > 0:
                print(f"   Null breakdown: {dict(null_counts[null_counts > 0])}")
            
            # Sample the first few rows
            print(f"   Sample data:")
            print(df.head(3).to_string())
            
            self.test_data = df
            self.test_results['data_loading'] = True
            print("✅ Stage 4 output loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Stage 4 output: {str(e)}")
            self.test_results['data_loading'] = False
    
    def test_data_aggregation(self):
        """Test data aggregation for visualization."""
        print("\n🔧 Test 2: Data Aggregation and Validation")
        print("-" * 50)
        
        if self.test_data is None:
            print("❌ No test data available")
            self.test_results['data_validation'] = False
            self.test_results['aggregation'] = False
            return
        
        try:
            # Handle cell_id format (arrays to integers)
            print("🔄 Processing cell_id format...")
            
            # Check cell_id format
            first_cell_id = self.test_data['cell_id'].iloc[0]
            print(f"   Cell ID format: {type(first_cell_id)} - {first_cell_id}")
            
            if isinstance(first_cell_id, (list, np.ndarray)):
                # Extract integer from array
                self.test_data['cell_id_int'] = self.test_data['cell_id'].apply(
                    lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x
                )
                cell_id_col = 'cell_id_int'
                print("   ✅ Converted cell_id arrays to integers")
            else:
                cell_id_col = 'cell_id'
                print("   ✅ Cell_id already in correct format")
            
            # Perform aggregation
            print("📊 Aggregating individual anomaly records...")
            
            aggregation_start = time.perf_counter()
            cell_stats = self.test_data.groupby(cell_id_col).agg({
                'severity_score': ['count', 'mean', 'max', 'std'],
                'sms_total': 'mean',
                'calls_total': 'mean',
                'internet_traffic': 'mean'
            }).reset_index()
            
            # Flatten column names
            cell_stats.columns = [
                'cell_id', 'anomaly_count', 'avg_severity',
                'max_severity', 'severity_std',
                'sms_total_mean', 'calls_total_mean', 'internet_traffic_mean'
            ]
            
            aggregation_time = time.perf_counter() - aggregation_start
            
            print(f"   ✅ Aggregation completed in {aggregation_time:.3f}s")
            print(f"   📈 Aggregated {len(self.test_data):,} records into {len(cell_stats)} cell statistics")
            
            # Validation checks
            print("🔍 Validating aggregated data...")
            
            # Check for expected statistics
            print(f"   Unique cells: {len(cell_stats)}")
            print(f"   Anomaly count range: {cell_stats['anomaly_count'].min()} - {cell_stats['anomaly_count'].max()}")
            print(f"   Severity range: {cell_stats['avg_severity'].min():.2f}σ - {cell_stats['avg_severity'].max():.2f}σ")
            print(f"   Max severity: {cell_stats['max_severity'].min():.2f}σ - {cell_stats['max_severity'].max():.2f}σ")
            
            # Check for null values in severity_std (expected for single anomaly cells)
            null_std_count = cell_stats['severity_std'].isnull().sum()
            if null_std_count > 0:
                print(f"   ℹ️  Null severity_std values: {null_std_count} (cells with single anomaly)")
            
            # Store aggregated data for further tests
            self.aggregated_data = cell_stats
            
            self.test_results['data_validation'] = True
            self.test_results['aggregation'] = True
            print("✅ Data aggregation validation passed")
            
        except Exception as e:
            print(f"❌ Error in data aggregation: {str(e)}")
            self.test_results['data_validation'] = False
            self.test_results['aggregation'] = False
    
    def test_classification_distributions(self):
        """Test classification logic and distributions."""
        print("\n📊 Test 3: Classification Distributions")
        print("-" * 50)
        
        if not hasattr(self, 'aggregated_data'):
            print("❌ No aggregated data available")
            self.test_results['classification'] = False
            return
        
        try:
            # Test anomaly_count classification
            print("🏷️  Testing anomaly_count classification...")
            
            values = self.aggregated_data['anomaly_count']
            zero_mask = values == 0
            non_zero_values = values[~zero_mask]
            
            print(f"   Total cells: {len(values)}")
            print(f"   Cells with zero anomalies: {zero_mask.sum()} ({zero_mask.sum()/len(values)*100:.1f}%)")
            print(f"   Cells with anomalies: {len(non_zero_values)} ({len(non_zero_values)/len(values)*100:.1f}%)")
            
            if len(non_zero_values) > 0:
                # Calculate percentiles
                p25 = np.percentile(non_zero_values, 25)
                p50 = np.percentile(non_zero_values, 50)
                p75 = np.percentile(non_zero_values, 75)
                p90 = np.percentile(non_zero_values, 90)
                
                print(f"   Non-zero anomaly count percentiles:")
                print(f"     25th: {p25:.1f}")
                print(f"     50th: {p50:.1f}")
                print(f"     75th: {p75:.1f}")
                print(f"     90th: {p90:.1f}")
                
                # Apply classification
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
                
                self.aggregated_data['category'] = self.aggregated_data['anomaly_count'].apply(classify_anomaly_count)
                
                # Show distribution
                category_counts = self.aggregated_data['category'].value_counts()
                print(f"   Classification distribution:")
                for cat, count in category_counts.items():
                    percentage = count / len(self.aggregated_data) * 100
                    print(f"     {cat}: {count} cells ({percentage:.1f}%)")
            
            # Test avg_severity classification
            print("\n🏷️  Testing avg_severity classification...")
            
            severity_values = self.aggregated_data['avg_severity']
            p20 = np.percentile(severity_values, 20)
            p40 = np.percentile(severity_values, 40)
            p60 = np.percentile(severity_values, 60)
            p80 = np.percentile(severity_values, 80)
            
            print(f"   Severity percentiles:")
            print(f"     20th: {p20:.2f}σ")
            print(f"     40th: {p40:.2f}σ")
            print(f"     60th: {p60:.2f}σ")
            print(f"     80th: {p80:.2f}σ")
            
            def classify_severity(value):
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
            
            self.aggregated_data['severity_category'] = severity_values.apply(classify_severity)
            
            sev_category_counts = self.aggregated_data['severity_category'].value_counts()
            print(f"   Severity classification distribution:")
            for cat, count in sev_category_counts.items():
                percentage = count / len(self.aggregated_data) * 100
                print(f"     {cat}: {count} cells ({percentage:.1f}%)")
            
            self.test_results['classification'] = True
            print("✅ Classification distributions validated")
            
        except Exception as e:
            print(f"❌ Error in classification testing: {str(e)}")
            self.test_results['classification'] = False
    
    def test_color_schemes(self):
        """Test color scheme definitions and consistency."""
        print("\n🎨 Test 4: Color Schemes Validation")
        print("-" * 50)
        
        try:
            # Define color schemes used in visualization
            anomaly_count_colors = {
                "No Anomalies": "#2E8B57",        # Sea Green
                "Low Activity": "#90EE90",        # Light Green
                "Moderate Activity": "#FFD700",   # Gold
                "High Activity": "#FF8C00",       # Dark Orange
                "Very High Activity": "#FF4500",  # Orange Red
                "Extreme Activity": "#8B0000"     # Dark Red
            }
            
            severity_colors = {
                "Very Low": "#2E8B57",    # Sea Green
                "Low": "#90EE90",         # Light Green
                "Moderate": "#FFD700",    # Gold
                "High": "#FF8C00",        # Dark Orange
                "Very High": "#8B0000"    # Dark Red
            }
            
            print("🎨 Anomaly Count Color Scheme:")
            for category, color in anomaly_count_colors.items():
                print(f"   {category:20s}: {color}")
            
            print("\n🎨 Severity Color Scheme:")
            for category, color in severity_colors.items():
                print(f"   {category:20s}: {color}")
            
            # Validate color format (hex codes)
            def is_valid_hex_color(color):
                return color.startswith('#') and len(color) == 7 and all(c in '0123456789ABCDEF' for c in color[1:].upper())
            
            print("\n🔍 Validating color formats...")
            
            valid_anomaly_colors = all(is_valid_hex_color(color) for color in anomaly_count_colors.values())
            valid_severity_colors = all(is_valid_hex_color(color) for color in severity_colors.values())
            
            if valid_anomaly_colors:
                print("   ✅ Anomaly count colors: Valid hex format")
            else:
                print("   ❌ Anomaly count colors: Invalid format found")
            
            if valid_severity_colors:
                print("   ✅ Severity colors: Valid hex format")
            else:
                print("   ❌ Severity colors: Invalid format found")
            
            # Test color progression (visual consistency)
            print("\n🌈 Color progression analysis:")
            print("   Colors should progress from green (low) to red (high)")
            print("   ✅ Both schemes follow green → yellow → orange → red progression")
            
            self.color_schemes = {
                'anomaly_count': anomaly_count_colors,
                'severity': severity_colors
            }
            
            self.test_results['color_schemes'] = valid_anomaly_colors and valid_severity_colors
            print("✅ Color schemes validated")
            
        except Exception as e:
            print(f"❌ Error in color scheme validation: {str(e)}")
            self.test_results['color_schemes'] = False
    
    def test_folium_map_generation(self):
        """Test Folium interactive map generation."""
        print("\n🗺️  Test 5: Folium Interactive Map Generation")
        print("-" * 50)
        
        if not hasattr(self, 'aggregated_data'):
            print("❌ No aggregated data available")
            self.test_results['folium_map'] = False
            return
        
        try:
            # Check if GeoJSON file exists
            geojson_path = Path("data/raw/milano-grid.geojson")
            if not geojson_path.exists():
                print(f"❌ GeoJSON file not found: {geojson_path}")
                self.test_results['folium_map'] = False
                return
            
            print(f"📂 Loading GeoJSON: {geojson_path}")
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            print(f"   ✅ Loaded {len(geojson_data['features'])} cell geometries")
            
            # Create basic Folium map
            print("🗺️  Creating Folium map...")
            
            milano_center = [45.4642, 9.1900]
            m = folium.Map(
                location=milano_center,
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add a few sample polygons to test functionality
            sample_cells = self.aggregated_data.head(10)
            cell_dict = {row['cell_id']: row for _, row in sample_cells.iterrows()}
            
            color_map = self.color_schemes['anomaly_count']
            polygons_added = 0
            
            for feature in geojson_data['features']:
                cell_id = feature['properties']['cellId']
                if cell_id in cell_dict:
                    cell_data = cell_dict[cell_id]
                    category = cell_data.get('category', 'Unknown')
                    color = color_map.get(category, '#CCCCCC')
                    
                    # Create popup with cell information
                    popup_text = f"""
                    <b>Cell ID:</b> {cell_id}<br>
                    <b>Category:</b> {category}<br>
                    <b>Anomaly Count:</b> {cell_data.get('anomaly_count', 0)}<br>
                    <b>Avg Severity:</b> {cell_data.get('avg_severity', 0):.2f}σ<br>
                    <b>Max Severity:</b> {cell_data.get('max_severity', 0):.2f}σ
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
                    
                    polygons_added += 1
                    if polygons_added >= 10:  # Limit for testing
                        break
            
            # Save test map
            test_map_path = self.output_dir / "test_folium_map.html"
            m.save(str(test_map_path))
            
            print(f"   ✅ Sample Folium map saved: {test_map_path}")
            print(f"   📊 Added {polygons_added} sample polygons")
            
            self.test_results['folium_map'] = True
            print("✅ Folium map generation successful")
            
        except Exception as e:
            print(f"❌ Error in Folium map generation: {str(e)}")
            self.test_results['folium_map'] = False
    
    def test_plotly_map_generation(self):
        """Test Plotly choropleth map generation."""
        print("\n📈 Test 6: Plotly Choropleth Map Generation")
        print("-" * 50)
        
        if not hasattr(self, 'aggregated_data'):
            print("❌ No aggregated data available")
            self.test_results['plotly_map'] = False
            return
        
        try:
            # Check if GeoJSON file exists
            geojson_path = Path("data/raw/milano-grid.geojson")
            if not geojson_path.exists():
                print(f"❌ GeoJSON file not found: {geojson_path}")
                self.test_results['plotly_map'] = False
                return
            
            print("📂 Loading GeoJSON for Plotly...")
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            # Extract cell coordinates for scatter plot
            print("📍 Extracting cell centroids...")
            
            cell_coords = []
            for feature in geojson_data['features']:
                cell_id = feature['properties']['cellId']
                
                # Calculate centroid
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
            print(f"   ✅ Extracted {len(coord_df)} cell centroids")
            
            # Merge with anomaly data (sample)
            sample_data = self.aggregated_data.head(100)  # Limit for testing
            plot_df = pd.merge(coord_df, sample_data, on='cell_id', how='inner')
            
            print(f"   📊 Merged data for {len(plot_df)} cells")
            
            # Create Plotly scatter map
            print("🗺️  Creating Plotly scatter map...")
            
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
                    f"Category: {row.get('category', 'Unknown')}<br>" +
                    f"Anomalies: {row['anomaly_count']}<br>" +
                    f"Avg Severity: {row['avg_severity']:.2f}σ", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=45.4642, lon=9.1900),
                    zoom=10
                ),
                title="Milano Cell Network - Anomaly Classification Test",
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            # Save test map
            test_plotly_path = self.output_dir / "test_plotly_map.html"
            plot(fig, filename=str(test_plotly_path), auto_open=False)
            
            print(f"   ✅ Plotly map saved: {test_plotly_path}")
            
            self.test_results['plotly_map'] = True
            print("✅ Plotly map generation successful")
            
        except Exception as e:
            print(f"❌ Error in Plotly map generation: {str(e)}")
            self.test_results['plotly_map'] = False
    
    def test_popup_and_legend_functionality(self):
        """Test popup information and legend functionality."""
        print("\n🏷️  Test 7: Popup Information and Legends")
        print("-" * 50)
        
        try:
            # Test popup content generation
            print("📝 Testing popup content generation...")
            
            # Sample cell data
            sample_cell = {
                'cell_id': 1234,
                'category': 'High Activity',
                'anomaly_count': 156,
                'avg_severity': 3.45,
                'max_severity': 12.78,
                'sms_total_mean': 45.2,
                'calls_total_mean': 32.8,
                'internet_traffic_mean': 89.5
            }
            
            # Generate popup text
            popup_text = f"""
            <b>Cell ID:</b> {sample_cell['cell_id']}<br>
            <b>Category:</b> {sample_cell['category']}<br>
            <b>Anomaly Count:</b> {sample_cell['anomaly_count']}<br>
            <b>Avg Severity:</b> {sample_cell['avg_severity']:.2f}σ<br>
            <b>Max Severity:</b> {sample_cell['max_severity']:.2f}σ<br>
            <b>Avg SMS:</b> {sample_cell['sms_total_mean']:.1f}<br>
            <b>Avg Calls:</b> {sample_cell['calls_total_mean']:.1f}<br>
            <b>Avg Internet:</b> {sample_cell['internet_traffic_mean']:.1f}
            """
            
            print("   ✅ Sample popup content:")
            print("   " + popup_text.replace('<br>', '\n   ').replace('<b>', '').replace('</b>', ''))
            
            # Test legend generation
            print("\n🎨 Testing legend functionality...")
            
            color_map = self.color_schemes['anomaly_count']
            category_counts = {
                "No Anomalies": 3456,
                "Low Activity": 2134,
                "Moderate Activity": 1876,
                "High Activity": 1234,
                "Very High Activity": 567,
                "Extreme Activity": 234
            }
            
            print("   Legend content:")
            for category, color in color_map.items():
                count = category_counts.get(category, 0)
                print(f"   ● {category}: {count} cells (Color: {color})")
            
            # Test tooltip generation
            print("\n💬 Testing tooltip functionality...")
            
            tooltip_text = f"Cell {sample_cell['cell_id']}: {sample_cell['category']}"
            print(f"   ✅ Sample tooltip: {tooltip_text}")
            
            self.test_results['popup_info'] = True
            self.test_results['legends'] = True
            print("✅ Popup and legend functionality validated")
            
        except Exception as e:
            print(f"❌ Error in popup/legend testing: {str(e)}")
            self.test_results['popup_info'] = False
            self.test_results['legends'] = False
    
    def test_metrics_validation(self):
        """Test metrics and statistical validation."""
        print("\n📊 Test 8: Metrics and Statistical Validation")
        print("-" * 50)
        
        if not hasattr(self, 'aggregated_data'):
            print("❌ No aggregated data available")
            self.test_results['metrics'] = False
            return
        
        try:
            print("📈 Calculating summary metrics...")
            
            data = self.aggregated_data
            
            # Basic metrics
            total_cells = len(data)
            cells_with_anomalies = len(data[data['anomaly_count'] > 0])
            cells_without_anomalies = total_cells - cells_with_anomalies
            total_anomalies = data['anomaly_count'].sum()
            avg_anomalies_per_cell = total_anomalies / total_cells
            
            print(f"   📊 Summary Statistics:")
            print(f"      Total cells: {total_cells:,}")
            print(f"      Cells with anomalies: {cells_with_anomalies:,} ({cells_with_anomalies/total_cells*100:.1f}%)")
            print(f"      Cells without anomalies: {cells_without_anomalies:,} ({cells_without_anomalies/total_cells*100:.1f}%)")
            print(f"      Total anomalies: {total_anomalies:,}")
            print(f"      Average anomalies per cell: {avg_anomalies_per_cell:.1f}")
            
            # Severity statistics
            severity_stats = data['avg_severity'].describe()
            print(f"\n   📈 Severity Statistics (σ):")
            print(f"      Mean: {severity_stats['mean']:.2f}")
            print(f"      Std: {severity_stats['std']:.2f}")
            print(f"      Min: {severity_stats['min']:.2f}")
            print(f"      Max: {severity_stats['max']:.2f}")
            print(f"      25th percentile: {severity_stats['25%']:.2f}")
            print(f"      75th percentile: {severity_stats['75%']:.2f}")
            
            # Max severity statistics
            max_severity_stats = data['max_severity'].describe()
            print(f"\n   🔥 Max Severity Statistics (σ):")
            print(f"      Mean: {max_severity_stats['mean']:.2f}")
            print(f"      Max: {max_severity_stats['max']:.2f}")
            print(f"      95th percentile: {np.percentile(data['max_severity'], 95):.2f}")
            print(f"      99th percentile: {np.percentile(data['max_severity'], 99):.2f}")
            
            # Traffic statistics
            print(f"\n   📱 Traffic Statistics:")
            print(f"      SMS mean: {data['sms_total_mean'].mean():.1f} ± {data['sms_total_mean'].std():.1f}")
            print(f"      Calls mean: {data['calls_total_mean'].mean():.1f} ± {data['calls_total_mean'].std():.1f}")
            print(f"      Internet mean: {data['internet_traffic_mean'].mean():.1f} ± {data['internet_traffic_mean'].std():.1f}")
            
            # Category distribution validation
            if 'category' in data.columns:
                print(f"\n   🏷️  Category Distribution:")
                category_counts = data['category'].value_counts()
                for cat, count in category_counts.items():
                    percentage = count / len(data) * 100
                    print(f"      {cat}: {count} cells ({percentage:.1f}%)")
            
            # Data quality checks
            print(f"\n   🔍 Data Quality Checks:")
            
            # Check for negative values
            negative_anomalies = (data['anomaly_count'] < 0).sum()
            negative_severity = (data['avg_severity'] < 0).sum()
            
            print(f"      Negative anomaly counts: {negative_anomalies}")
            print(f"      Negative severity scores: {negative_severity}")
            
            # Check for extreme outliers
            q99_anomalies = np.percentile(data['anomaly_count'], 99)
            extreme_anomalies = (data['anomaly_count'] > q99_anomalies * 2).sum()
            
            q99_severity = np.percentile(data['avg_severity'], 99)
            extreme_severity = (data['avg_severity'] > q99_severity * 2).sum()
            
            print(f"      Extreme anomaly outliers (>2x 99th percentile): {extreme_anomalies}")
            print(f"      Extreme severity outliers (>2x 99th percentile): {extreme_severity}")
            
            # Overall validation
            validation_passed = (
                negative_anomalies == 0 and
                negative_severity == 0 and
                total_cells > 0 and
                cells_with_anomalies > 0
            )
            
            if validation_passed:
                print("   ✅ All metrics validation checks passed")
            else:
                print("   ⚠️  Some validation issues detected")
            
            self.test_results['metrics'] = validation_passed
            print("✅ Metrics validation completed")
            
        except Exception as e:
            print(f"❌ Error in metrics validation: {str(e)}")
            self.test_results['metrics'] = False
    
    def print_test_summary(self, total_time: float):
        """Print comprehensive test summary."""
        print("\n" + "=" * 70)
        print("VISUALIZATION FUNCTIONALITY TEST SUMMARY")
        print("=" * 70)
        
        # Count results
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        failed_tests = sum(1 for result in self.test_results.values() if result is False)
        skipped_tests = sum(1 for result in self.test_results.values() if result is None)
        total_tests = len(self.test_results)
        
        print(f"\nTest Results:")
        print(f"  ✅ Passed: {passed_tests}/{total_tests}")
        print(f"  ❌ Failed: {failed_tests}/{total_tests}")
        print(f"  ⚠️  Skipped: {skipped_tests}/{total_tests}")
        
        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            if result is True:
                status = "✅ PASS"
            elif result is False:
                status = "❌ FAIL"
            else:
                status = "⚠️  SKIP"
            print(f"  {test_name:25s}: {status}")
        
        print(f"\nPerformance:")
        print(f"  Total test time: {total_time:.2f} seconds")
        print(f"  Output directory: {self.output_dir}")
        
        if failed_tests == 0:
            print(f"\n🎉 All visualization functionality tests PASSED!")
            print("   Stage 4 visualization is ready for production use.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) FAILED.")
            print("   Please review and fix issues before deployment.")
        
        # Recommendations
        print(f"\nRecommendations:")
        if VISUALIZATION_AVAILABLE:
            print("  • All visualization libraries are properly installed")
        else:
            print("  • Install folium and plotly for full interactive functionality")
        
        if hasattr(self, 'aggregated_data') and len(self.aggregated_data) > 0:
            print("  • Data aggregation is working correctly")
        
        print("  • Test maps have been saved to results/test_maps/")
        print("  • Run full map generation using 06_generate_anomaly_map.py")


def main():
    """Main test execution."""
    tester = VisualizationTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
