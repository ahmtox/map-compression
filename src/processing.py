"""Preprocessing and data handling functions for map compression."""
import os
import numpy as np
from PIL import Image
import requests

def load_and_split_layers(image_path):
    """Split an image into semantic or color-plane layers"""
    img = Image.open(image_path)
    
    # Check image mode to handle both RGB and single-band images
    if img.mode == 'RGB':
        # For RGB images, do color-plane decomposition
        r, g, b = img.split()
        return {
            'R': np.array(r),
            'G': np.array(g),
            'B': np.array(b)
        }
    else:
        # For single-band images (grayscale/binary)
        return {
            'Gray': np.array(img)
        }

def block_decomposition(layer, block_size=128):
    """Decompose a layer into blocks"""
    height, width = layer.shape
    blocks = {}
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            h = min(block_size, height - y)
            w = min(block_size, width - x)
            block = layer[y:y+h, x:x+w]
            # Pad if needed to reach full block size
            if h < block_size or w < block_size:
                padded_block = np.zeros((block_size, block_size), dtype=layer.dtype)
                padded_block[:h, :w] = block
                block = padded_block
            
            blocks[(y//block_size, x//block_size)] = block
    
    return blocks

def download_osm_extract(region_name, output_dir='data/raw'):
    """Download OSM extract from Geofabrik"""
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://download.geofabrik.de/"
    # For example, using a small city extract
    url = f"{base_url}europe/germany/berlin-latest.osm.pbf"
    output_path = f"{output_dir}/{region_name}.osm.pbf"
    
    print(f"Downloading {url} to {output_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return output_path

def prepare_additional_maps():
    """Acquire additional map types for testing"""
    print("Processing additional maps...")
    
    # Check if data exists
    if not os.path.exists("data/raw/suburban.osm.pbf"):
        print("Suburban data not found. Please download manually.")
        return []

    # Create the processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    try:
        # Convert OSM to GeoPackage
        os.system("ogr2ogr -f GPKG data/processed/suburban.gpkg data/raw/suburban.osm.pbf")
        
        # Extract buildings and roads
        os.system('ogr2ogr -f GPKG data/processed/suburban_buildings.gpkg data/processed/suburban.gpkg -sql "SELECT * FROM multipolygons WHERE building IS NOT NULL"')
        os.system('ogr2ogr -f GPKG data/processed/suburban_roads.gpkg data/processed/suburban.gpkg -sql "SELECT * FROM lines WHERE highway IS NOT NULL"')
        
        # Rasterize
        os.system("gdal_rasterize -of GTiff -a_srs EPSG:3857 -ts 4096 4096 -burn 1 data/processed/suburban_buildings.gpkg data/processed/suburban_buildings.tif")
        os.system("gdal_rasterize -of GTiff -a_srs EPSG:3857 -ts 4096 4096 -burn 1 data/processed/suburban_roads.gpkg data/processed/suburban_roads.tif")
        
        # Convert to PNG
        os.system("gdal_translate -of PNG data/processed/suburban_buildings.tif data/processed/suburban_buildings.png")
        os.system("gdal_translate -of PNG data/processed/suburban_roads.tif data/processed/suburban_roads.png")
        
        # Create combined image
        roads = np.array(Image.open('data/processed/suburban_roads.png'))
        buildings = np.array(Image.open('data/processed/suburban_buildings.png'))
        
        combined = np.zeros((roads.shape[0], roads.shape[1], 3), dtype=np.uint8)
        combined[:,:,0] = (buildings > 0) * 255  # Red channel
        combined[:,:,1] = (roads > 0) * 255      # Green channel
        
        img = Image.fromarray(combined)
        img.save('data/processed/suburban_combined.png')
        
        print("Additional maps processed successfully")
        return [
            "data/processed/suburban_roads.png",
            "data/processed/suburban_buildings.png",
            "data/processed/suburban_combined.png"
        ]
    except Exception as e:
        print(f"Error processing additional maps: {e}")
        return []