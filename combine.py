# Make an RGB image from your two layers
from PIL import Image
import numpy as np

# Load binary images 
roads = np.array(Image.open('data/processed/berlin_roads.png'))
buildings = np.array(Image.open('data/processed/berlin_buildings.png'))

# Create RGB image (red=buildings, green=roads, blue=empty)
combined = np.zeros((roads.shape[0], roads.shape[1], 3), dtype=np.uint8)
combined[:,:,0] = (buildings > 0) * 255  # Red channel
combined[:,:,1] = (roads > 0) * 255      # Green channel

# Save the combined image
img = Image.fromarray(combined)
img.save('data/processed/berlin_combined.png')