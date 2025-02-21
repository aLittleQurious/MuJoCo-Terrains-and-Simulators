import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

# Generate a random height map
terrain = np.random.uniform(0, 255, (512, 512)).astype(np.uint8)

# Smooth the height map
terrain = gaussian_filter(terrain, sigma=20)

# Scale the values to ensure reasonable height variations
terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100

# Save as grayscale PNG
img = Image.fromarray(terrain.astype(np.uint8))
img.save("rocky_terrain.png")
