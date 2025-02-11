import numpy as np
from PIL import Image

# Generate random noise for terrain
terrain = np.random.uniform(0, 255, (512, 512)).astype(np.uint8)

# Smooth it with a Gaussian filter for a natural look
from scipy.ndimage import gaussian_filter
terrain = gaussian_filter(terrain, sigma=10)

# Save as PNG
img = Image.fromarray(terrain)
img.save("rocky_terrain.png")
