# This python file will extract cloud datapoints from 'extract_cloud.png'
from PIL import Image
import numpy as np
import csv

# 1) Load the image & get dimensions
img = Image.open('images/extract_cloud.png')
width, height = img.size

# 2) Turn it into a numpy array
arr = np.array(img)  # shape (H, W, 3)

# 3) Build a boolean mask of "white" pixels (R,G,B >= threshold)
threshold = 240
white_mask = np.all(arr >= threshold, axis=2)

# 4) List all white-pixel coords (row-major order)
ys, xs = np.where(white_mask)
coords = list(zip(xs, ys))

# 5) Flip Y so origin is bottom-left
cloud_points = [(x, height - y) for x, y in coords]

# 6) Down‐sample to at most 15000 points, evenly spaced
max_points = 50000
total_pts  = len(cloud_points)
if total_pts > max_points:
    # pick indices evenly across the full list
    indices = np.linspace(0, total_pts - 1, max_points, dtype=int)
    cloud_points = [cloud_points[i] for i in indices]

# 7) Write out CSV
with open('data/cloud.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    writer.writerows(cloud_points)

print(f"Wrote {len(cloud_points)} cloud data points to cloud.csv")
