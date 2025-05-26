#!/usr/bin/env python3
from PIL import Image
import numpy as np
import csv

# 1) Load the image & get dimensions
img = Image.open('images/extract_snow.png')
width, height = img.size

# 2) Turn it into a numpy array
arr = np.array(img)  # shape (H, W, 3)

# 3) Build a boolean mask of "white" pixels (R,G,B >= threshold)
threshold = 240
white_mask = np.all(arr >= threshold, axis=2)

# 4) List all white-pixel coords
ys, xs = np.where(white_mask)
coords = list(zip(xs, ys))

# 5) Randomly pick N of them, but don't exceed population
num_points = 10000
total_pts  = len(coords)
if num_points > total_pts:
    num_points = total_pts

choice_idx = np.random.choice(total_pts, num_points, replace=False)
selected   = [coords[i] for i in choice_idx]

# 6) Flip Y so origin is bottom-left
white_points = [(x, height - y) for x, y in selected]

# 7) Write out CSV
with open('data/snow.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x','y'])
    writer.writerows(white_points)

print(f"Wrote {len(white_points)} white-pixel positions to snow.csv")
