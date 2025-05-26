#!/usr/bin/env python3
from PIL import Image
import numpy as np
import csv
from matplotlib.path import Path

# 1) Load the image to get dimensions
img = Image.open('images/original.png')
width, height = img.size

# 2) Quadratic Bézier helper
def bezier(p0, p1, p2, n=200):
    t = np.linspace(0, 1, n)
    x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * p1[1] + t**2 * p2[1]
    return np.column_stack((x, y))

# 3) Build the forest/mountain boundary
curve1 = bezier((0, 856), (715, 700), (1158, 163), n=200)

# 4) Anchor points along the ridge
forest_pts = np.array([
    (552, 663),(621, 679),(532, 711),(660, 724),(552, 769),
    (739, 789),(658, 808),(849, 819),(795, 840),(1027, 861),
    (766, 895),(1254, 921),(934, 964),(1449,1024)
])

# 5) Find where the curve first hits the forest start
dists    = np.hypot(curve1[:,0] - forest_pts[0,0],
                    curve1[:,1] - forest_pts[0,1])
idx_min  = np.argmin(dists)
mount_seg = curve1[:idx_min+1]

# 6) Close the polygon with a straight "edge"
edge        = np.array([(1449,1024),(0,1024),(0,856)])
forest_poly = np.vstack((mount_seg, forest_pts, edge))
path        = Path(forest_poly)

# — compute bounding box of the forest polygon —
x_min, x_max = forest_poly[:,0].min(), forest_poly[:,0].max()
y_min, y_max = forest_poly[:,1].min(), forest_poly[:,1].max()

# 7) Sample uniformly inside that bounding box until we have enough trees
num_trees = 4000   # increase for higher density
trees     = []
while len(trees) < num_trees:
    x_rand = np.random.uniform(x_min, x_max)
    y_rand = np.random.uniform(y_min, y_max)
    if path.contains_point((x_rand, y_rand)):
        # flip Y so origin is bottom-left
        trees.append((x_rand, height - y_rand))

# 8) Write out CSV
with open('data/trees.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['x','y'])
    w.writerows(trees)

print(f"Wrote {len(trees)} tree positions to trees.csv")
