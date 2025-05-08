import laspy
import numpy as np
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt

segmented_file = "../Data/segmented.las"
dtm = "../Data/DTM.las"

output_dir = "../Data"
os.makedirs(output_dir, exist_ok=True)

seg_las = laspy.read(segmented_file)
dtm_las = laspy.read(dtm)

height_threshold = 3  # threshold to control the height from dtm for classifying a point as stem or non-stem (in meters)
distance_threshold = 0.8  # to remove terrain points wrt to the dtm


seg_points = np.column_stack((seg_las.x, seg_las.y, seg_las.z))
seg_attrs = {}

for attr in ['red', 'green', 'blue', 'label']:
    seg_attrs[attr] = getattr(seg_las, attr)
    
dtm_points = np.column_stack((dtm_las.x, dtm_las.y, dtm_las.z))

print(f"Segmented file has {len(seg_points)} points")
print(f"DTM file has {len(dtm_points)} points")


# computing heights with respect to the DTM for each point based on the closest point in the dtm
dtm_tree = cKDTree(dtm_points[:, :2])

# finding the closest dtm point for each point in the point cloud using only the x-y coordinate
dist, indices = dtm_tree.query(seg_points[:, :2], k=1)

terrain_height = dtm_points[indices, 2] # z-values
height_above_dtm = seg_points[:, 2] - terrain_height

print("Heights calculated")


# separating the stem and non-stem points using the segmentation label and height from dtm

labels = seg_attrs['label']

# non-stem points: label=0,1,2 and below height_threshold
# stem points: label=3 and above height_threshold = 2.5
non_stem_mask = np.isin(labels, [0, 1, 2]) & (height_above_dtm <= height_threshold)
stem_mask = np.isin(labels, [3]) & (height_above_dtm > height_threshold)

# print(np.sum(stem_mask))
# print(np.sum(non_stem_mask))

# separating stem and non-stem points
stem_points = seg_points[stem_mask]
non_stem_points = seg_points[non_stem_mask]


stem_attrs = {}
for k, v in seg_attrs.items():
    stem_attrs[k] = v[stem_mask]

stem_path = os.path.join(output_dir, "stem_only.las")

stem_las = laspy.create(
    point_format=seg_las.header.point_format, 
    file_version=seg_las.header.version
)

stem_las.x = stem_points[:, 0]
stem_las.y = stem_points[:, 1]
stem_las.z = stem_points[:, 2]

for key in ['red', 'green', 'blue', 'label']:
    setattr(stem_las, key, stem_attrs[key])

stem_las.write(stem_path)

print(f"saved stem points in stem_only.las")

# Removing terrain points from the set of non-stem points

dtm_tree_3d = cKDTree(dtm_points)
distances, idx = dtm_tree_3d.query(non_stem_points, k=1)

non_terrain_mask = distances > distance_threshold # assuming non-terrain points would have a greater distance form the dtm
filtered_points = non_stem_points[non_terrain_mask]

# plotting filtered poitns
# x = filtered_points[:, 0]
# y = filtered_points[:, 1]

# indices = np.random.choice(len(x), 100000, replace=False)
# x = x[indices]
# y = y[indices]

# plt.figure(figsize=(10, 8))
# plt.scatter(x, y, s=0.3, c='blue', alpha=0.7)
# plt.show()


terrain_count = np.sum(~non_terrain_mask)
non_terrain_count = np.sum(non_terrain_mask)

print("Removed terrain points: ",terrain_count)
print("Count of non-terrain points: ", non_terrain_count)

filtered_attrs = {}
for k, v in seg_attrs.items():
    filtered_attrs[k] = v[non_stem_mask][non_terrain_mask]

non_stem_path = os.path.join(output_dir, "filtered_non_stem.las")

filtered_las = laspy.create(
    point_format=seg_las.header.point_format, 
    file_version=seg_las.header.version
)

filtered_las.x = filtered_points[:, 0]
filtered_las.y = filtered_points[:, 1]
filtered_las.z = filtered_points[:, 2]

for key in ['red', 'green', 'blue', 'label']:
    setattr(filtered_las, key, filtered_attrs[key])

filtered_las.write(non_stem_path)
print("saved non-stem points in filtered_non_stem.las")

