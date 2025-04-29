import laspy
import numpy as np

# Load the LAS file
las = laspy.read("pt000002.las")

# Extract scaled point coordinates
x = las.x
y = las.y
z = las.z

# Get the bounding box dimensions
x_range = x.max() - x.min()
y_range = y.max() - y.min()
z_range = z.max() - z.min()

# Calculate area and volume
area_m2 = x_range * y_range  # in square meters
volume_m3 = area_m2 * z_range  # in cubic meters

# Total number of points
num_points = len(x)

# Compute densities
density_per_m2 = num_points / area_m2 if area_m2 != 0 else 0
density_per_m3 = num_points / volume_m3 if volume_m3 != 0 else 0

# Print the results
print(f"Total points: {num_points}")
print(f"XY area (m²): {area_m2:.2f}")
print(f"Volume (m³): {volume_m3:.2f}")
print(f"Point density per m²: {density_per_m2:.2f} pts/m²")
print(f"Point density per m³: {density_per_m3:.2f} pts/m³")
