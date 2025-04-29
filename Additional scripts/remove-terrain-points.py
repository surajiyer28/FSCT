import laspy
import numpy as np
from scipy.spatial import cKDTree
import argparse
import os

def remove_dtm_points(complete_pc_path, dtm_path, output_path, distance_threshold=0.05):
    """
    Remove points from the complete point cloud that lie on the DTM.
    
    Parameters:
    -----------
    complete_pc_path : str
        Path to the complete point cloud file (.las)
    dtm_path : str
        Path to the DTM point cloud file (.las)
    output_path : str
        Path for the output point cloud file (.las)
    distance_threshold : float
        Maximum distance for a point to be considered lying on the DTM (in meters)
    """
    print(f"Loading complete point cloud: {complete_pc_path}")
    complete_pc = laspy.read(complete_pc_path)
    
    print(f"Loading DTM: {dtm_path}")
    dtm = laspy.read(dtm_path)
    
    # Get coordinates
    complete_points = np.vstack((complete_pc.x, complete_pc.y, complete_pc.z)).transpose()
    dtm_points = np.vstack((dtm.x, dtm.y, dtm.z)).transpose()
    
    print(f"Complete point cloud has {len(complete_points)} points")
    print(f"DTM has {len(dtm_points)} points")
    
    # Create KD-Tree for DTM points
    print("Building KD-Tree for DTM points...")
    dtm_tree = cKDTree(dtm_points)
    
    # Find distances to nearest DTM point
    print("Finding distances to nearest DTM points...")
    distances, _ = dtm_tree.query(complete_points, k=1)
    
    # Create mask for points NOT on the DTM
    non_dtm_mask = distances > distance_threshold
    
    # Count points to be kept
    num_points_to_keep = np.sum(non_dtm_mask)
    print(f"Filtering points (keeping {num_points_to_keep} of {len(complete_points)})...")
    
    # Create a new LAS file with the same point format, but correct point count
    header = laspy.LasHeader(point_format=complete_pc.header.point_format.id, version=complete_pc.header.version)
    output_las = laspy.LasData(header)
    
    # Create a new set of points with the correct length
    for dimension_name in complete_pc.point_format.dimension_names:
        dimension_data = getattr(complete_pc, dimension_name)[non_dtm_mask]
        setattr(output_las, dimension_name, dimension_data)
    
    # Save the filtered point cloud
    print(f"Saving filtered point cloud to {output_path}")
    output_las.write(output_path)
    
    print("Done!")
    print(f"Removed {len(complete_points) - num_points_to_keep} points ({100 * (1 - num_points_to_keep / len(complete_points)):.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove points from a point cloud that lie on the DTM")
    parser.add_argument("complete_pc", help="Path to the complete point cloud file (.las)")
    parser.add_argument("dtm", help="Path to the DTM point cloud file (.las)")
    parser.add_argument("output", help="Path for the output point cloud file (.las)")
    parser.add_argument("--threshold", type=float, default=0.05, 
                        help="Maximum distance for a point to be considered lying on the DTM (in meters), default=0.05")
    
    args = parser.parse_args()
    
    remove_dtm_points(args.complete_pc, args.dtm, args.output, args.threshold)