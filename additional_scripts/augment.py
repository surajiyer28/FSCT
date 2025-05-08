import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from multiprocessing import get_context
from copy import deepcopy
import laspy

def subsample_single(args):
    point_data, min_distance = args

    neighbors_model = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(point_data[:, :3])
    distances, indices = neighbors_model.kneighbors(point_data[:, :3])
    points_to_keep = point_data[distances[:, 1] >= min_distance]
    
    # Identify points that are too close and need checking
    close_points_mask = [distances[:, 1] < min_distance][0]
    lower_z_mask = [point_data[indices[:, 0], 2] < point_data[indices[:, 1], 2]][0]
    points_to_check = point_data[np.logical_and(close_points_mask, lower_z_mask)]

    while np.shape(points_to_check)[0] > 1:
        neighbors_model = NearestNeighbors(n_neighbors=2, algorithm="kd_tree", metric="euclidean").fit(points_to_check[:, :3])
        distances, indices = neighbors_model.kneighbors(points_to_check[:, :3])
        points_to_keep = np.vstack((points_to_keep, points_to_check[distances[:, 1] >= min_distance, :]))
        
        close_points_mask = [distances[:, 1] < min_distance][0]
        lower_z_mask = [points_to_check[indices[:, 0], 2] < points_to_check[indices[:, 1], 2]][0]
        points_to_check = points_to_check[np.logical_and(close_points_mask, lower_z_mask)]
    
    return points_to_keep

def subsample_parallel(point_cloud, min_distance=0.01, num_cores=4):
    
    # subsampling to ensure minimum distance
    print("Original number of points: ", point_cloud.shape[0])

    if num_cores > 1:
        num_slices = num_cores
        x_min = np.min(point_cloud[:, 0])
        x_max = np.max(point_cloud[:, 0])
        x_range = x_max - x_min
        slice_list = []
        kdtree = spatial.cKDTree(np.atleast_2d(point_cloud[:, 0]).T, leafsize=10000)
        
        for i in range(num_slices):
            slice_min_bound = x_min + i * (x_range / num_slices)
            point_indices = kdtree.query_ball_point(np.array([slice_min_bound]), r=x_range / num_slices)
            point_cloud_slice = point_cloud[point_indices]
            slice_list.append([point_cloud_slice, min_distance])

        point_cloud = np.zeros((0, point_cloud.shape[1]))
        with get_context("spawn").Pool(processes=num_cores) as pool:
            for processed_slice in pool.imap_unordered(subsample_single, slice_list):
                point_cloud = np.vstack((point_cloud, processed_slice))
    else:
        point_cloud = subsample_single([point_cloud, min_distance])

    print("Subsampled number of points:", point_cloud.shape[0])
    return point_cloud

def duplicate_cloud(point_cloud, duplicate_count, min_distance, num_cores):
    print("Original point cloud shape: ",point_cloud.shape)
    point_cloud_original = deepcopy(point_cloud)
    
    for i in range(duplicate_count):
        print("Duplicate Count: ", i+1)
        duplicate = deepcopy(point_cloud_original)

        duplicate[:, :3] = duplicate[:, :3] + np.hstack(
            (
                np.random.normal(-0.025, 0.025, size=(duplicate.shape[0], 1)),
                np.random.normal(-0.025, 0.025, size=(duplicate.shape[0], 1)),
                np.random.normal(-0.025, 0.025, size=(duplicate.shape[0], 1)),
            )
        )
        point_cloud = np.vstack((point_cloud, duplicate))
        point_cloud = subsample_parallel(point_cloud, min_distance, num_cores)
    
    print("Augmented point cloud shape:", point_cloud.shape)
    return point_cloud

if __name__ == "__main__":
    input_file = "pt000002new.las"
    output_file = "augmented.las"
    duplicate_count = 4
    min_distance = 0.01
    cores = 4
    
    # start_time = time.time()
    
    # loading input point cloud
    print(f"Loading file {input_file}...")
    input_las = laspy.read(input_file)
    
    
    points = np.column_stack((input_las.x, input_las.y, input_las.z))
    
    
    if hasattr(input_las, 'red') and hasattr(input_las, 'green') and hasattr(input_las, 'blue'):
        rgb = np.column_stack((input_las.red, input_las.green, input_las.blue))
        points = np.column_stack((points, rgb))
    
    print(f"Loaded point cloud with {len(points)} points")
    
    # main function for duplication
    augmented_pointcloud = duplicate_cloud(points, duplicate_count, min_distance, cores)
    
   
    # saving augmented point cloud
    output_las = laspy.create(
        point_format=input_las.header.point_format,
        file_version=input_las.header.version
    )

    output_las.x = augmented_pointcloud[:, 0]
    output_las.y = augmented_pointcloud[:, 1]
    output_las.z = augmented_pointcloud[:, 2]
    
  
    if augmented_pointcloud.shape[1] > 3 and hasattr(input_las, 'red'):
        output_las.red = augmented_pointcloud[:, 3].astype(np.uint16)
        output_las.green = augmented_pointcloud[:, 4].astype(np.uint16)
        output_las.blue = augmented_pointcloud[:, 5].astype(np.uint16)
    
    # saving augmented file
    output_las.write(output_file)
    
    print(f"Augmented point cloud saved to {output_file}")

