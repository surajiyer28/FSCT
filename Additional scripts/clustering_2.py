import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# --- STEP 1: Cylinder generation ---

def generate_oriented_cylinder_cluster(center, direction, radius, height, n_points, cylinder_noise=0.2):
    direction = direction / np.linalg.norm(direction)

    t_vals = np.random.uniform(0, height, n_points)
    axis_points = center + np.outer(t_vals, direction)

    if np.allclose(direction, [0, 0, 1]):
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.cross(direction, [0, 0, 1])
        v1 /= np.linalg.norm(v1)
    v2 = np.cross(direction, v1)

    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.random.uniform(0, radius + cylinder_noise, n_points)
    radial_offsets = (np.outer(radii * np.cos(angles), v1) +
                      np.outer(radii * np.sin(angles), v2))

    return axis_points + radial_offsets

def generate_5_cylindrical_clusters_with_noise(points_per_cluster=150, 
                                                cylinder_noise=0.2, 
                                                background_noise_points=500,
                                                radius=1.0):
    height = 75 * radius
    clusters = []

    for _ in range(5):
        center = np.random.uniform(-20, 20, size=3)
        direction = np.random.normal(size=3)
        cluster = generate_oriented_cylinder_cluster(center, direction, radius, height, points_per_cluster, cylinder_noise)
        clusters.append(cluster)

    all_cylinder_points = np.vstack(clusters)
    background_noise = np.random.uniform(-30, 30, size=(background_noise_points, 3))
    all_points = np.vstack((all_cylinder_points, background_noise))

    return all_points

# --- STEP 2: Clustering + Initial Merging ---

def slice_points(points, slice_thickness=5.0):
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])

    slices = []
    z_slices = np.arange(min_z, max_z, slice_thickness)

    for z_start in z_slices:
        z_end = z_start + slice_thickness
        mask = (points[:, 2] >= z_start) & (points[:, 2] < z_end)
        slices.append(points[mask])
    
    return slices

def cluster_in_slices(slices, eps=2.5, min_samples=5):
    slice_clusters = []
    for slice_points in slices:
        if len(slice_points) < min_samples:
            continue
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(slice_points)
        labels = clustering.labels_
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = slice_points[labels == label]
            slice_clusters.append(cluster)
    return slice_clusters

def fit_axis_pca(cluster_points):
    pca = PCA(n_components=1)
    pca.fit(cluster_points)
    axis_dir = pca.components_[0]
    centroid = cluster_points.mean(axis=0)
    return centroid, axis_dir

def fit_cylinder(cluster_points):
    centroid, axis_dir = fit_axis_pca(cluster_points)

    diffs = cluster_points - centroid
    projections = np.dot(diffs, axis_dir)[:, None] * axis_dir
    radial_vectors = diffs - projections
    radial_distances = np.linalg.norm(radial_vectors, axis=1)

    radius = np.median(radial_distances)

    return centroid, axis_dir, radius

# --- STEP 3: Adaptive Cylinder Merging with Natural Clustering ---

def distance_between_axes(c1, ax1, c2, ax2):
    """Compute the minimum distance between two infinite lines in 3D space
    defined by point c1 and direction ax1, and point c2 and direction ax2"""
    # Cross product of the direction vectors
    cross = np.cross(ax1, ax2)
    cross_norm = np.linalg.norm(cross)
    
    # If axes are parallel or nearly parallel
    if cross_norm < 1e-10:
        # Project c2-c1 onto ax1 to get the parallel distance
        # The perpendicular distance is ||(c2-c1) - proj||
        proj = np.dot(c2 - c1, ax1) * ax1
        return np.linalg.norm((c2 - c1) - proj)
    
    # For non-parallel lines, the distance is:
    # |(c2-c1)·(ax1×ax2)| / |ax1×ax2|
    return abs(np.dot(c2 - c1, cross)) / cross_norm

def adaptive_cylinder_merging(initial_clusters):
    """
    Merge cylinders using adaptive clustering that naturally discovers the 
    underlying groups without forcing a target number.
    
    Parameters:
    - initial_clusters: List of point arrays from initial clustering
    
    Returns:
    - List of merged cylinder clusters
    """
    # Skip small clusters & fit cylinders to remaining ones
    fitted_cylinders = []
    for cluster in initial_clusters:
        if len(cluster) < 10:  # Skip tiny clusters
            continue
        centroid, axis_dir, radius = fit_cylinder(cluster)
        fitted_cylinders.append((centroid, axis_dir, radius, cluster))
    
    # Handle edge cases
    if len(fitted_cylinders) <= 1:
        return [fc[3] for fc in fitted_cylinders]
    
    n_cylinders = len(fitted_cylinders)
    
    # Build a comprehensive similarity matrix
    similarity_matrix = np.ones((n_cylinders, n_cylinders))  # Diagonal = 1 (identical)
    
    for i in range(n_cylinders):
        ci, axi, ri, _ = fitted_cylinders[i]
        for j in range(i+1, n_cylinders):
            cj, axj, rj, _ = fitted_cylinders[j]
            
            # 1. Axis alignment score (1 = perfect alignment, 0 = perpendicular)
            cos_angle = np.abs(np.dot(axi, axj))
            alignment_score = cos_angle ** 2  # Square to penalize misalignment more
            
            # 2. Axis proximity - minimum distance between the two axis lines
            axis_distance = distance_between_axes(ci, axi, cj, axj)
            avg_radius = (ri + rj) / 2
            
            # Normalize by average radius (1 = touching, <1 = overlapping, >1 = separated)
            normalized_distance = axis_distance / avg_radius
            proximity_score = np.exp(-normalized_distance / 2)  # Exponential falloff
            
            # 3. Point density continuity - check if the cylinders form a continuous structure
            # Compute extents along axis for both cylinders
            pi = fitted_cylinders[i][3]  # points of cylinder i
            pj = fitted_cylinders[j][3]  # points of cylinder j
            
            # Project points onto respective axes
            proj_i = np.dot(pi - ci, axi)
            proj_j = np.dot(pj - cj, axj)
            
            # Get min/max projections to find endpoints
            min_i, max_i = np.min(proj_i), np.max(proj_i)
            min_j, max_j = np.min(proj_j), np.max(proj_j)
            
            # Normalize axes directions for consistent endpoint calculations
            if np.dot(axi, axj) < 0:  # If axes point in opposite directions
                axj = -axj  # Flip one to make them consistent
                min_j, max_j = -max_j, -min_j  # Flip extents too
            
            # Calculate "endmost" points of each cylinder
            p1_start = ci + min_i * axi
            p1_end = ci + max_i * axi
            p2_start = cj + min_j * axj
            p2_end = cj + max_j * axj
            
            # Find smallest distance between any endpoints
            endpoint_distances = [
                np.linalg.norm(p1_start - p2_start),
                np.linalg.norm(p1_start - p2_end),
                np.linalg.norm(p1_end - p2_start),
                np.linalg.norm(p1_end - p2_end)
            ]
            
            min_endpoint_dist = min(endpoint_distances)
            continuity_score = np.exp(-min_endpoint_dist / (10 * avg_radius))
            
            # Combined score with weighting
            # High weight on alignment, medium on proximity, lower on continuity
            combined_score = (
                0.6 * alignment_score + 
                0.3 * proximity_score +
                0.1 * continuity_score
            )
            
            # Very poor alignment should be a hard cutoff
            if alignment_score < 0.7:  # cos² < 0.7 means angle > ~33°
                combined_score *= 0.1  # Severely penalize poor alignment
                
            # Too far apart should also be a hard cutoff
            if normalized_distance > 5:  # More than 5x the radius apart
                combined_score *= 0.1  # Severely penalize too distant cylinders
            
            similarity_matrix[i, j] = similarity_matrix[j, i] = combined_score
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1.0 - similarity_matrix
    
    # Convert to condensed form for scipy hierarchy functions
    condensed_dist = squareform(distance_matrix)
    
    # Apply hierarchical clustering with automatic cutoff determination
    Z = linkage(condensed_dist, method='average')
    
    # Determine optimal number of clusters using the "elbow method"
    # We look at the distances at which merges occur and find a significant jump
    last_n_merges = 10  # Look at the last n merges
    if len(Z) > last_n_merges:
        merge_dists = Z[-last_n_merges:, 2]  # Heights of last merges
        diffs = np.diff(merge_dists)
        
        # Find the largest "jump" in merge distances
        if len(diffs) > 0:
            largest_jump_idx = np.argmax(diffs)
            # Use the distance right after the largest jump as threshold
            threshold = merge_dists[largest_jump_idx + 1]
        else:
            # Fallback if not enough data points
            threshold = 0.5
    else:
        # Fallback for small datasets
        threshold = 0.5
    
    # Ensure threshold is reasonable (neither too small nor too large)
    threshold = max(0.3, min(threshold, 0.8))
    
    # Cluster using the dynamic threshold
    labels = fcluster(Z, threshold, criterion='distance') - 1  # 0-based indexing
    
    # Merge clusters based on labels
    final_clusters = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        merged_points = []
        for idx in indices:
            merged_points.append(fitted_cylinders[idx][3])
        
        final_clusters.append(np.vstack(merged_points))
    
    return final_clusters

# --- STEP 4: Visualization ---

def visualize_generated_points(points):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='black', alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Generated Points (Cylinders + Noise)')
    plt.show()

def visualize_final_clusters(clusters):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.get_cmap('tab10', len(clusters))

    for i, cluster in enumerate(clusters):
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=2, color=colors(i))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Final Merged Cylinders (Each with same color)')
    plt.show()

def visualize_cylinder_fits(clusters):
    """Visualize the merged clusters with their fitted cylinders"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.get_cmap('tab10', len(clusters))
    
    for i, cluster_points in enumerate(clusters):
        # Plot points
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                  s=2, color=colors(i), alpha=0.7)
        
        # Fit cylinder and show axis
        centroid, axis_dir, radius = fit_cylinder(cluster_points)
        
        # Plot axis line (extend in both directions)
        line_length = 50  # Adjust based on your data scale
        endpoint1 = centroid + line_length * axis_dir
        endpoint2 = centroid - line_length * axis_dir
        ax.plot([endpoint1[0], endpoint2[0]], 
                [endpoint1[1], endpoint2[1]], 
                [endpoint1[2], endpoint2[2]], 
                color=colors(i), linewidth=2)
        
        # Plot centroid
        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], 
                  color=colors(i), s=50, marker='o', edgecolor='black')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Merged Cylinders with Fitted Axes')
    plt.show()

# --- FULL PIPELINE ---

def full_pipeline(points):
    slices = slice_points(points, slice_thickness=5.0)
    initial_clusters = cluster_in_slices(slices, eps=2.5, min_samples=5)
    
    # Use the adaptive cylinder merging
    merged_cylinders = adaptive_cylinder_merging(initial_clusters)
    
    return merged_cylinders

# --- RUN EVERYTHING ---

# Step 1: Generate Data
points = generate_5_cylindrical_clusters_with_noise(points_per_cluster=150, 
                                                    cylinder_noise=0.2, 
                                                    background_noise_points=500,
                                                    radius=1.0)

# Step 2: Visualize raw points
visualize_generated_points(points)

# Step 3: Full clustering pipeline
final_clusters = full_pipeline(points)

# Step 4: Visualize final nicely colored merged cylinders
visualize_final_clusters(final_clusters)

# Step 5: Visualize with cylinder axes
visualize_cylinder_fits(final_clusters)

print(f"Number of final merged cylinders detected: {len(final_clusters)}")