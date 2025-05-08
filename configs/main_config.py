
parameters = dict(
    # Adjust if needed
    plot_centre=None,  # [X, Y] Coordinates of the plot centre (metres). If "None", plot_centre is computed based on the point cloud bounding box.
    # Circular Plot options - Leave at 0 if not using.
    plot_radius=0,  # If 0 m, the plot is not cropped. Otherwise, the plot is cylindrically cropped from the plot centre with plot_radius + plot_radius_buffer.
    plot_radius_buffer=0,  # See README. If non-zero, this is used for "Tree Aware Plot Cropping Mode".
    # Set these appropriately for your hardware.
    batch_size=2,  # You will get CUDA errors if this is too high, as you will run out of VRAM. This won't be an issue if running on CPU only. Must be >= 2.
    num_cpu_cores=0,  # Number of CPU cores you want to use. If you run out of RAM, lower this. 0 means ALL cores.
    use_CPU_only=False,  # Set to True if you do not have an Nvidia GPU, or if you don't have enough vRAM.
    # Optional settings - Generally leave as they are.
    slice_thickness=0.15,  # If your point cloud resolution is a bit low (and only if the stem segmentation is still reasonably accurate), try increasing this to 0.2.
    # If your point cloud is really dense, you may get away with 0.1.
    slice_increment=0.05,  # The smaller this is, the better your results will be, however, this increases the run time.
    sort_stems=1,  # If you don't need the sorted stem points, turning this off speeds things up.
    # Veg sorting is required for tree height measurement, but stem sorting isn't necessary for standard use.
    height_percentile=100,  # If the data contains noise above the canopy, you may wish to set this to the 98th percentile of height, otherwise leave it at 100.
    tree_base_cutoff_height=5,  # A tree must have a cylinder measurement below this height above the DTM to be kept. This filters unsorted branches from being called individual trees.
    generate_output_point_cloud=1,  # Turn on if you would like a semantic and instance segmented point cloud. This mode will override the "sort_stems" setting if on.
    # If you activate "tree aware plot cropping mode", this function will use it.
    ground_veg_cutoff_height=3,  # Any vegetation points below this height are considered to be understory and are not assigned to individual trees.
    veg_sorting_range=1.5,  # Vegetation points can be, at most, this far away from a cylinder horizontally to be matched to a particular tree.
    stem_sorting_range=1,  # Stem points can be, at most, this far away from a cylinder in 3D to be matched to a particular tree.
    taper_measurement_height_min=0,  # Lowest height to measure diameter for taper output.
    taper_measurement_height_max=30,  # Highest height to measure diameter for taper output.
    taper_measurement_height_increment=0.2,  # diameter measurement increment.
    taper_slice_thickness=0.4,  # Cylinder measurements within +/- 0.5*taper_slice_thickness are used for taper measurement at a given height. The largest diameter is used.
    delete_working_directory=True,  # Generally leave this on. Deletes the files used for segmentation after segmentation is finished.
    # You may wish to turn it off if you want to re-run/modify the segmentation code so you don't need to run pre-processing every time.
    minimise_output_size_mode=0,  # Will delete a number of non-essential outputs to reduce storage use.
)
other_parameters = dict(
    model_filename="model.pth",
    box_dimensions=[6, 6, 6],  # Dimensions of the sliding box used for semantic segmentation.
    box_overlap=[0.5, 0.5, 0.5],  # Overlap of the sliding box used for semantic segmentation.
    min_points_per_box=1000,  # Minimum number of points for input to the model. Too few points and it becomes near impossible to accurately label them (though assuming vegetation class is the safest bet here).
    max_points_per_box=20000,  # Maximum number of points for input to the model. The model may tolerate higher numbers if you decrease the batch size accordingly (to fit on the GPU), but this is not tested.
    noise_class=0,  # Don't change
    terrain_class=1,  # Don't change
    vegetation_class=2,  # Don't change
    cwd_class=3,  # Don't change
    stem_class=4,  # Don't change
    grid_resolution=0.5,  # Resolution of the DTM.
    vegetation_coverage_resolution=0.2,
    num_neighbours=5,
    sorting_search_angle=20,
    sorting_search_radius=1,
    sorting_angle_tolerance=90,
    max_search_radius=3,
    max_search_angle=30,
    min_cluster_size=30,  # Used for HDBSCAN clustering step. Recommend not changing for general use.
    cleaned_measurement_radius=0.2,  # During cleaning, this w
    subsample=0,  # Generally leave this on, but you can turn off subsampling.
    subsampling_min_spacing=0.01,  # The point cloud will be subsampled such that the closest any 2 points can be is 0.01 m.
    minimum_CCI=0.3,  # Minimum valid Circuferential Completeness Index (CCI) for non-interpolated circle/cylinder fitting. Any measurements with CCI below this are deleted.
    min_tree_cyls=10,  # Deletes any trees with fewer than 10 cylinders (before the cylinder interpolation step).
)  # Very ugly hack that can sometimes be useful on point clouds which are on the borderline of having not enough points to be functional with FSCT. Set to a positive integer. Point cloud will be copied this many times (with noise added) to artificially increase point density giving the segmentation model more points.
