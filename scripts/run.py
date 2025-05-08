from run_tools import FSCT, file_mode
import sys
import os

# Add the parent directory of 'configs' to sys.path
sys.path.append(os.path.abspath(os.path.join('../')))

from configs.main_config import parameters, other_parameters

if __name__ == "__main__":
    """Choose one of the following or modify as needed.
    Directory mode will find all .las files within a directory and sub directories but will ignore any .las files in
    folders with "FSCT_output" in their names.

    File mode will allow you to select multiple .las files within a directory.

    Alternatively, you can just list the point cloud file paths.

    If you have multiple point clouds and wish to enter plot coords for each, have a look at "run_with_multiple_plot_centres.py"
    """

    point_clouds_to_process = file_mode()

    for point_cloud_filename in point_clouds_to_process:
        point_cloud_filename=point_cloud_filename,
        parameters["point_cloud_filename"] = point_cloud_filename[0]
        parameters.update(other_parameters)
        FSCT(
            parameters=parameters,
            # Set below to 0 or 1 (or True/False). Each step requires the previous step to have been run already.
            # For standard use, just leave them all set to 1 except "clean_up_files".
            preprocess=1,  # Preparation for semantic segmentation.
            segmentation=1,  # Deep learning based semantic segmentation of the point cloud.
            postprocessing=1,  # Creates the DTM and applies some simple rules to clean up the segmented point cloud.
        ) 
