from preprocessing import Preprocessing
from inference import SemanticSegmentation
from post_segmentation_script import PostProcessing
import tkinter as tk
import tkinter.filedialog as fd
import os


def FSCT(
    parameters,
    preprocess=True,
    segmentation=True,
    postprocessing=True,
):
    print("Current point cloud being processed: ", parameters["point_cloud_filename"])
    if parameters["num_cpu_cores"] == 0:
        print("Using default number of CPU cores (all of them).")
        parameters["num_cpu_cores"] = os.cpu_count()
    print("Processing using ", parameters["num_cpu_cores"], "/", os.cpu_count(), " CPU cores.")

    if preprocess:
        preprocessing = Preprocessing(parameters)
        preprocessing.preprocess_point_cloud()
        del preprocessing

    if segmentation:
        sem_seg = SemanticSegmentation(parameters)
        sem_seg.inference()
        del sem_seg

    if postprocessing:
        post_processing = PostProcessing(parameters)
        post_processing.process_point_cloud()
        del post_processing


def file_mode():
    root = tk.Tk()
    point_clouds_to_process = fd.askopenfilenames(
        parent=root, title="Choose files", filetypes=[("LAS", "*.las"), ("LAZ", "*.laz"), ("CSV", "*.csv")]
    )
    root.destroy()
    return point_clouds_to_process
