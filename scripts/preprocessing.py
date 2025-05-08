import os
import numpy as np
import pandas as pd
import time
import random
import threading
from tools import load_file, save_file, make_folder_structure, subsample_point_cloud

class Preprocessing:
    def __init__(self, config):
        self.start_time = time.time()
        self.config = config
        self.input_path = config["point_cloud_filename"]
        self.directory = os.path.dirname(os.path.abspath(self.input_path)) + "/"
        self.filename = os.path.basename(self.input_path)

        self.box_dims = np.array(config["box_dimensions"])
        self.box_overlap = np.array(config["box_overlap"])
        self.min_pts = config["min_points_per_box"]
        self.max_pts = config["max_points_per_box"]
        self.num_cores = config["num_cpu_cores"]

        self.output_dir, self.work_dir = make_folder_structure(self.directory + self.filename)

        self.point_cloud, _, self.num_points_orig = load_file(
            filename=os.path.join(self.directory, self.filename),
            plot_centre=config["plot_centre"],
            plot_radius=config["plot_radius"],
            plot_radius_buffer=config["plot_radius_buffer"],
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
            return_num_points=True,
        )

        self.num_points_trimmed = self.point_cloud.shape[0]

        if self.config["plot_centre"] is None:
            mins = np.min(self.point_cloud[:, :2], axis=0)
            maxes = np.max(self.point_cloud[:, :2], axis=0)
            self.config["plot_centre"] = (mins + maxes) / 2

        if config["subsample"]:
            self.point_cloud = subsample_point_cloud(
                self.point_cloud,
                config["subsampling_min_spacing"],
                self.num_cores,
            )

        self.num_points_subsampled = self.point_cloud.shape[0]

        save_file(
            os.path.join(self.output_dir, "working_point_cloud.las"),
            self.point_cloud,
            headers_of_interest=["x", "y", "z", "red", "green", "blue"],
        )

        self.point_cloud = self.point_cloud[:, :3]
        self.point_cloud[:, :2] -= self.config["plot_centre"]

    @staticmethod
    def save_boxes(point_cloud, box_size, min_pts, max_pts, save_path, id_start, box_centers):
        box_min = box_centers - 0.5 * box_size
        box_max = box_centers + 0.5 * box_size

        for i, center in enumerate(box_centers):
            mask = np.all(
                (point_cloud >= box_min[i]) & (point_cloud < box_max[i]),
                axis=1
            )
            box_points = point_cloud[mask]

            if box_points.shape[0] > min_pts:
                if box_points.shape[0] > max_pts:
                    selected_indices = np.random.choice(box_points.shape[0], max_pts, replace=False)
                    box_points = box_points[selected_indices]

                np.save(os.path.join(save_path, f"{str(id_start + i).zfill(7)}.npy"), box_points)

    def preprocess_point_cloud(self):
        print("Preprocessing point cloud...")

        min_bounds = np.min(self.point_cloud, axis=0)
        max_bounds = np.max(self.point_cloud, axis=0)
        ranges = max_bounds - min_bounds

        num_boxes = np.ceil(ranges / self.box_dims).astype(int)

        x_centers = np.linspace(
            min_bounds[0], min_bounds[0] + num_boxes[0] * self.box_dims[0],
            int(num_boxes[0] / (1 - self.box_overlap[0])) + 1
        )
        y_centers = np.linspace(
            min_bounds[1], min_bounds[1] + num_boxes[1] * self.box_dims[1],
            int(num_boxes[1] / (1 - self.box_overlap[1])) + 1
        )
        z_centers = np.linspace(
            min_bounds[2], min_bounds[2] + num_boxes[2] * self.box_dims[2],
            int(num_boxes[2] / (1 - self.box_overlap[2])) + 1
        )

        box_centers = np.stack(np.meshgrid(x_centers, y_centers, z_centers), axis=-1).reshape(-1, 3)
        box_splits = np.array_split(box_centers, self.num_cores)

        threads = []
        id_offset = 0
        for split in box_splits:
            thread = threading.Thread(
                target=self.save_boxes,
                args=(self.point_cloud, self.box_dims, self.min_pts, self.max_pts, self.work_dir, id_offset, split)
            )
            threads.append(thread)
            id_offset += len(split)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        print(f"Preprocessing completed in {total_time:.2f} seconds")

        self._write_summary(total_time)

    def _write_summary(self, preprocessing_time):
        summary_headers = [
            "PlotId", "Point Cloud Filename", "Plot Centre X", "Plot Centre Y",
            "Plot Radius", "Plot Radius Buffer", "Num Points Original PC",
            "Num Points Trimmed PC", "Num Points Subsampled PC",
            "Preprocessing Time (s)"
        ]

        plot_summary = pd.DataFrame(columns=summary_headers)

        plot_summary.loc[0] = [
            self.filename[:-4],
            self.config["point_cloud_filename"],
            self.config["plot_centre"][0],
            self.config["plot_centre"][1],
            self.config["plot_radius"],
            self.config["plot_radius_buffer"],
            self.num_points_orig,
            self.num_points_trimmed,
            self.num_points_subsampled,
            preprocessing_time,
        ]

        plot_summary.to_csv(os.path.join(self.output_dir, "plot_summary.csv"), index=False)
        print("Plot summary written.")
