from create_dataset.render import render_pointcloud
from create_dataset.util import (
    annotations_mask,
    visible_points_mask,
    convert_to_camera_coordinates,
    project_points_to_image,
    gaussian_kernel
)
import open3d as o3d
import numpy as np
from scipy.signal import convolve2d

import torch
import torch.nn.functional as F
from utils.data_parser import DataParser
import torchvision
## Initialize
import os
# Move computation to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Create and move kernel to GPU
size = 64
sigma = 8
kernel = gaussian_kernel(size, sigma).to(device)
depth_threshold = 0.01

modified_dataset_root = "modified_data"

data_parser = DataParser("data")

def save_heatmap(heatmap, visit_id, video_id, timestamp):
    # create modified_data folder if it doesn't exist
    os.makedirs(modified_dataset_root, exist_ok=True)
    
    # create visit_id folder if it doesn't exist
    os.makedirs(os.path.join(modified_dataset_root, visit_id), exist_ok=True)
    
    # create video_id folder if it doesn't exist
    os.makedirs(os.path.join(modified_dataset_root, visit_id, video_id), exist_ok=True)
    
    # save heatmap
    torch.save(heatmap, os.path.join(modified_dataset_root, visit_id, video_id, f"{timestamp}.pt"))
    # torchvision.utils.save_image(heatmap, os.path.join(modified_dataset_root, visit_id, video_id, f"{timestamp}.png"))


def get_timestamps(visit_id, video_id):
    intrinsics_map = data_parser.get_camera_intrinsics(visit_id, video_id)
    trajectory = data_parser.get_camera_trajectory(visit_id, video_id)
    depth_map = data_parser.get_depth_frames(visit_id, video_id)

    
    if not intrinsics_map or not trajectory or not depth_map:
        return []  # Return empty list if either map is empty
        
    # get intersection of both keys and sort them
    timestamps = sorted(list(set(intrinsics_map.keys()) & set(trajectory.keys()) & set(depth_map.keys())))
    return timestamps

def create_heatmap(visit_id, video_id, timestamp):
    # get laser scan points
    laser_scan = data_parser.get_laser_scan(visit_id)
    points = np.asarray(laser_scan.points)

    # get annotated points
    annotations = data_parser.get_annotations(visit_id)
    annotated_mask = annotations_mask(points, annotations)
    points = points[annotated_mask]

    # get camera intrinsics
    intrinsics_map = data_parser.get_camera_intrinsics(visit_id, video_id)
    intrinsic_path = intrinsics_map[timestamp]
    width, height, fx, fy, cx, cy = data_parser.read_camera_intrinsics(intrinsic_path)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)

    # get camera extrinsics
    trajectory = data_parser.get_camera_trajectory(visit_id, video_id)
    extrinsic = trajectory[timestamp]

    # get depth map
    depth_map = data_parser.get_depth_frames(visit_id, video_id)
    depth_map_path = depth_map[timestamp]
    depth_map = data_parser.read_depth_frame(depth_map_path)

    visible_mask = visible_points_mask(points, intrinsic, extrinsic, depth_map, depth_threshold)
    points = points[visible_mask]

    points_cam = convert_to_camera_coordinates(points, extrinsic)
    points_img = project_points_to_image(points_cam, intrinsic)

    # round the points to the nearest pixel
    points_img = np.round(points_img).astype(np.int64)

    # Create the base image
    image_tensor = torch.zeros((intrinsic.height, intrinsic.width), device=device) # (H, W)
    points_tensor = torch.from_numpy(points_img[:, :2]).to(device)

    
    # Add points to image
    for point in points_tensor:
        if point[0] < image_tensor.shape[1] and point[1] < image_tensor.shape[0]:
            image_tensor[point[1], point[0]] += 1
        else:
            print("point out of bounds", point)


    # Perform convolution
    with torch.no_grad():
        output = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=size//2)

    # save output as torch tensor file
    save_heatmap(output, visit_id, video_id, timestamp)


if __name__ == "__main__":
    import csv
    from tqdm import tqdm

    # First collect all combinations
    combinations = []
    with open("test.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            visit_id, video_id = row
            timestamps = get_timestamps(visit_id, video_id)
            for timestamp in timestamps:
                combinations.append((visit_id, video_id, timestamp))

    print(f"Processing {len(combinations)} frames")
    
    # Now iterate over the combinations with a progress bar
    for visit_id, video_id, timestamp in tqdm(combinations, desc="Processing frames"):
        if not os.path.exists(os.path.join(modified_dataset_root, visit_id, video_id, f"{timestamp}.pt")):
            create_heatmap(visit_id, video_id, timestamp)
    