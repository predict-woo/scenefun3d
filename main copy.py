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
depth_threshold = 0.05

modified_dataset_root = "modified_data"

data_parser = DataParser("data")

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
    
    
    points_cam = convert_to_camera_coordinates(points, extrinsic)
    points_img = project_points_to_image(points_cam, intrinsic)
    
    # plot this as a scatter plot with z as color and width and height from intrinsic
    from matplotlib import pyplot as plt
    plt.figure(figsize=(width/100, height/100))
    plt.scatter(points_img[:, 0], points_img[:, 1], c='red', s=1)
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Flip y-axis to match image coordinates
    plt.savefig(f"heatmap_{visit_id}_{video_id}_{timestamp}.png")
    plt.close()

    # get depth map
    depth_map = data_parser.get_depth_frames(visit_id, video_id)
    depth_map_path = depth_map[timestamp]
    depth_map = data_parser.read_depth_frame(depth_map_path)

    visible_mask = visible_points_mask(points, intrinsic, extrinsic, depth_map, depth_threshold)
    points = points[visible_mask]

    points_cam = convert_to_camera_coordinates(points, extrinsic)
    points_img = project_points_to_image(points_cam, intrinsic)
    
    # plot this as a scatter plot with z as color and width and height from intrinsic
    from matplotlib import pyplot as plt
    plt.figure(figsize=(width/100, height/100))
    plt.scatter(points_img[:, 0], points_img[:, 1], c='red', s=1)
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Flip y-axis to match image coordinates
    plt.savefig(f"heatmap_depth_{visit_id}_{video_id}_{timestamp}.png")
    plt.close()


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
        
    # plot output as heatmap
    plt.imshow(output.squeeze().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f"heatmap_conv_{visit_id}_{video_id}_{timestamp}.png")
    plt.close()

    # save output as torch tensor file

create_heatmap("421002", "42444876", "99365.467")
