from create_dataset.render import render_pointcloud
from create_dataset.util import (
    parse_intrinsic,
    parse_extrinsic, 
    parse_annotations,
    parse_depth_map,
    annotations_mask,
    visible_points_mask,
    convert_to_camera_coordinates,
    project_points_to_image
)
import open3d as o3d
import numpy as np
from scipy.signal import convolve2d


# Paths to your data files
# Paths to your data files
annotations_file = 'data/420683/420683_annotations.json'
laser_scan_file = 'data/420683/420683_laser_scan.ply'
annotated_pcd_file = 'data/420683/420683_laser_scan_annotated.ply'
intrinsics_file = 'data/420683/42445137/hires_wide_intrinsics/42445137_5535.957.pincam'
trajectory_file = 'data/420683/42445137/hires_poses.traj'
timestamp = '5535.957'
depth_map_file = 'data/420683/42445137/hires_depth/42445137_5535.957.png'  # Update with actual path
depth_threshold = 0.005
pcd = o3d.io.read_point_cloud(laser_scan_file)
annotations = parse_annotations(annotations_file)
extrinsic = parse_extrinsic(trajectory_file, timestamp)
intrinsics = parse_intrinsic(intrinsics_file)
depth_map = parse_depth_map(depth_map_file)



######## render only visible annotated points

# points = np.asarray(pcd.points)

# annotated_mask = annotations_mask(points, annotations)
# points = points[annotated_mask]

# visible_mask = visible_points_mask(points, intrinsics, extrinsic, depth_map, depth_threshold)
# points = points[visible_mask]

# # Create a new point cloud with the filtered points
# filtered_pcd = o3d.geometry.PointCloud()
# filtered_pcd.points = o3d.utility.Vector3dVector(points)
# filtered_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(points), 1)))


# render_pointcloud(filtered_pcd, intrinsics, extrinsic, "visible_points.png")

######## create a gaussian heatmap of the visible annotated points

points = np.asarray(pcd.points)

annotated_mask = annotations_mask(points, annotations)
points = points[annotated_mask]

visible_mask = visible_points_mask(points, intrinsics, extrinsic, depth_map, depth_threshold)
points = points[visible_mask]

points_cam = convert_to_camera_coordinates(points, extrinsic)
points_img = project_points_to_image(points_cam, intrinsics)

# round the points to the nearest pixel
points_img = np.round(points_img).astype(int)


# Function to generate a Gaussian Kernel
def gaussian_kernel(size: int, sigma: float):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

# print maximum column valus of points_img
print(np.max(points_img, axis=0))

size = 100
sigma = 10

# Create the base image with padding
image_np = np.zeros((intrinsics.height, intrinsics.width))  # Note: switched width/height order

for point_img in points_img:
    y, x = point_img[:2]  # Only take x,y coordinates, ignore z
    image_np[x, y] += 1

gaussian_kernel = gaussian_kernel(size, sigma)

# convolve the gaussian kernel with the image
image_np = convolve2d(image_np, gaussian_kernel, mode='same', boundary='symm')

from matplotlib import pyplot as plt
plt.imshow(image_np)
plt.savefig("heatmap.png")