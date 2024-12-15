import numpy as np
import open3d as o3d
import json
from PIL import Image
import torch

def parse_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def annotations_mask(points, annotations):
    mask = np.zeros(len(points), dtype=bool)
    for annotation in annotations:
        for index in annotation['indices']:
            mask[index] = True
    return mask

def mark_annotations(pcd, annotations):
    for annotation in annotations:
        for index in annotation['indices']:
            pcd.colors[index] = [1, 0, 0]
    return pcd

def convert_to_camera_coordinates(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Convert points from world to camera coordinates
    Args:
        points: Points in world coordinates, shape (N,3)
        extrinsic: 4x4 extrinsic matrix transforming world to camera coordinates
    Returns:
        Points in camera coordinates, shape (N,3)
    """
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones)).T  # Shape: (4, N)
    extrinsic = np.linalg.inv(extrinsic)
    points_cam = extrinsic @ points_h  # Shape: (4, N)
    points_cam = points_cam[:3, :].T  # Shape: (N, 3)
    return points_cam

def project_points_to_image(points: np.ndarray, intrinsics: o3d.camera.PinholeCameraIntrinsic) -> np.ndarray:
    fx, fy = intrinsics.get_focal_length()
    cx, cy = intrinsics.get_principal_point()
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    u = (fx * x) / z + cx
    v = (fy * y) / z + cy
    
    # convert to (N, 3)
    points_img = np.vstack((u, v, z)).T
    
    return points_img

def visible_points_mask(points, intrinsics, extrinsic, depth_map, depth_threshold):
    points_cam = convert_to_camera_coordinates(points, extrinsic)
    points_img = project_points_to_image(points_cam, intrinsics)
    
    height, width = intrinsics.height, intrinsics.width
        
    u = points_img[:, 0].astype(np.int32)           
    v = points_img[:, 1].astype(np.int32)
    z = points_img[:, 2]
    
    image_frame_valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    u = u[image_frame_valid]
    v = v[image_frame_valid]
    z = z[image_frame_valid]
    
    # Get depth values from the depth map
    depth_map_values = depth_map[v, u]  # Convert mm to meters if needed
    
    # Compute the difference between point depths and depth map values
    depth_diff = np.abs(z - depth_map_values)
    
    # Filter points that are within the depth threshold
    depth_valid = depth_diff < depth_threshold
    
    valid_mask = np.zeros(len(points), dtype=bool)
    valid_mask[image_frame_valid] = depth_valid
    
    return valid_mask


# Function to generate a Gaussian Kernel
def gaussian_kernel(size: int, sigma: float):
    # Create meshgrid
    x = torch.arange(-(size-1)/2, (size-1)/2 + 1, dtype=torch.float32)
    y = torch.arange(-(size-1)/2, (size-1)/2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    # Generate kernel
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2))/(2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Reshape for convolution (1, 1, size, size)
    return kernel.view(1, 1, size, size)

