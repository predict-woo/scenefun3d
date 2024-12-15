import numpy as np
import open3d as o3d
import json
from PIL import Image

# Function to render the annotated points without opening a window
def render_pointcloud(pcd, intrinsics, extrinsic, output_path):
    # Set up the renderer
    width = intrinsics.width
    height = intrinsics.height
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])  # Set background to white

    # Create a material for the point cloud
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 10.0

    renderer.scene.add_geometry("pcd", pcd, material)

    # Ensure the coordinate systems match
    renderer.setup_camera(intrinsics, extrinsic)

    # Render the scene
    img = renderer.render_to_image()

    # Save the rendered image
    o3d.io.write_image(output_path, img)