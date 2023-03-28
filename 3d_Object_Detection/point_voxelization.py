import numpy as np
import open3d as o3d
from spconv.utils import points_to_voxel

def load_kitti_velodyne_bin(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    return point_cloud.reshape((-1, 4))

def voxelization(point_cloud_data, voxel_size, point_cloud_range, max_points_per_voxel=30):
    voxel_grid = points_to_voxel(point_cloud_data,
                                 voxel_size=voxel_size,
                                 coors_range=point_cloud_range,
                                 max_points=max_points_per_voxel,
                                 max_voxels=20000)
    
    voxel_centers = voxel_grid['coordinates'][:, [2, 1, 0]] * voxel_size + voxel_size / 2
    return voxel_centers

def visualize_voxelized_point_cloud(voxel_centers):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers)
    pcd.paint_uniform_color([0.3, 0.7, 0.3])  # Green color
    point_size = 0.1

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=800)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color
    
    vis.run()
    vis.destroy_window()

bin_file = '000000.bin'
point_cloud_data = load_kitti_velodyne_bin(bin_file)

# Define voxelization parameters
voxel_size = [0.2, 0.2, 0.2]  # Adjust this to control the voxel size
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # Define the point cloud range

voxel_centers = voxelization(point_cloud_data, voxel_size, point_cloud_range)
visualize_voxelized_point_cloud(voxel_centers)
