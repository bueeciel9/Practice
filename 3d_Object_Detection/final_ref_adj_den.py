import numpy as np
import open3d as o3d
import torch
from spconv.pytorch.utils import PointToVoxel

def load_kitti_velodyne_bin(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    return point_cloud.reshape((-1, 4))

def voxelization(points, voxel_size, point_cloud_range, num_point_features, max_num_voxels, max_num_points_per_voxel):
    voxel_generator = PointToVoxel(voxel_size, point_cloud_range, num_point_features, max_num_voxels, max_num_points_per_voxel)
    pc_th = torch.from_numpy(points)
    voxels, coords, num_points = voxel_generator(pc_th)
    voxel_centers = coords[:, [2, 1, 0]] * torch.tensor(voxel_size) + torch.tensor(voxel_size) / 2
    num_points = num_points.numpy()
    return voxel_centers, num_points

def calculate_density(num_points, max_num_points_per_voxel):
    density = num_points / max_num_points_per_voxel
    return density

def find_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size):
    adjacent_indices = []
    reference_voxel_tensor = torch.tensor(reference_voxel, dtype=torch.float).unsqueeze(0)
    for i, center in enumerate(voxel_centers):
        diff = torch.abs(center - reference_voxel_tensor)
        conditions = torch.logical_and(diff >= 0, diff <= torch.tensor(voxel_size))
        if torch.all(conditions) and not torch.all(torch.eq(center, reference_voxel_tensor)):
            adjacent_indices.append(i)
    return adjacent_indices

def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size):
    colors = np.zeros((voxel_centers.shape[0], 3))

    for reference_voxel in reference_voxels:
        adjacent_indices = find_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size)

        # Color reference voxel in yellow
        ref_idx = np.where(np.all(voxel_centers.numpy() == reference_voxel.numpy(), axis=1))[0][0]
        colors[ref_idx] = [1, 1, 0]

        # Color adjacent voxels in green
        for adj_idx in adjacent_indices:
            colors[adj_idx] = [0, 1, 0]

        # Color highest density voxel in red
        if adjacent_indices:
            highest_density_idx = np.argmax(density[adjacent_indices])
            colors[adjacent_indices[highest_density_idx]] = [1, 0, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)

    point_size = 0.1

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=800)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color

    vis.run()
    vis.destroy_window()


def save_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size, output_filename):
    colors = np.zeros((voxel_centers.shape[0], 3))

    for reference_voxel in reference_voxels:
        adjacent_indices = find_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size)

        # Color reference voxel in yellow
        ref_idx = np.where(np.all(voxel_centers.numpy() == reference_voxel.numpy(), axis=1))[0][0]
        colors[ref_idx] = [1, 1, 0]

        # Color adjacent voxels in green
        for adj_idx in adjacent_indices:
            colors[adj_idx] = [0, 1, 0]

        # Color highest density voxel in red
        if adjacent_indices:
            highest_density_idx = np.argmax(density[adjacent_indices])
            colors[adjacent_indices[highest_density_idx]] = [1, 0, 0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_filename, pcd)




bin_file = '000000.bin'
point_cloud_data = load_kitti_velodyne_bin(bin_file)

# Define voxelization parameters
voxel_size = [0.1, 0.1, 0.2]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
num_point_features = 4
max_num_points_per_voxel = 5
max_voxel_num = 20000

voxel_centers, num_points = voxelization(point_cloud_data, voxel_size, point_cloud_range, num_point_features, max_voxel_num, max_num_points_per_voxel)
density = calculate_density(num_points, max_num_points_per_voxel)

# Select 10 random voxel centers
random_voxel_indices = np.random.choice(voxel_centers.shape[0], 10, replace=False)
reference_voxels = voxel_centers[random_voxel_indices]

# Visualize the voxelized point cloud with reference and adjacent voxels
visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size)


# output_filename = "output.ply"
# save_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size, output_filename)
