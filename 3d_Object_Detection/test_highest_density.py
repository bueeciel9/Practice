# import numpy as np
# import open3d as o3d
# import spconv.pytorch as spconv
# import torch
# from spconv.pytorch.utils import PointToVoxel

# def load_kitti_velodyne_bin(bin_file):
#     point_cloud = np.fromfile(bin_file, dtype=np.float32)
#     return point_cloud.reshape((-1, 4))

# def voxelization(points, voxel_size, point_cloud_range, num_point_features, max_num_voxels, max_num_points_per_voxel):
#     voxel_generator = PointToVoxel(voxel_size, point_cloud_range, num_point_features, max_num_voxels, max_num_points_per_voxel)
#     pc_th = torch.from_numpy(points)
#     voxels, coords, num_points = voxel_generator(pc_th)
#     voxel_centers = coords[:, [2, 1, 0]] * torch.tensor(voxel_size) + torch.tensor(voxel_size) / 2
#     num_points = num_points.numpy()
#     return voxel_centers, num_points

# def calculate_density(num_points, max_num_points_per_voxel):
#     density = num_points / max_num_points_per_voxel
#     return density


# def find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxels, voxel_size):
#     reference_indices = []

#     for reference_voxel in reference_voxels:
#         reference_voxel_tensor = torch.tensor(reference_voxel, dtype=torch.float)
#         distances = torch.abs(voxel_centers - reference_voxel_tensor)
#         reference_idx = np.where(np.all(distances.numpy() < 1e-6, axis=1, keepdims=True))[0]

#         adjacent_indices = []
#         if reference_idx.size != 0:
#             reference_indices.append(reference_idx[0])

#             for i, center in enumerate(voxel_centers):
#                 diff = torch.abs(center - reference_voxel_tensor)
#                 conditions = torch.logical_and(diff >= torch.tensor(voxel_size), diff <= torch.tensor(voxel_size) * 2)
#                 if torch.all(conditions):
#                     adjacent_indices.append(i)

#     return reference_indices, adjacent_indices

# def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size):
#     reference_indices, adjacent_indices = find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxels, voxel_size)

#     if adjacent_indices:
#         highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()
#     else:
#         highest_density_idx = []

#     indices_to_visualize = []
#     if reference_indices:
#         indices_to_visualize.extend(reference_indices)
#     indices_to_visualize.extend(adjacent_indices)
#     indices_to_visualize.extend(highest_density_idx)

#     voxel_centers_to_visualize = voxel_centers.numpy()[indices_to_visualize]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(voxel_centers_to_visualize)

#     colors = np.zeros((len(indices_to_visualize), 3))

#     # Color reference voxels in yellow
#     for i, idx in enumerate(reference_indices):
#         colors[i] = [1, 1, 0]

#     # Color adjacent voxels in green
#     for i, idx in enumerate(adjacent_indices):
#         colors[len(reference_indices)+i] = [0, 1, 0]

#     # Color highest density voxels in red
#     for i, idx in enumerate(highest_density_idx):
#         colors[len(reference_indices)+len(adjacent_indices)+i] = [1, 0, 0]

#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     point_size = 0.1

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1000, height=800)
#     vis.add_geometry(pcd)
#     render_option = vis.get_render_option()
#     render_option.point_size = point_size
#     render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color

#     vis.run()
#     vis.destroy_window()

# bin_file = '000000.bin'
# point_cloud_data = load_kitti_velodyne_bin(bin_file)

# # Define voxelization parameters
# voxel_size = [0.1, 0.1, 0.2]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
# num_point_features = 4
# max_num_points_per_voxel = 5
# max_voxel_num = 20000

# voxel_centers, num_points = voxelization(point_cloud_data, voxel_size, point_cloud_range, num_point_features, max_voxel_num, max_num_points_per_voxel)
# density = calculate_density(num_points, max_num_points_per_voxel)


# # Load point cloud data
# point_cloud_data = load_kitti_velodyne_bin(bin_file)

# # Select 100 random voxel centers
# random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)

# for i in range(10):
#     # Select a new set of 100 random voxel centers

#     # Visualize the voxelized point cloud with reference and adjacent voxels

#     # Select a new set of 100 random voxel centers for the next iteration
#     random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)
#     reference_voxels = voxel_centers[random_voxel_indices]
#     visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size)

#######################################################################################################################


import numpy as np
import open3d as o3d
import spconv.pytorch as spconv
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

def find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxels, voxel_size):
    reference_indices = []
    adjacent_indices_list = []
    highest_density_indices_list = []

    for reference_voxel in reference_voxels:
        reference_voxel_tensor = torch.tensor(reference_voxel, dtype=torch.float)
        distances = torch.abs(voxel_centers - reference_voxel_tensor)
        reference_idx = np.where(np.all(distances.numpy() < 1e-6, axis=1, keepdims=True))[0]

        adjacent_indices = []
        if reference_idx.size != 0:
            reference_indices.append(reference_idx[0])

            for i, center in enumerate(voxel_centers):
                diff = torch.abs(center - reference_voxel_tensor)
                conditions = torch.logical_and(diff >= torch.tensor(voxel_size), diff <= torch.tensor(voxel_size) * 2)
                if torch.all(conditions):
                    adjacent_indices.append(i)

            adjacent_indices_list.append(adjacent_indices)

            if adjacent_indices:
                highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()
            else:
                highest_density_idx = []
            
            highest_density_indices_list.append(highest_density_idx)

    return reference_indices, adjacent_indices_list, highest_density_indices_list

def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, adjacent_indices_list, highest_density_indices_list, voxel_size):
    indices_to_visualize = []
    colors = np.zeros((voxel_centers.shape[0], 3))

    # Color reference voxels in yellow
    for i, idx in enumerate(reference_voxels):
        colors[idx] = [1, 1, 0]

        # Color adjacent voxels in green
        for adj_idx in adjacent_indices_list[i]:
            colors[adj_idx] = [0, 1, 0]

        # Color highest density voxels in red
        for hd_idx in highest_density_indices_list[i]:
            colors[hd_idx] = [1, 0, 0]

        indices_to_visualize.extend([idx] + adjacent_indices_list[i] + highest_density_indices_list[i])

    voxel_centers_to_visualize = voxel_centers.numpy()[indices_to_visualize]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers_to_visualize)
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


# Load point cloud data
point_cloud_data = load_kitti_velodyne_bin(bin_file)

# Select 100 random voxel centers
random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)

for i in range(10):
    # Select a new set of 100 random voxel centers
    random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)
    reference_voxels = voxel_centers[random_voxel_indices]

    # Visualize the voxelized point cloud with reference and adjacent voxels
    visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size)















# # Select 100 random voxel centers
# random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)
# reference_voxels = voxel_centers[random_voxel_indices]

# visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size)





# def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxel, voxel_size):
#     reference_idx, adjacent_indices = find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size)

#     if adjacent_indices:
#         highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()
#     else:
#         highest_density_idx = []

#     indices_to_visualize = []
#     # if reference_idx.size != 0:
#     if reference_idx:
#         indices_to_visualize.extend(reference_idx)
#     indices_to_visualize.extend(adjacent_indices)
#     indices_to_visualize.extend(highest_density_idx)

#     voxel_centers_to_visualize = voxel_centers.numpy()[indices_to_visualize]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(voxel_centers_to_visualize)

#     colors = np.zeros((len(indices_to_visualize), 3))

#     for i, idx in enumerate(indices_to_visualize):
#         if idx in highest_density_idx:
#             colors[i] = [1, 0, 0]  # Red color for the highest density voxel
#         elif idx in adjacent_indices:
#             colors[i] = [0, 1, 0]  # Green color for the adjacent voxels
#         else:
#             colors[i] = [0, 0, 1]  # Blue color for the reference voxel

#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     point_size = 0.1

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1000, height=800)
#     vis.add_geometry(pcd)
#     render_option = vis.get_render_option()
#     render_option.point_size = point_size
#     render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color

#     vis.run()
#     vis.destroy_window()




# reference_voxel = [35.2, 0, 1]
# visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxel, voxel_size)



# def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxel, voxel_size):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(voxel_centers.numpy())
    
#     reference_idx, adjacent_indices = find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size)

#     colors = np.zeros((len(density), 3))
#     for i, d in enumerate(density):
#         red_intensity = d
#         colors[i] = [red_intensity, 0.7, 0.3]
    
#     if adjacent_indices:
#         highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()
#         for idx in highest_density_idx:
#             colors[idx] = [1, 0, 0]  # Red color for the highest density voxel

#     if reference_idx.size != 0:
#         colors[reference_idx] = [0, 0, 1]  # Blue color for the reference voxel

#     for idx in adjacent_indices:
#         colors[idx] = [0, 1, 0]  # Green color for the adjacent voxels

#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     point_size = 0.1

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1000, height=800)
#     vis.add_geometry(pcd)
#     render_option = vis.get_render_option()
#     render_option.point_size = point_size
#     render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color
    
#     vis.run()
#     vis.destroy_window()

# def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxel, voxel_size):
#     reference_idx, adjacent_indices = find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxel, voxel_size)

#     if adjacent_indices:
#         highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()

#     indices_to_visualize = []
#     if reference_idx.size != 0:
#         indices_to_visualize.extend(reference_idx)
#     indices_to_visualize.extend(adjacent_indices)
#     indices_to_visualize.extend(highest_density_idx)

#     voxel_centers_to_visualize = voxel_centers.numpy()[indices_to_visualize]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(voxel_centers_to_visualize)

#     colors = np.zeros((len(indices_to_visualize), 3))

#     for i, idx in enumerate(indices_to_visualize):
#         if idx in highest_density_idx:
#             colors[i] = [1, 0, 0]  # Red color for the highest density voxel
#         elif idx in adjacent_indices:
#             colors[i] = [0, 1, 0]  # Green color for the adjacent voxels
#         else:
#             colors[i] = [0, 0, 1]  # Blue color for the reference voxel

#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     point_size = 0.1

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1000, height=800)
#     vis.add_geometry(pcd)
#     render_option = vis.get_render_option()
#     render_option.point_size = point_size
#     render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color

#     vis.run()
#     vis.destroy_window()


###############################################################################################################
import numpy as np
import open3d as o3d
import spconv.pytorch as spconv
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

def find_reference_and_adjacent_voxel_indices(voxel_centers, reference_voxels, voxel_size):
    reference_indices = []
    adjacent_indices_list = []
    highest_density_indices_list = []

    for reference_voxel in reference_voxels:
        reference_voxel_tensor = torch.tensor(reference_voxel, dtype=torch.float)
        distances = torch.abs(voxel_centers - reference_voxel_tensor)
        reference_idx = np.where(np.all(distances.numpy() < 1e-6, axis=1, keepdims=True))[0]

        adjacent_indices = []
        if reference_idx.size != 0:
            reference_indices.append(reference_idx[0])

            for i, center in enumerate(voxel_centers):
                diff = torch.abs(center - reference_voxel_tensor)
                conditions = torch.logical_and(diff >= torch.tensor(voxel_size), diff <= torch.tensor(voxel_size) * 2)
                if torch.all(conditions):
                    adjacent_indices.append(i)

            adjacent_indices_list.append(adjacent_indices)

            if adjacent_indices:
                highest_density_idx = np.argwhere(density == np.max(density[adjacent_indices])).flatten()
            else:
                highest_density_idx = []
            
            highest_density_indices_list.append(highest_density_idx)

    return reference_indices, adjacent_indices_list, highest_density_indices_list

def visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, adjacent_indices_list, highest_density_indices_list, voxel_size):
    indices_to_visualize = []
    colors = np.zeros((voxel_centers.shape[0], 3))

    # Color reference voxels in yellow
    for i, idx in enumerate(reference_voxels):
        colors[idx] = [1, 1, 0]

        # Color adjacent voxels in green
        for adj_idx in adjacent_indices_list[i]:
            colors[adj_idx] = [0, 1, 0]

        # Color highest density voxels in red
        for hd_idx in highest_density_indices_list[i]:
            colors[hd_idx] = [1, 0, 0]

        indices_to_visualize.extend([idx] + adjacent_indices_list[i] + highest_density_indices_list[i])

    voxel_centers_to_visualize = voxel_centers.numpy()[indices_to_visualize]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_centers_to_visualize)
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


# Load point cloud data
point_cloud_data = load_kitti_velodyne_bin(bin_file)

# Select 100 random voxel centers
random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)

for i in range(10):
    # Select a new set of 100 random voxel centers
    random_voxel_indices = np.random.choice(voxel_centers.shape[0], 100, replace=False)
    reference_voxels = voxel_centers[random_voxel_indices]

    # Visualize the voxelized point cloud with reference and adjacent voxels
    visualize_voxelized_point_cloud_with_reference_and_adjacent(voxel_centers, density, reference_voxels, voxel_size)