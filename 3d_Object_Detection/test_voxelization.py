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
    # voxel_centers = coords[:, [2, 1, 0]] * voxel_size + voxel_size / 2
    num_points = num_points.numpy()
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
voxel_size = [0.1, 0.1, 0.2]  # Adjust this to control the voxel size
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # Define the point cloud range
num_point_features = 4
max_num_points_per_voxel = 5
max_voxel_num = 20000

voxel_centers = voxelization(point_cloud_data, voxel_size, point_cloud_range, num_point_features, max_voxel_num, max_num_points_per_voxel)
# voxel_centers = voxelization(point_cloud_data, voxel_size, point_cloud_range)
visualize_voxelized_point_cloud(voxel_centers)


## spconv guide

def main_pytorch_voxel_gen():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    gen = PointToVoxel(vsize_xyz=[0.1, 0.1, 0.1],
                       coors_range_xyz=[-80, -80, -6, 80, 80, 6],
                       num_point_features=3,
                       max_num_voxels=5000,
                       max_num_points_per_voxel=5)

    pc = np.random.uniform(-4, 4, size=[1000, 3])
    pc_th = torch.from_numpy(pc)
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
    voxels_np = voxels_th.numpy()
    indices_np = indices_th.numpy()
    num_p_in_vx_np = num_p_in_vx_th.numpy()
    print(f"------Raw Voxels {voxels_np.shape[0]}-------")
    print(voxels_np[0])
    # run voxel gen and FILL MEAN VALUE to voxel remain
    voxels_th, indices_th, num_p_in_vx_th = gen(pc_th, empty_mean=True)
    voxels_np = voxels_th.numpy()
    indices_np = indices_th.numpy()
    num_p_in_vx_np = num_p_in_vx_th.numpy()
    print("------Voxels with mean filled-------")
    print(voxels_np[0])
    voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
    print("------Voxel ids for every point-------")
    print(pc_voxel_id[:10])


def main_pytorch_voxel_gen_cuda():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    pc = np.random.uniform(-2, 8, size=[1000, 3]).astype(np.float32)

    for device in [torch.device("cuda:0"), torch.device("cpu:0")]:
        gen = PointToVoxel(vsize_xyz=[0.25, 0.25, 0.25],
                        coors_range_xyz=[0, 0, 0, 10, 10, 10],
                        num_point_features=3,
                        max_num_voxels=5000,
                        max_num_points_per_voxel=5,
                        device=device)

        pc_th = torch.from_numpy(pc).to(device)
        voxels_th, indices_th, num_p_in_vx_th = gen(pc_th)
        voxels_np = voxels_th.cpu().numpy()
        indices_np = indices_th.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_th.cpu().numpy()
        print(f"------{device} Raw Voxels {voxels_np.shape[0]}-------")
        print(voxels_np[0])
        # run voxel gen and FILL MEAN VALUE to voxel remain
        voxels_tv, indices_tv, num_p_in_vx_tv = gen(pc_th, empty_mean=True)
        voxels_np = voxels_tv.cpu().numpy()
        indices_np = indices_tv.cpu().numpy()
        num_p_in_vx_np = num_p_in_vx_tv.cpu().numpy()
        print(f"------{device} Voxels with mean filled-------")
        print(voxels_np[0])
        voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
        print(f"------{device} Reconstruct Indices From Voxel ids for every point-------")
        indices_th_float = indices_th.float()
        # we gather indices by voxel_id to see correctness of voxel id.
        indices_th_voxel_id = gather_features_by_pc_voxel_id(indices_th_float, pc_voxel_id)
        indices_th_voxel_id_np = indices_th_voxel_id[:10].cpu().numpy()
        print(pc[:10])
        print(indices_th_voxel_id_np[:, ::-1] / 4)


def main_gather_features_by_pc_voxel_id():
    np.random.seed(50051)
    # voxel gen source code: spconv/csrc/sparse/pointops.py
    device = torch.device("cuda:0")
    gen = PointToVoxel(vsize_xyz=[0.25, 0.25, 0.25],
                       coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                       num_point_features=3,
                       max_num_voxels=2000,
                       max_num_points_per_voxel=5,
                       device=device)

    pc = np.random.uniform(-8, 8, size=[5000, 3]).astype(np.float32)
    pc_th = torch.from_numpy(pc).to(device)

    voxels_th, indices_th, num_p_in_vx_th, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
    res_features_from_seg = torch.zeros((voxels_th.shape[0], 128), dtype=torch.float32, device=device)
    
    pc_features = gather_features_by_pc_voxel_id(res_features_from_seg, pc_voxel_id)
    print(pc.shape, pc_features.shape)

def main():
    main_pytorch_voxel_gen()
    main_pytorch_voxel_gen_cuda()
    main_gather_features_by_pc_voxel_id()

if __name__ == '__main__':
    main()

