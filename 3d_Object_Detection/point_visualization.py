import numpy as np
import plotly.express as px
import open3d as o3d

def load_kitti_velodyne_bin(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    return point_cloud.reshape((-1, 4))

def visualize_point_cloud(point_cloud_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    o3d.visualization.draw_geometries([pcd])

# def visualize_point_cloud_plotly(point_cloud_data):
#     fig = px.scatter_3d(
#         x=point_cloud_data[:, 0],
#         y=point_cloud_data[:, 1],
#         z=point_cloud_data[:, 2],
#         opacity=0.5,
#         size_max=2
#     )
#     fig.update_traces(marker=dict(size=1))
#     fig.update_layout(width=1000, height=800)
#     fig.show()

# bin_file = '000001.bin'
# point_cloud_data = load_kitti_velodyne_bin(bin_file)
# visualize_point_cloud_plotly(point_cloud_data)



def visualize_point_cloud(point_cloud_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    
    # Set point size and colors
    pcd.paint_uniform_color([0.3, 0.7, 0.3])  # Green color
    point_size = 0.05
    
    # Set background color
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=800)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background color
    
    vis.run()
    vis.destroy_window()
    
bin_file = '000001.bin'
point_cloud_data = load_kitti_velodyne_bin(bin_file)
visualize_point_cloud(point_cloud_data)