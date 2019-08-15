import kitti
import numpy as np
import open3d as o3d
import utils


def draw_lidar_label(points, labels):
    # boxes = [utils.xyzwhl2eight(x.pos, x.w, x.h, x.l) for x in labels]
    # lines = [[0, 1], [1, 2], [2, 3], [3, 0],
    #          [0, 4], [1, 5], [3, 7], [2, 6],
    #          [4, 5], [5, 6], [6, 7], [7, 4]]
    box = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
           [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3],
             [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # for box in boxes:
    #     print(box)
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.Vector3dVector(box)
    lineset.lines  = o3d.Vector2iVector(lines)
    lineset.colors = o3d.Vector3dVector(colors)
    vis.add_geometry(lineset)

    vis.run()
    vis.destroy_window()


def draw_lidar(points):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    option = vis.get_render_option()
    option.point_size = 1.0
    ctrl = vis.get_view_control()
    ctrl.change_field_of_view(step=90)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    dataset = kitti.KITTI('data', split='training')
    points = dataset.get_lidar(0)
    labels = dataset.get_label(0)
    draw_lidar_label(points[:, :3], labels)
