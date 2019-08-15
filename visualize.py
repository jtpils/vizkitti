import kitti
import numpy as np
import open3d as o3d
import utils


def draw_lidar_label(points, labels):
    boxes = [utils.compute_box3d(x) for x in labels]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4]]
    colors = [[1, 0, 0] for i in range(len(lines))]

    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    for i, box in enumerate(boxes):
        if labels[i].cls == 'DontCare':
            continue
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
    index = 0
    points = dataset.get_lidar(index)
    labels = dataset.get_label(index)
    calib  = dataset.get_calibration(index)
    points = utils.lidar2rect(points[:, :3], calib)
    draw_lidar_label(points, labels)
