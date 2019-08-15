import cv2
import numpy as np


class Object3d(object):
    def __init__(self, string):
        data = string.split(' ')
        data[1:] = [float(x) for x in data[1:]]
        self.cls = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]
        self.occlusion = data[2]
        self.alpha = data[3]  # observation angle [-pi..pi]
        self.box2d = np.array(data[4:8])
        self.h = data[8]
        self.w = data[9]
        self.l = data[10]
        self.pos = np.array(data[11:14])
        self.ry = data[14]  # yaw angle


class Calibration(object):
    pass


def load_image(fname):
    return cv2.imread(fname)


def load_velodyne(fname, dtype=np.float32):
    pcd = np.fromfile(fname, dtype=dtype)
    pcd = pcd.reshape([-1, 4])
    return pcd


def load_label(fname):
    with open(fname, 'r') as fp:
        labels = [Object3d(line) for line in fp]
    return labels


def cart2hom(xyz):
    n = xyz.shape[0]
    hom = np.hstack([xyz, np.ones((n, 1))])
    return hom


def xyzwhl2eight(xyz, w, h, l):
    """Convert 3D box to eight corners.

    Returns:
       np.ndarray: vertices of shape (8, 3) in the following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1

    """
    x, y, z = xyz
    w, h, l = w / 2, h / 2, l / 2
    return np.array([
        [x + w, x + w, x - w, x - w, x + w, x + w, x - w, x - w],
        [y - h, y + h, y + h, y - h, y - h, y + h, y + h, y - h],
        [z - l, z - l, z - l, z - l, z + l, z + l, z + l, z + l]
    ]).T
