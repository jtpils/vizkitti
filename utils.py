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
    def __init__(self, fname):
        with open(fname) as fp:
            lines = fp.readlines()
        line = lines[2].strip().split(' ')[1:]
        self.P2 = np.array(line, dtype=np.float32)
        self.P2 = self.P2.reshape(3, 4)
        line = lines[3].strip().split(' ')[1:]
        self.P3 = np.array(line, dtype=np.float32)
        self.P3 = self.P3.reshape(3, 4)
        line = lines[4].strip().split(' ')[1:]
        self.R0 = np.array(line, dtype=np.float32)
        self.R0 = self.R0.reshape(3, 3)
        line = lines[5].strip().split(' ')[1:]
        self.V2C = np.array(line, dtype=np.float32)
        self.V2C = self.V2C.reshape(3, 4)


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
    points = np.hstack([xyz, np.ones((n, 1))])
    return points


def lidar2rect(xyz, calib):
    points = cart2hom(xyz)
    points = np.dot(points, np.dot(calib.V2C.T, calib.R0.T))
    return points


def roty(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def compute_box3d(obj):
    R = roty(obj.ry)
    l, w, h = obj.l, obj.w, obj.h
    x = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y = [0,0,0,0,-h,-h,-h,-h]
    z = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    box = np.dot(R, np.vstack([x, y, z]))
    box = box.T + obj.pos
    return box
