import os
import numpy as np
import utils
import cv2


class KITTI(object):
    """Load and parse KITTI object into an usable format.

    """
    def __init__(self, root, split='training'):
        self.root = root
        self.split = split
        self.path = os.path.join(root, split)
        self.image = os.path.join(self.path, 'image_2')
        self.label = os.path.join(self.path, 'label_2')
        self.calib = os.path.join(self.path, 'calib')
        self.lidar = os.path.join(self.path, 'velodyne')
        self.size = 7481 if split == 'training' else 7518

    def __len__(self):
        return self.size

    def get_image(self, i):
        assert(i < self.size)
        fname = os.path.join(self.image, '{:06d}.png'.format(i))
        return utils.load_image(fname)

    def get_lidar(self, i, dtype=np.float32):
        assert(i < self.size)
        fname = os.path.join(self.lidar, '{:06d}.bin'.format(i))
        return utils.load_velodyne(fname, dtype=dtype)

    def get_calibration(self, i):
        assert(i < self.size)
        fname = os.path.join(self.calib, '{:06d}.txt'.format(i))
        return utils.Calibration(fname)

    def get_label(self, i):
        assert(i < self.size)
        fname = os.path.join(self.label, '{:06d}.txt'.format(i))
        return utils.load_label(fname)
