#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import time

import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R

class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,DA=False,flip_sign=False,rot=False,drop_points=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points
        # for bev
        self.voxel_size = np.array([0.5, 0.5, 1]) #[0.1, 0.1, 0.8]) # xyz
        self.coors_range = np.array([0, -40, -3, 70.4, 40, 1]) # xyzxyz, minmax
        self.with_reflectivity = True
        self.bev_map = None
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 4), dtype=np.float32)  # [m, 3]: x, y, z

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)


        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)

        self.set_points(points)

    def set_points(self, points):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # put in attribute
        self.points = points  # get
        if self.flip_sign:
            self.points[:, 1] = -self.points[:, 1]
        if self.DA:
            jitter_x = random.uniform(-5,5)
            jitter_y = random.uniform(-3, 3)
            jitter_z = random.uniform(-1, 0)
            self.points[:, 0] += jitter_x
            self.points[:, 1] += jitter_y
            self.points[:, 2] += jitter_z
        if self.rot:
            self.points = self.points @ R.random(random_state=1234).as_dcm().T

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            # self.do_range_projection() # RV
            self.do_bev_projection() # bev

    def _points_to_bevmap_reverse_kernel(self, coor_to_voxelidx, height_lowers, max_voxels): 
        #TODO
        N = self.points.shape[0]
        ndim = self.points.shape[1] - 1
        # ndim = 3
        ndim_minus_1 = ndim - 1
        grid_size = (self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size
        # np.round(grid_size)
        # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
        grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
        height_slice_size = self.voxel_size[-1]
        coor = np.zeros(shape=(3,), dtype=np.int32)  # DHW
        voxel_num = 0
        failed = False
        for i in range(N):
            failed = False
            for j in range(ndim):
                c = np.floor((self.points[i, j] - self.coors_range[j]) / self.voxel_size[j])
                if c < 0 or c >= grid_size[j]:
                    failed = True
                    break
                coor[ndim_minus_1 - j] = c
            if failed:
                continue
            voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
            if voxelidx == -1:
                voxelidx = voxel_num
                if voxel_num >= max_voxels:
                    break
                voxel_num += 1
                coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
                # coors_2d[voxelidx] = coor[1:]
            self.bev_map[-1, coor[1], coor[2]] += 1
            height_norm = self.bev_map[coor[0], coor[1], coor[2]]
            incomimg_height_norm = (
                self.points[i, 2] - height_lowers[coor[0]]
            ) / height_slice_size
            if incomimg_height_norm > height_norm:
                self.bev_map[coor[0], coor[1], coor[2]] = incomimg_height_norm
                if self.with_reflectivity: # heights 0123, reflectivity: 4, num_points: 5
                    # self.bev_map.shape: (6, 160, 141)
                    self.bev_map[-2, coor[1], coor[2]] = self.points[i, 3]

    def do_bev_projection(self): #TODO
        if not isinstance(self.voxel_size, np.ndarray):
            self.voxel_size = np.array(self.voxel_size, dtype=self.points.dtype)
        if not isinstance(self.coors_range, np.ndarray):
            self.coors_range = np.array(self.coors_range, dtype=self.points.dtype)
        voxelmap_shape = (self.coors_range[3:] - self.coors_range[:3]) / self.voxel_size
        voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]  # DHW format
        coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
        # coors_2d = np.zeros(shape=(max_voxels, 2), dtype=np.int32)
        bev_map_shape = list(voxelmap_shape)
        bev_map_shape[0] += 1
        height_lowers = np.linspace(
            self.coors_range[2], self.coors_range[5], voxelmap_shape[0], endpoint=False
        )
        if self.with_reflectivity:
            bev_map_shape[0] += 1
        self.bev_map = np.zeros(shape=bev_map_shape, dtype=self.points.dtype)
        self._points_to_bevmap_reverse_kernel(
            # self.points,
            # voxel_size,
            # coors_range,
            coor_to_voxelidx,
            # bev_map,
            height_lowers,
            # with_reflectivity,
            max_voxels=40000,
        )


    def do_range_projection(self): # need change
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)



class GtLaserScan(LaserScan):
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0,DA=False,flip_sign=False,rot=False,drop_points=False):
        super(GtLaserScan, self).__init__(project, H, W, fov_up, fov_down,DA=DA,flip_sign=flip_sign,rot=rot, drop_points=drop_points)

        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.DA = DA
        self.flip_sign = flip_sign
        self.rot = rot
        self.drop_points = drop_points

        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 4), dtype=np.float32)  # [m, 4]: x, y, z, remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset xjust in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 6))

        # put in attribute
        points = scan[:, [0, 1, 2, -1]]  # get xyz and remission
        if self.drop_points is not False:
            self.points_to_drop = np.random.randint(0, len(points)-1,int(len(points)*self.drop_points))
            points = np.delete(points,self.points_to_drop,axis=0)

        self.set_points(points)
