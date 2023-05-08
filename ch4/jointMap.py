import cv2 as cv
import sys
sys.path.append('..')
import slam_utils as su
import numpy as np
from numba import jit, njit
import numba as nb
import open3d as o3d


colorImgs = []
depthImgs = []
for i in range(5):
    colorImgs.append(cv.imread(f"./color/{i+1}.png", cv.IMREAD_COLOR))
    depthImgs.append(cv.imread(f"./depth/{i+1}.pgm", -1))

poses = su.read_trajectory("./pose.txt", startIdx=0)


@njit
def rangeToCloud(color, depth, T):
    cx = 325.5
    cy = 253.5
    fx = 518.0
    fy = 519.0
    depthScale = 1000.0
    pointcloud = nb.typed.List()
    colors = nb.typed.List()
    for v in np.arange(0, color.shape[0]):
        for u in np.arange(0, color.shape[1]):
            d = depth[v, u]/depthScale
            if d != 0:  # 0 means no valid value
                point = np.array([(u - cx)*d/fx, (v - cy)*d/fy, d, 1])
                pointWorld = np.dot(T, point)
                pointcloud.append(pointWorld[:3])
                colors.append(color[v][u]/255.0)
    return pointcloud, colors


Colors = []
Points = []
for i in range(5):
    print(f"Converting image {i+1}")
    color = colorImgs[i]
    depth = depthImgs[i]
    T = np.asarray(poses[i].to_homogenous_matrix().to_list()).astype(float)
    p, c = rangeToCloud(color, depth, T)
    Colors = Colors + np.array(c).tolist()
    Points = Points + np.array(p).tolist()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Points)
pcd.colors = o3d.utility.Vector3dVector(Colors)
o3d.visualization.draw_geometries([pcd])