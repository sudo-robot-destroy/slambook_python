import cv2 as cv
import csv
import symforce.symbolic as sf
import open3d as o3d
import numpy as np


def make_skew6(xi: sf.Vector6) -> sf.M44:
    phi = sf.Vector3(xi[:3])
    rho = sf.Vector3(xi[3:])
    phi_hat = phi.skew_symmetric(phi)
    return sf.M44.block_matrix([
        [phi_hat, rho],
        [sf.Vector3.zeros(3, 1).transpose(), sf.Vector1(0)]
    ])


def skew_to_vec6(M: sf.M44) -> sf.Vector6:
    rot_part = M[:3, :3]
    vec = skew_to_vec3(rot_part)
    rho = M[:3, 3]
    return sf.Vector6.block_matrix([[rho], [vec]])


def skew_to_vec3(M: sf.M33) -> sf.Vector3:
    return sf.Vector3(M[2, 1], M[0, 2], M[1, 0])


def read_trajectory(filename, startIdx=1):
    trajectory = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            qx = float(row[startIdx + 3])
            qy = float(row[startIdx + 4])
            qz = float(row[startIdx + 5])
            qw = float(row[startIdx + 6])
            tx = float(row[startIdx])
            ty = float(row[startIdx + 1])
            tz = float(row[startIdx + 2])
            rot = sf.Rot3(sf.Quaternion(sf.V3(qx, qy, qz), qw))
            trans = sf.V3(tx, ty, tz)
            p1 = sf.Pose3_SE3(rot, trans)
            trajectory.append(p1)
        return trajectory


def gen_line_set(trajectory, plotFrames=True, traj_color=[0, 0, 0]):
    """Returns line_set"""
    index = 0
    points = []
    lines = []
    colors = []
    for traj in trajectory:
        length = 0.1
        Ow = traj.t
        Xw = traj * (length*sf.V3(1, 0, 0))
        Yw = traj * (length*sf.V3(0, 1, 0))
        Zw = traj * (length*sf.V3(0, 0, 1))
        points.append(Ow.to_flat_list())
        points.append(Xw.to_flat_list())
        points.append(Yw.to_flat_list())
        points.append(Zw.to_flat_list())

        if plotFrames:
            lines.append([index, index + 1])
            lines.append([index, index + 2])
            lines.append([index, index + 3])

            colors.append([1, 0, 0])
            colors.append([0, 1, 0])
            colors.append([0, 0, 1])

        index = index + 4

    index = 0
    for traj in trajectory:
        if index == 0:
            pass
        else:
            lines.append([index, index - 4])
            colors.append(traj_color)
        index = index + 4

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def find_feature_matches(img_1, img_2):
    # initalize
    matcher = cv.DescriptorMatcher.create(
        cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

    # detect Oriented FAST and compute BRIEF descriptor
    orb = cv.ORB_create()
    keypoints_1, descriptors_1 = orb.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img_2, None)
    matches = matcher.match(descriptors_1, descriptors_2)
    # sort them and remove outliers
    matches = sorted(matches, key=lambda x: x.distance)
    min_dist = matches[0].distance
    max_dist = matches[-1].distance
    print(f"Max dist: {max_dist}")
    print(f"Min dist: {min_dist}")

    # remove bad matches
    good_matches = []
    for i in range(descriptors_1.shape[0]):
        if matches[i].distance <= max(2*min_dist, 30.0):
            good_matches.append(matches[i])
    return keypoints_1, keypoints_2, good_matches


def pixel2cam(p, K):
    return np.array([(p[0] - K[0, 2]) / K[0, 0],
                     (p[1] - K[1, 2]) / K[1, 1]])


def pose_estimation_2d2d(keypoints_1, keypoints_2, matches):
    principal_point = (325.1, 249.7)  # TUM dataset
    focal_length = 521  # TUM dataset

    points1 = []
    points2 = []

    for match in matches:
        points1.append(keypoints_1[match.queryIdx].pt)
        points2.append(keypoints_2[match.trainIdx].pt)

    fundamental_matrix, _ = cv.findFundamentalMat(np.array(points1),
                                                  np.array(points2),
                                                  cv.FM_8POINT)
    print("fundamental_matrix is ")
    print(fundamental_matrix)

    essential_matrix, _ = cv.findEssentialMat(np.array(points1),
                                              np.array(points2),
                                              focal=focal_length,
                                              pp=principal_point)
    print("essential_matrix is ")
    print(essential_matrix)

    homography_matrix, _ = cv.findHomography(np.array(points1),
                                             np.array(points2),
                                             cv.RANSAC, 3)
    print("homography_matrix is ")
    print(homography_matrix)
    _, R, t, _ = cv.recoverPose(essential_matrix, np.array(points1),
                                np.array(points2), focal=focal_length,
                                pp=principal_point)
    print(f"R is {R}")
    print(f"t is {t}")
    return R, t


def triangulation(R, t, pts_1, pts_2):
    T1 = np.eye(3, 4)
    T2 = np.concatenate((R, t), axis=1)
    pts_4d = cv.triangulatePoints(T1, T2, pts_1, pts_2)
    points = []
    for i in range(pts_4d.shape[1]):
        x = pts_4d[:, i]
        p = np.array([[x[0]/x[3]], [x[1]/x[3]], [x[2]/x[3]]])
        points.append(p)
    return points
