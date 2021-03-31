"""Code for loading KITTI odometry dataset"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *

#####################################################################################
# Load poses
#####################################################################################

def transfrom_cam2velo(Tcam):
    R = np.array([ 7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo

def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)


def get_delta_pose(transforms):
    rel_transforms = []
    for i in range(len(transforms)-1):
        w_T_p1 = transforms[i]
        w_T_p2 = transforms[i+1]

        p1_T_w = T_inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        rel_transforms.append(p1_T_p2)
    return rel_transforms

#####################################################################################
# Load scans
#####################################################################################


""" Helper functions from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py """


def load_bin_scan(file):
    """Load and reshape binary file containing single point cloud"""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_bin_scans(bin_files):
    """Generator to load multiple point clouds sequentially"""
    for file in bin_files:
        yield load_bin_scan(file)

def visualize_scan_open3d(ptcloud_xyz, colors = []):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
        if colors != []:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

def visualize_sequence_open3d(bin_files, n_scans):
    """Visualize scans using Open3D"""

    scans = yield_bin_scans(bin_files)

    for i in range(n_scans):
        scan = next(scans)
        ptcloud_xyz = scan[:, :-1]
        print(ptcloud_xyz.shape)
        visualize_scan_open3d(ptcloud_xyz)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn

#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    # Set the dataset location here:
    basedir = '/mnt/088A6CBB8A6CA742/Datasets/Kitti/dataset/'

    ##################
    # Test poses

    fig, axs = plt.subplots(4, 6, constrained_layout=True)
    fig.suptitle('KITTI sequences', fontsize=16)
    for i in range(22):
        sequence = str(i)
        if i < 10:
            sequence = '0' + str(i)
        sequence_path = basedir + 'sequences/' + sequence + '/'
        poses_file = sorted(
            glob.glob(os.path.join(sequence_path, 'poses.txt')))
        _, positions = load_poses_from_txt(poses_file[0])
        print('seq: ', sequence, 'len', len(positions))

        axs[i//6, i % 6].plot(positions[:,0], positions[:,1])
        axs[i//6, i % 6].set_title('seq: ' + sequence + 'len' + str(len(positions)))

    plt.show()

    ##################
    # Test scans

    sequence = '00'
    sequence_path = basedir + 'sequences/' + sequence + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))
    # Visualize some scans
    visualize_sequence_open3d(bin_files, 2)

    ##################
    # Test timestamps
    timestamps_file = basedir + 'sequences/' + sequence + '/times.txt'
    timestamps = load_timestamps(timestamps_file)
    print("Start time (s): ", timestamps[0])
    print("End time (s): ", timestamps[-1])

    print('Test complete.')
