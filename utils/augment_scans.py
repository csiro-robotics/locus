"""Code for distorting point clouds."""

import numpy as np
import open3d as o3d

def occlude_scan(scan, angle):
    # Remove points within a sector of fixed angle and random heading direction.
    thetas = (180/np.pi) * np.arctan2(scan[:,1],scan[:,0])
    heading = (180-angle/2)*np.random.uniform(-1,1)
    occ_scan = np.vstack((scan[thetas < (heading - angle/2)] , scan[thetas > (heading + angle/2)]))
    return occ_scan

def random_rotate_scan(scan, r_angle, is_random = True):
    # Rotate about z-axis. 
    if is_random:
        r_angle = r_angle*np.random.uniform()
    cos_angle = np.cos(r_angle)
    sin_angle = np.sin(r_angle)
    rot_matrix = np.array([[cos_angle, sin_angle, 0],
                                [-sin_angle, cos_angle, 0],
                                [0,             0,      1]])
    augmented_scan = np.dot(scan, rot_matrix)

    return np.asarray(augmented_scan, dtype=np.float32)

def downsample_scan(scan, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points)    

def distort_scan(scan, n_sigma, r_angle):
    # Add gaussian noise and rotate about z-axis. 
    noise = np.clip(n_sigma * np.random.randn(*scan.shape), -0.1, 0.1)
    noisy_scan = scan + noise

    return random_rotate_scan(noisy_scan, r_angle, False)

#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    # Set the dataset location here:
    basedir = '/mnt/7a46b84a-7d34-49f2-b8f0-00022755f514/datasets/Kitti/dataset/'
    sequence = '00'

    import os
    import sys
    import glob 
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.misc_utils import *
    from utils.kitti_dataloader import *
    
    sequence_path = basedir + 'sequences/' + sequence + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))
    scans = yield_bin_scans(bin_files)

    for i in range(10): 
        scan = next(scans)
        scan = scan[:, :-1]
        print('Scan ID: ', i)
        visualize_scan_open3d(scan)
        visualize_scan_open3d(occlude_scan(scan, 90))
        visualize_scan_open3d(random_rotate_scan(scan, np.pi, True))
        visualize_scan_open3d(distort_scan(scan, 0.8, 0))
        visualize_scan_open3d(downsample_scan(scan, 0.5))
