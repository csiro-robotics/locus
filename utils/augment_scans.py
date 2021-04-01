"""Code for distorting point clouds."""

import numpy as np
import open3d as o3d

def occlude_scan(scan, angle):
    # Remove points within a sector of fixed angle (degrees) and random heading direction.
    thetas = (180/np.pi) * np.arctan2(scan[:,1],scan[:,0])
    heading = (180-angle/2)*np.random.uniform(-1,1)
    occ_scan = np.vstack((scan[thetas < (heading - angle/2)] , scan[thetas > (heading + angle/2)]))
    return occ_scan

def random_rotate_scan(scan, r_angle, is_random = True):
    # If is_random = True: Rotate about z-axis by random angle upto 'r_angle'. 
    # Else: Rotate about z-axis by fixed angle 'r_angle'.
    r_angle = (np.pi/180) * r_angle 
    if is_random:
        r_angle = r_angle*np.random.uniform()
    cos_angle = np.cos(r_angle)
    sin_angle = np.sin(r_angle)
    rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                                [sin_angle, cos_angle, 0],
                                [0,             0,      1]])
    augmented_scan = np.dot(scan, rot_matrix)

    return np.asarray(augmented_scan, dtype=np.float32), rot_matrix

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

def augmented_scan(scan, aug_type, param):
    if aug_type == 'occ':
        return occlude_scan(scan, param)
    elif aug_type == 'rot':
        return random_rotate_scan(scan, param)
    elif aug_type == 'ds':
        return downsample_scan(scan, param)
    else:
        return []

#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    # Set the dataset location here:
    sequence = '06'

    import os
    import sys
    import glob 
    import yaml
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.misc_utils import *
    from utils.kitti_dataloader import *

    cfg_file = open('config.yml', 'r')
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    basedir = cfg_params['paths']['KITTI_dataset']
    
    sequence_path = basedir + 'sequences/' + sequence + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))
    scans = yield_bin_scans(bin_files)

    for i in range(10): 
        scan = next(scans)
        scan = scan[:, :-1]
        print('Scan ID: ', i)
        visualize_scan_open3d(scan)
        visualize_scan_open3d(augmented_scan(scan, 'occ', 90))
        visualize_scan_open3d(augmented_scan(scan, 'rot', 180)[0])
        visualize_scan_open3d(augmented_scan(scan, 'ds', 0.5))
