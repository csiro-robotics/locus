"""Code for extracting Euclidean segments from a point cloud."""

import numpy as np
import pcl
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.kitti_dataloader import visualize_scan_open3d

def extract_cluster_indices(cloud_filtered, seg_params):
    tree = cloud_filtered.make_kdtree()      
    ec = cloud_filtered.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(seg_params['c_tolerence']) 
    ec.set_MinClusterSize(seg_params['c_min_size'])
    ec.set_MaxClusterSize(seg_params['c_max_size'])
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices    

def extract_segments(scan, seg_params):
    if seg_params['visualize']:
        visualize_scan_open3d(scan)

    # Ground Plane Removal
    cloud = pcl.PointCloud(scan)
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(seg_params['g_dist_thresh'])
    seg.set_normal_distance_weight(seg_params['g_normal_dist_weight'])
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()

    crop_xyz = np.asarray(cloud)
    for k, indice in enumerate(indices):
        crop_xyz[indice][2] = -20.0
    
    crop_xyz = crop_xyz[crop_xyz[:, -1] > -seg_params['g_height']]

    # Voxel filter (optional)  
    ds_f = seg_params['ds_factor']
    if ds_f > 0.01:    
        cloud = pcl.PointCloud(crop_xyz)
        vg = cloud.make_voxel_grid_filter()
        vg.set_leaf_size(ds_f, ds_f, ds_f)
        cloud_filtered = vg.filter()
    else:
        cloud_filtered = pcl.PointCloud(crop_xyz)

    if seg_params['visualize']:
        visualize_scan_open3d(cloud_filtered)

    # Euclidean Cluster Extraction
    cluster_indices = extract_cluster_indices(cloud_filtered, seg_params)

    if seg_params['visualize']:
        print('cluster_indices : ' , np.shape(cluster_indices))

    segments = []
    points_database = []
    colours_database = []
    init = False

    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 3), dtype=np.float32)

        for k, indice in enumerate(indices):
            points[k][0] = cloud_filtered[indice][0]
            points[k][1] = cloud_filtered[indice][1]
            points[k][2] = cloud_filtered[indice][2]

        # Additional filtering step to remove flat(ground-plane) segments
        x_diff = (max(points[:,0]) - min(points[:,0]))
        y_diff = (max(points[:,1]) - min(points[:,1]))
        z_diff = (max(points[:,2]) - min(points[:,2]))
        if (not seg_params['filter_flat_seg']) or (max(x_diff,y_diff)/z_diff < seg_params['horizontal_ratio']):
            segments.append(points)
            colour = np.random.random_sample((3))
            if init:
                points_database = np.vstack((points_database, points))
                colour = np.tile(colour, (len(indices), 1))
                colours_database = np.vstack((colours_database, colour))
            else:
                points_database = points
                colour = np.tile(colour, (len(indices), 1))
                colours_database = colour
                init = True

    if seg_params['visualize']:
        visualize_scan_open3d(points_database, colours_database)    

    return segments

def get_segments(scan, seg_params):
    segments = extract_segments(scan, seg_params)

    # Handling rare degenerate scenes
    c_tolerence = seg_params['c_tolerence']
    while seg_params['enforce_min_seg_count'] and len(segments) < seg_params['min_seg_count']: 
        seg_params['c_tolerence'] -= 0.01
        if seg_params['c_tolerence'] < 0.01:
            break
        segments = extract_segments(scan, seg_params)
    seg_params['c_tolerence'] = c_tolerence
    return segments

#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    import glob 
    import yaml
    from utils.kitti_dataloader import yield_bin_scans
    
    seq = '00'

    cfg_file = open('config.yml', 'r')
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    seg_params = cfg_params['segmentation']

    basedir = cfg_params['paths']['KITTI_dataset']
    sequence_path = basedir + 'sequences/' + seq + '/'
    bin_files = sorted(glob.glob(os.path.join(
        sequence_path, 'velodyne', '*.bin')))
    scans = yield_bin_scans(bin_files)    

    segments_database = []

    for i in range(10): 
        scan = next(scans)
        segments = extract_segments(scan[:, :-1], seg_params)
        print('Extracted segments: ', np.shape(segments))
        segments_database.append(segments)
    
    save_dir = cfg_params['paths']['save_dir'] + seq
    save_pickle(segments_database, save_dir +
                '/segments_database.pickle')