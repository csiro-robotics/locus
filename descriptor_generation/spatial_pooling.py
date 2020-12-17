""" Topological relationships and spatial feature pooling. """

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import distance
import open3d as o3d
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *

def get_segment_MTD(segments, topk):
    """ Topological relationships based on Minimum Translational Distance (MTD)"""
    """ Returns IDs and distances to nearest segments for all segments"""

    # Compute convex hulls of segments
    hulls = []
    for segment in segments:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(segment)
        hull, _ = pcd.compute_convex_hull()
        hulls.append(np.asarray(hull.vertices))

    # Calculate MTDs between all segments
    num_points = len(segments)
    dist_mat = np.zeros((num_points, num_points))
    for j in range(num_points):
        p_j = hulls[j]
        for k in range(num_points):
            if j >= k:  # Only need to calculate upper triangle.
                continue
            p_k = hulls[k]
            dist = np.min(distance.cdist(p_j, p_k, 'euclidean'))
            dist_mat[j][k] = dist
            dist_mat[k][j] = dist

    # Find 'topk' closest segments for each segment
    min_dists = []
    min_dist_ids = []
    for s in range(num_points):
        dist_vec = dist_mat[s]
        min_dist_id = dist_vec.argsort()[1:topk+1]
        min_dist_ids.append(min_dist_id)
        min_dists.append(dist_vec[min_dist_id])
    return np.asarray(min_dists), np.asarray(min_dist_ids)


def get_spatial_features(idx, topk, database_dict):
    """ Return the pooled feature using topological relationships """

    features = database_dict['features_database'][idx]
    segments = database_dict['segments_database'][idx]

    if len(features) < 7:
        return []

    seg_tdists, seg_tdist_ids = get_segment_MTD(segments, topk)
    pooled_softmax_features = np.zeros((len(features), np.shape(features)[1]))

    # For each segment, pool features from related segments
    for c in range(len(segments)):
        dist = seg_tdists[c]
        ind = seg_tdist_ids[c]
        exp_dists = np.exp(-0.1*dist)
        exp_dists /= np.sum(exp_dists)

        for nn_idx in range(min(topk, len(ind))):
            f_vec = features[ind[nn_idx]]
            pooled_softmax_features[c] += exp_dists[nn_idx]*f_vec

    return pooled_softmax_features


#####################################################################################
# Test
#####################################################################################

if __name__ == "__main__":

    seq = '06'
    data_dir = '/mnt/7a46b84a-7d34-49f2-b8f0-00022755f514/seg_test/kitti/' + seq
    topk = 5

    features_database = load_pickle(data_dir + '/features_database.pickle')
    segments_database = load_pickle(data_dir + '/segments_database.pickle')
    database_dict = {'segments_database': segments_database,
                     'features_database': features_database}
    num_queries = len(features_database)
    pooled_softmax_features_database = []

    for query_idx in range(num_queries):
        pooled_softmax_features = get_spatial_features(
            query_idx, topk, database_dict)
        pooled_softmax_features_database.append(pooled_softmax_features)

        if (query_idx % 100 == 0):
            print('', query_idx, 'complete:', (query_idx*100)/num_queries, '%')
            sys.stdout.flush()

    save_dir = '/mnt/bracewell/seg_test/kitti/' + seq
    save_pickle(pooled_softmax_features_database, save_dir +
                '/second_order/spatial_features_database.pickle')

    print('Test complete.')
