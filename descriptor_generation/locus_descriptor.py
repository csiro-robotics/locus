""" Generate Locus descriptor. Spatiotemporal feature pooling followed by O2P + PE. """

import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize

from spatial_pooling import *
from temporal_pooling import *

def get_locus_descriptor(idx, config_dict, database_dict):

    features = database_dict['features_database'][idx]
    feature_dim = 64 #np.shape(features)[1]

    # Get spatially and temporally pooled features. 
    spatial_features = get_spatial_features(
        idx, config_dict['spatial_topk'], database_dict)
    temporal_features = get_temporal_features(
        idx, config_dict['n_frames_max'], [], database_dict)  

    if spatial_features == [] or temporal_features == []:
        print('Degenerate scene. ID: ', idx)
        return []

    # Second order pooling (O2P) of complementary features.
    locus_matrix = np.zeros((feature_dim, feature_dim))
    for feature_idx in range(len(features)):
        sa_feature = np.asarray(features[feature_idx])
        spatial_feature = np.asarray(spatial_features[feature_idx])
        temporal_feature = np.asarray(temporal_features[feature_idx])
        spatiotemporal_feature = (spatial_feature + temporal_feature)/2

        if config_dict['fb_mode'] == 'structural':
            second_order_feature = np.outer(sa_feature, sa_feature)
        elif config_dict['fb_mode'] == 'spatial':
            second_order_feature = np.outer(sa_feature, spatial_feature)
        elif config_dict['fb_mode'] == 'temporal':
            second_order_feature = np.outer(sa_feature, temporal_feature)
        else:
            second_order_feature = np.outer(sa_feature, spatiotemporal_feature)

        locus_matrix = np.maximum(locus_matrix, second_order_feature)

    # Power Euclidean (PE) non-linear transform.
    u_, s_, vh_ = np.linalg.svd(locus_matrix)
    s_alpha = np.power(s_, config_dict['PE_alpha'])
    locus_matrix_PE = np.dot(u_ * s_alpha, vh_)

    # Flatten and normalize.
    if config_dict['fb_mode'] == 'structural':
        locus_descriptor = locus_matrix_PE[np.triu_indices(feature_dim)]
        locus_descriptor = locus_descriptor/norm(locus_descriptor)
        descriptor_length = int((feature_dim/2)*(feature_dim+1))

    else:
        locus_descriptor = normalize(locus_matrix_PE, norm='l2', axis=1, copy=True, return_norm=False)
        locus_descriptor = locus_descriptor/norm(locus_descriptor)
        locus_descriptor = np.hstack(locus_descriptor)
        descriptor_length = feature_dim*feature_dim

    return locus_descriptor.reshape(-1, descriptor_length)


#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":

    import sys
    import yaml

    seq = '08'

    cfg_file = open('config.yml', 'r')
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
    desc_params = cfg_params['descriptor_generation']

    poses_file = cfg_params['paths']['KITTI_dataset'] + 'sequences/' + seq + '/poses.txt'
    transforms, _ = load_poses_from_txt(poses_file)
    rel_transforms = get_delta_pose(transforms)

    data_dir = cfg_params['paths']['save_dir'] + seq
    features_database = load_pickle(data_dir + '/features_database.pickle')
    segments_database = load_pickle(data_dir + '/segments_database.pickle')

    num_queries = len(features_database)
    seg_corres_database = []
    database_dict = {'segments_database': segments_database,
                     'features_database': features_database,
                     'seg_corres_database': seg_corres_database,
                     'rel_transforms': rel_transforms} 

    locus_descriptor_database = []

    for query_idx in range(num_queries):
        locus_descriptor = get_locus_descriptor(query_idx, desc_params, database_dict)
        locus_descriptor_database.append(locus_descriptor)

        if (query_idx % 100 == 0):
            print('', query_idx, 'complete:', (query_idx*100)/num_queries, '%')
            sys.stdout.flush()

    save_dir = cfg_params['paths']['save_dir'] + seq
    save_pickle(locus_descriptor_database, save_dir +
                '/second_order/locus_descriptor_database.pickle')
