""" Segment and generate Locus descriptor for each scan in a sequence. """

import sys
import os
import glob
import yaml
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'descriptor_generation'))
from utils.kitti_dataloader import *
from segmentation.extract_segments import *
from segmentation.generate_segment_features import *
from descriptor_generation.locus_descriptor import *

# Load params
parser = argparse.ArgumentParser()
parser.add_argument("--seq", help="KITTI sequence number")
args = parser.parse_args()

cfg_file = open('config.yml', 'r')
cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
desc_params = cfg_params['descriptor_generation']
seg_params = cfg_params['segmentation']
seg_params['visualize'] = False

# Load data
basedir = cfg_params['paths']['KITTI_dataset']
sequence_path = basedir + 'sequences/' + args.seq + '/'
bin_files = sorted(glob.glob(os.path.join(
    sequence_path, 'velodyne', '*.bin')))
scans = yield_bin_scans(bin_files)    

transforms, x, y, z = load_poses_from_txt(sequence_path + 'poses.txt')
rel_transforms = get_delta_pose(transforms)


# Setup database variables
num_queries = len(rel_transforms)
segments_database = []
features_database = []
seg_corres_database = []
locus_descriptor_database = []
database_dict = {'segments_database': segments_database,
                    'features_database': features_database,
                    'seg_corres_database': seg_corres_database,
                    'rel_transforms': rel_transforms} 

start_time = time.time()
q_start_time = time.time()

for query_idx in range(num_queries):
    scan = next(scans)

    # Extract segments
    segments = extract_segments(scan[:, :-1], seg_params)
    segments_database.append(segments)

    # Extract segment features
    features = get_segment_features(segments)
    features_database.append(features)

    # Generate 'Locus' global descriptor
    locus_descriptor = get_locus_descriptor(query_idx, desc_params, database_dict)
    locus_descriptor_database.append(locus_descriptor)

    if (query_idx % 100 == 0):
        print('', query_idx, 'complete:', (query_idx*100)/num_queries, '%')
        print("--- %s seconds ---" % (time.time() - q_start_time))
        q_start_time = time.time()
        sys.stdout.flush()  

time_taken = time.time() - start_time
print('Total time taken:')
print("--- %s seconds ---" % time_taken)
print('Average time per scan:')
print("--- %s seconds ---" % (time_taken/num_queries) )

save_dir = cfg_params['paths']['save_dir'] + args.seq
save_pickle(locus_descriptor_database, save_dir +
            '/second_order/locus_descriptor_database.pickle')