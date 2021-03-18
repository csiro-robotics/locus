""" Segment and generate Locus descriptor for each scan in a sequence. """

import sys
import os
import glob
import yaml
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'descriptor_generation'))
from utils.kitti_dataloader import *
from utils.augment_scans import *
from segmentation.extract_segments import *
from segmentation.extract_segment_features import *
from descriptor_generation.locus_descriptor import *

seg_timer, feat_timer, desc_timer = Timer(), Timer(), Timer()

# Load params
parser = argparse.ArgumentParser()
parser.add_argument("--seq", default='02', help="KITTI sequence number")
parser.add_argument("--aug_type", default='none', help="Scan augmentation type ['occ', 'rot', 'ds']")
parser.add_argument("--aug_param", default=0, type=float, help="Scan augmentation parameter")
args = parser.parse_args()
print('Sequence: ', args.seq, ', Augmentation: ', args.aug_type, ', Param: ', args.aug_param)

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

transforms, _ = load_poses_from_txt(sequence_path + 'poses.txt')
rel_transforms = get_delta_pose(transforms)


# Setup database variables
num_queries = len(rel_transforms)
segments_database, features_database  = [], []
seg_corres_database, locus_descriptor_database = [], []
database_dict = {'segments_database': segments_database,
                    'features_database': features_database,
                    'seg_corres_database': seg_corres_database,
                    'rel_transforms': rel_transforms} 


for query_idx in tqdm(range(num_queries)):
    scan = next(scans)
    scan = scan[:, :-1]
    if args.aug_type != 'none':
        scan = augmented_scan(scan, args.aug_type, args.aug_param)

    # Extract segments
    seg_timer.tic()
    segments = get_segments(scan, seg_params)
    segments_database.append(segments)
    seg_timer.toc()

    # Extract segment features
    feat_timer.tic()
    features = get_segment_features(segments)
    features_database.append(features)
    feat_timer.toc()

    # Generate 'Locus' global descriptor
    desc_timer.tic()
    locus_descriptor = get_locus_descriptor(query_idx, desc_params, database_dict)
    locus_descriptor_database.append(locus_descriptor)
    desc_timer.toc()

print('Average time per scan:')
print(f"--- seg: {seg_timer.avg}s, feat: {feat_timer.avg}s, desc: {desc_timer.avg}s ---")

save_dir = cfg_params['paths']['save_dir'] + args.seq
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_pickle(locus_descriptor_database, save_dir +
            '/locus_descriptor_database.pickle')