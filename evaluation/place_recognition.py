""" Online retrieval-based place-recognition using pre-computed global descriptors. """
# Based on: https://github.com/irapkaist/scancontext/blob/master/fast_evaluator/main.m
# With updated evaluation criterea as set in: config.yaml->place_recognition

import numpy as np 
import math
import yaml
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import *
from utils.kitti_dataloader import *

# Load params
parser = argparse.ArgumentParser()
parser.add_argument("--seq", default='00', help="KITTI sequence number")
parser.add_argument("--aug_type", default='none', help="Scan augmentation type ['occ', 'rot', 'ds']")
parser.add_argument("--aug_param", default=0, type=float, help="Scan augmentation parameter")
args = parser.parse_args()

test_name = 'initial_' + args.seq

cfg_file = open('config.yml', 'r')
cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)
pr_params = cfg_params['place_recognition']
desc_dir = cfg_params['paths']['save_dir'] + args.seq
basedir = cfg_params['paths']['KITTI_dataset']
sequence_path = basedir + 'sequences/' + args.seq + '/'
revisit_criteria = pr_params['revisit_criteria']
not_revisit_criteria = pr_params['not_revisit_criteria']
skip_time = pr_params['skip_time']
kdtree_retrieval = pr_params['kdtree_retrieval']
thresholds = np.linspace(pr_params['cd_thresh_min'], pr_params['cd_thresh_max'], pr_params['num_thresholds'])
desc_file_name = '/locus_descriptor_' + cfg_params['descriptor_generation']['fb_mode']
if args.aug_type != 'none':
    desc_file_name += '_' + args.aug_type + str(int(args.aug_param))
    test_name = args.aug_type + str(int(args.aug_param)) + '_' + args.seq

#####################################################################################

locus_descriptor_database = load_pickle(desc_dir + desc_file_name + '.pickle')
_, positions_database = load_poses_from_txt(sequence_path + 'poses.txt')
timestamps = load_timestamps(sequence_path + '/times.txt')

num_queries = len(positions_database) -1
num_thresholds = len(thresholds)

# Databases of previously visited/'seen' places.
seen_poses, seen_descriptors = [], []

# Store results of evaluation.  
num_true_positive = np.zeros(num_thresholds)
num_false_positive = np.zeros(num_thresholds)
num_true_negative = np.zeros(num_thresholds)
num_false_negative = np.zeros(num_thresholds)

ret_timer = Timer()

for query_idx in tqdm(range(num_queries)):
    
    locus_descriptor = locus_descriptor_database[query_idx]
    query_pose = positions_database[query_idx]
    query_time = timestamps[query_idx]
    
    if len(locus_descriptor) < 1:
        continue

    seen_descriptors.append(locus_descriptor)
    seen_poses.append(query_pose)

    if (query_time - skip_time) < 0:
       continue
    
    # Build retrieval database using entries 30s prior to current query. 
    tt = next(x[0] for x in enumerate(timestamps) if x[1] > (query_time - skip_time))
    db_seen_descriptors = np.copy(seen_descriptors)
    db_seen_poses = np.copy(seen_poses)
    db_seen_poses = db_seen_poses[:tt+1]
    db_seen_descriptors = db_seen_descriptors[:tt+1]
    db_seen_descriptors = db_seen_descriptors.reshape(-1, np.shape(locus_descriptor)[1])

    nns = len(db_seen_descriptors) # If exaustive search
    if kdtree_retrieval: # If KDTree search
        tree = KDTree(db_seen_descriptors)  
        nn = 50
        if (np.shape(db_seen_descriptors)[0] < nn):
            nn = np.shape(db_seen_descriptors)[0]

        dist, ind = tree.query(vlad, k=nn) 
        nns = np.shape(dist)[1]

    # Find top-1 candidate.
    nearest_idx = 0
    min_dist = math.inf
    ret_timer.tic()
    for ith_candidate in range(nns):
        candidate_idx = ith_candidate
        if kdtree_retrieval:
            candidate_idx = ind[0][ith_candidate]
        
        candidate_descriptor = seen_descriptors[candidate_idx]
        distance_to_query = cosine_distance(locus_descriptor, candidate_descriptor)

        if( distance_to_query < min_dist):
            nearest_idx = candidate_idx
            min_dist = distance_to_query

    ret_timer.toc()
    place_candidate = seen_poses[nearest_idx]

    is_revisit = check_if_revisit(query_pose, db_seen_poses, revisit_criteria)

    # Evaluate top-1 candidate.
    for thres_idx in range(num_thresholds):
        threshold = thresholds[thres_idx]

        if( min_dist < threshold): # Positive Prediction
            p_dist = norm(query_pose - place_candidate)
            if p_dist < revisit_criteria:
                num_true_positive[thres_idx] += 1

            elif p_dist > not_revisit_criteria:
                num_false_positive[thres_idx] += 1   

        else: # Negative Prediction
            if(is_revisit == 0): 
                num_true_negative[thres_idx] += 1
            else:            
                num_false_negative[thres_idx] += 1 
              

print('Average retrieval time per scan:')
print(f"--- {ret_timer.avg}s---")

save_dir = cfg_params['paths']['save_dir'] + 'pr_results/' + test_name
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Saving pickles: ', test_name)
save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
save_pickle(num_false_positive, save_dir + '/num_false_positive.pickle')
save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
save_pickle(num_false_negative, save_dir + '/num_false_negative.pickle')

