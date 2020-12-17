""" Miscellaneous functions """
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle

#####################################################################################
# Data loading/saving


def load_pickle(file_name):
    dbfile1 = open(file_name, 'rb')
    file_data = pickle.load(dbfile1)
    dbfile1.close()
    return file_data


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    print('Finished saving: ', file_name)

#####################################################################################
# Place recognition

def check_if_revisit(query_pose, db_poses, thres):
    num_dbs = np.shape(db_poses)[0]
    is_revisit = 0

    for i in range(num_dbs):
        dist = norm(query_pose - db_poses[i])
        if ( dist < thres ):
            is_revisit = 1
            break

    return is_revisit

#####################################################################################
# Math

def cosine_distance(feature_a, feature_b): 
    return 1 - dot(feature_a, np.transpose(feature_b))/(norm(feature_a)*norm(feature_b))

def T_inv(T_in):
    """ Return the inverse of input homogeneous transformation matrix """
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))


def is_nan(x):
    return (x != x)


def euclidean_to_homogeneous(e_point):
    """ y and z are switched to account for transfrom in KITTI data """
    h_point = np.ones(4)
    h_point[0] = e_point[0]
    h_point[1] = e_point[2]
    h_point[2] = e_point[1]
    return h_point


def homogeneous_to_euclidean(h_point):
    """ y and z are switched to account for transfrom in KITTI data """
    e_point = np.ones(3)
    e_point[0] = h_point[0] / h_point[3]
    e_point[1] = h_point[2] / h_point[3]
    e_point[2] = h_point[1] / h_point[3]
    return e_point

#####################################################################################
# Config

font = {'family': 'serif',
        # 'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
font_legend = {'family': 'serif',
        # 'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
