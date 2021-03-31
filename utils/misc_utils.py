""" Miscellaneous functions """
import numpy as np
from numpy import dot
from numpy.linalg import norm
import time
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
    """ Coversion from Eclidean coordinates to Homogeneous """
    h_point = np.concatenate([e_point,[1]])
    return h_point


def homogeneous_to_euclidean(h_point):
    """ Coversion from Homogeneous coordinates to Eclidean """
    e_point = h_point/ h_point[3]
    e_point = e_point[:3]
    return e_point

#####################################################################################
# Timing

class Timer(object):
  """A simple timer."""
  # Ref: https://github.com/chrischoy/FCGF/blob/master/lib/timer.py

  def __init__(self, binary_fn=None, init_val=0):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.binary_fn = binary_fn
    self.tmp = init_val

  def reset(self):
    self.total_time = 0
    self.calls = 0
    self.start_time = 0
    self.diff = 0

  @property
  def avg(self):
    return self.total_time / self.calls

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    if self.binary_fn:
      self.tmp = self.binary_fn(self.tmp, self.diff)
    if average:
      return self.avg
    else:
      return self.diff

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
