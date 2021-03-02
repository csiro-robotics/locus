"""Code for extracting Segmap-CNN features for input segments. """
# Ref: https://github.com/ethz-asl/segmap/blob/master/segmappy/bin/segmappy_plot_roc_from_matches
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from segmappy.core.config import *
from segmappy.core.dataset import *
from segmappy.core.preprocessor import *
from segmappy.core.generator import *
from segmappy.tools.classifiertools import get_default_preprocessor


# read config file
configfile = "default_training.ini"
config = Config(configfile)

# load preprocessor
preprocessor = get_default_preprocessor(config)

def get_segment_features(segments):

    # Create dummy inputs
    n_classes = len(segments)
    classes  = np.arange(len(segments), dtype=np.int64)
    positions = [map(float, np.zeros(3)) for i in range(len(segments))]

    preprocessor.init_segments(segments, classes, positions=np.asarray(positions))
    gen_test = Generator(
        preprocessor,
        range(len(segments)),
        n_classes,
        train=False,
        batch_size=config.batch_size,
        shuffle=False,
    )

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(
        os.path.join(config.cnn_model_folder, "model.ckpt.meta")
    )

    # get key tensorflow variables
    cnn_graph = tf.get_default_graph()
    cnn_input = cnn_graph.get_tensor_by_name("InputScope/input:0")
    scales = cnn_graph.get_tensor_by_name("scales:0")
    descriptor = cnn_graph.get_tensor_by_name("OutputScope/descriptor_read:0")

    cnn_features = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(config.cnn_model_folder))

        for batch in range(gen_test.n_batches):
            batch_segments, _ = gen_test.next()
            batch_descriptors = sess.run(
                descriptor,
                feed_dict={cnn_input: batch_segments, scales: preprocessor.last_scales},
            )
            for batch_descriptor in batch_descriptors:
                cnn_features.append(batch_descriptor)
    
    return np.array(cnn_features)


#####################################################################################
# Test
#####################################################################################


if __name__ == "__main__":
    import os
    import sys
    import yaml
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.misc_utils import *

    seq = '06'
    cfg_file = open('config.yml', 'r')
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

    data_dir = cfg_params['paths']['save_dir'] + seq
    segments_database = load_pickle(data_dir + '/segments_database.pickle')

    features_database = []

    for idx in range(len(segments_database)):
        segments = segments_database[idx]
        features = get_segment_features(segments)
        print("cnn_features: ", np.shape(features))
        features_database.append(features)

    save_dir = cfg_params['paths']['save_dir'] + seq
    save_pickle(features_database, save_dir +
                '/segment_features_database.pickle')