import numpy as np
import tensorflow as tf
import nsdloader.tfrecord_util as tfu
import os, sys
sys.path.append("~/NSD/Code/Masters-Thesis/")
from param import config as c

def load_dataset(subj, subset, nparallel):
    root = c['DATASETS']
    workdir = os.path.join(root, subj + "_" + subset)
    filepaths = list()
    filepaths += [os.path.join(workdir, p) for p in os.listdir(workdir)]

    means = np.loadtxt(os.path.join(c["NSD_STAT"], "means_" + subj + ".csv"))
    stds = np.loadtxt(os.path.join(c["NSD_STAT"], "stds_" + subj + ".csv"))
    
    ds = tf.data.TFRecordDataset(filepaths, num_parallel_reads=nparallel)
    ds = ds.map(lambda x: tfu.read_tfrecord_with_info(x))
    ds = ds.map(lambda a, b, c, d, e, f: (a, e, f)) # (betas, idx, img)
    # Z-scoring
    ds = ds.map(lambda a, e, f: ((a - means) / stds, e, f))
    
    return ds
