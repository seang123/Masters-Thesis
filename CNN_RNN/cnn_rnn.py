# Show and Tell | CNN RNN network implementation

# import
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
from nsd_access import NSDAccess
import sys
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import json
import pandas as pd
import tensorflow as tf


# Select training set
n_training_samples = 2 # /73k

nsd_loader = NSDAccess("/home/seagie/NSD")

nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

# Pick random training img indicies
training_img_idx = np.random.randint(0, 73000, n_training_samples)
training_img_idx = np.sort(training_img_idx)
print(training_img_idx)

# load relevant annotations
training_img_annt = utils.load_annotations_dict(training_img_idx)


images = nsd_loader.read_images(training_img_idx)

print(images.shape)

print(training_img_annt)

# TODO: Extract Image features using pre-trained VGG16 and save to disk
