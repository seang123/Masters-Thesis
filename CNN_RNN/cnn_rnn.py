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
import string

# Select training set
n_training_samples = 2 # /73k

nsd_loader = NSDAccess("/home/seagie/NSD")

nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

# load relevant annotations {idx :[]}
# training_img_annt = utils.load_annotations_dict(training_img_idx)
training_img_annt = utils.load_json("../modified_annotations_dictionary.json")

# load the relevant images
#images = nsd_loader.read_images(training_img_idx)



# TODO: Extract Image features using pre-trained VGG16 and save to disk



