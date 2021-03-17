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

# Pick random training img indicies
#training_img_idx = np.random.randint(0, 73000, n_training_samples)
#training_img_idx = np.sort(training_img_idx)
#print(training_img_idx)
training_img_idx = np.arange(0, 73000)


# load relevant annotations {idx :[]}
training_img_annt = utils.load_annotations_dict(training_img_idx)


training_img_annt_edit = {}
# append <start> <end> to each caption
for key, val in training_img_annt.items():
    ls = []
    new_key = int(key) # json dump won't allow int64 type
    for cap in val:
        # strip punctuation from captions
        new_cap = cap.translate(str.maketrans('', '', string.punctuation))
        # append <start> <end> tokens
        ls.append(f"<start> {new_cap} <end>")
    training_img_annt_edit[new_key] = ls

training_img_annt = training_img_annt_edit

# save modified captions dictionary
utils.dump_json(training_img_annt, "../modified_annotations_dictionary.json")


# load the relevant images
#images = nsd_loader.read_images(training_img_idx)



# TODO: Extract Image features using pre-trained VGG16 and save to disk



