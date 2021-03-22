# Show and Tell | CNN RNN network implementation

# import
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import json
import pandas as pd
import logging
import tensorflow as tf
import string
import time
import tqdm
from itertools import zip_longest
import h5py

# INFO messages NOT printed | DEBUG - everything printed | WARNINGS-ERRORS 
tf.get_logger().setLevel('INFO')

# Allow memory growth on GPU devices | Otherwise InceptionV3 won't run due to insufficient memory 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)


start_time = time.time()

# init NSD loader
nsd_loader = NSDAccess("/home/seagie/NSD")

nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

#
# load relevant annotations {idx :[]}
#
# training_img_annt = utils.load_annotations_dict(training_img_idx)
# training_img_annt = utils.load_json("../modified_annotations_dictionary.json")

# load the relevant images
#images = nsd_loader.read_images(training_img_idx)


# TODO: Extract Image features using pre-trained VGG16 and save to disk

def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx)
    #img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# INIT the IncecptionV3 model 
image_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')#, input_shape=(299,299,3))

new_input = image_model.input
hidden_layer = image_model.layers[-1].output # take the last convolutional layer as output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)



training_img_idx = list(range(0,73000))  # training img indicies

# feature extraction
# image_dataset = tf.data.Dataset.from_tensors(training_img_idx)
# image_dataset = tf.data.Dataset.range(0, 100)
# image_dataset = image_dataset.map(load_image, num_parallel_calls=2).batch(16)


batch_size = 5
with h5py.File("feature_weights.hdf5", "w") as f:
    f.create_dataset('features', data=np.zeros((batch_size,64,2048)), compression="gzip", chunks=True, maxshape=(None,None,None,), dtype=np.float32)

# each batch ~530kb | 10 batches is 5mb | so store 10 in memory and then write append them to the save file

with h5py.File("feature_weights.hdf5", 'a') as f:
    for i in tqdm.tqdm(range(0, len(training_img_idx), batch_size)):
        indicies = training_img_idx[i:i+batch_size] 
        imgs = load_image(indicies)
        batch_features = image_features_extract_model(imgs)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3])) # (batch_size, 64, 2080) <32float>
        #print("-----------------------------------")
        #print("BATCH FEATURES SHAPE: ", batch_features.shape, batch_features.dtype)

        f['features'].resize((f['features'].shape[0] + batch_features.shape[0]), axis = 0)
        f['features'][-batch_features.shape[0]:] = batch_features






print(f"Elapsed time: {(time.time() - start_time):.3f}")
