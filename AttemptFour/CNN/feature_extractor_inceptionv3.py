import numpy as np
import tensorflow as tf
import time
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/AttemptFour')
#import utils
import tqdm
from nsd_access import NSDAccess
import pandas as pd
from DataLoaders import load_avg_betas as loader

"""
    Uses the InceptionV3 network
"""

gpu_to_use = 0

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


all_keys = list(range(0, 73000))

nsd_loader = NSDAccess("/home/seagie/NSD3")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSD access initalized ...")

image_model = tf.keras.applications.InceptionV3(include_top = False, weights = 'imagenet')
print("Image model loaded ... ")

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
print("Image model built ... ")


def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

features = np.zeros((73000, 8, 8, 2048), dtype=np.float32)


batch_size = 64
for i in tqdm.tqdm(range(0, len(all_keys), batch_size)):
    keys = all_keys[i:i+batch_size] 
    imgs = load_image(keys)
    
    feat = image_features_extract_model(imgs) # (bs, 4096)
    features[i:i+batch_size] = feat


print("feature extraction complete ... ")
print(features.shape)


for i, key in tqdm.tqdm(enumerate(all_keys)):
    with open(f"/huge/seagie/data/inception_v3/KID{key+1}.npy", "wb") as f:
        np.save(f, features[i])



print("done.")
