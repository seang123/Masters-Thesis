import numpy as np
import tensorflow as tf
import time
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import tqdm
from nsd_access import NSDAccess
import pandas as pd
from nv_monitor import monitor 

gpu_to_use = monitor(5000)

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

nsd_loader = NSDAccess("/home/seagie/NSD")

nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


with tf.device('/gpu:2'):
    image_model = tf.keras.applications.VGG16(include_top = True, weights = 'imagenet')

new_input = image_model.input
hidden_layer = image_model.layers[-2].output # take last fc layer (4096,)

#print("--- layers ---")
#for i in range(0, len(image_model.layers)):
#    print(image_model.layers[i])

with tf.device('/gpu:2'):
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

training_img_idx = list(range(0, 73000))

def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

features = np.zeros((73000, 4096))

with tf.device('/gpu:2'): 
    batch_size = 32
    for i in tqdm.tqdm(range(0, len(training_img_idx), batch_size)):
        indicies = training_img_idx[i:i+batch_size]
        #indicies = i
        imgs = load_image(indicies)
        
        feat = image_features_extract_model(imgs) # (bs, 4096)
        features[i:i+batch_size] = feat
        #features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

print("feature extraction complete")
print(features.shape)

with open("img_features_vgg16", "wb") as f:
    np.save(f, features)


print("done.")
