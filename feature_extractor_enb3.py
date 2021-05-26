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

"""

CNN feature extractor using Efficient-net B3

"""

gpu_to_use = monitor(9000, wait = 1)

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

nsd_loader = NSDAccess("/home/seagie/NSD")

nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)


image_model = tf.keras.applications.EfficientNetB3(include_top = False, weights = 'imagenet', pooling="avg", input_shape=(425, 425, 3))

new_input = image_model.input
hidden_layer = image_model.layers[-1].output # take last layer, which is a global avg pooling -> (batch, 1536)
print("Output from EfficientNetB3: ", hidden_layer.shape)
features_shape = hidden_layer.shape[-1]

#print("--- layers ---")
#for i in range(0, len(image_model.layers)):
#    print(image_model.layers[i].output)


image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

training_img_idx = list(range(0, 73000))

def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx) # (bs, 425, 425, 3)
    #img = tf.image.resize(img, (224, 224)) # vgg16
    #img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

features = np.zeros((73000, features_shape))

batch_size = 64
print(f"Extracting features... | batch_size: {batch_size}")

with tf.device(f'/gpu:{gpu_to_use}'): 
    for i in tqdm.tqdm(range(0, len(training_img_idx), batch_size)):
        indicies = training_img_idx[i:i+batch_size]
        ## Load and Pre-process image
        imgs = load_image(indicies)
        
        feat = image_features_extract_model(imgs) # (bs, 1536)
        features[i:i+batch_size] = feat
        #features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

print("feature extraction complete")
print(features.shape)
print(features[-1, 0:10])

with open("img_features_enb3", "wb") as f:
    np.save(f, features)


print("done.")
