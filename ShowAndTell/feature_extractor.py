import numpy as np
import tensorflow as tf
import time
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import tqdm
from nsd_access import NSDAccess
import pandas as pd
#from nv_monitor import monitor 

#gpu_to_use = monitor(5000)
gpu_to_use = 0

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

# Init NSD Loader
nsd_loader = NSDAccess("/home/seagie/NSD3")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

#img = nsd_loader.read_images(73000)

# Load the model
with tf.device(f'/gpu:{gpu_to_use}'):
    image_model = tf.keras.applications.VGG16(include_top = True, weights = 'imagenet')


#print("--- layers ---")
for i in range(0, len(image_model.layers)):
    print(image_model.layers[i].output_shape)
raise
print("Final CNN layer:", image_model.layers[-6].output_shape)

# Pick layer
new_input = image_model.input
hidden_layer = image_model.layers[-6].output # last fc layer [-2] (4096,)

# Init Model
with tf.device(f'/gpu:{gpu_to_use}'):
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Samples 
#training_img_idx = list(range(0, 73000))
df = pd.read_csv(f"./TrainData/subj02_conditions.csv")
training_img_idx = list(df['nsd_key'].values)
print(type(training_img_idx))


def load_image(idx: int):
    if not isinstance(idx, list):
        idx = [idx]
    img = nsd_loader.read_images(idx)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img

features = np.zeros((len(training_img_idx), 512, 14*14), dtype=np.float32) #np.zeros((73000, 4096))

with tf.device(f'/gpu:{gpu_to_use}'):
    batch_size = 20
    for i in tqdm.tqdm(range(0, len(training_img_idx), batch_size)):
        indicies = training_img_idx[i:i+batch_size]
        #indicies = i
        imgs = load_image(indicies)
        
        feat = image_features_extract_model(imgs) # (bs, 4096) for fc | (bs, 14, 14, 512) for CNN
        feat = np.swapaxes(np.reshape(feat, (feat.shape[0], 14*14, 512)), 2,1) # (bs, 512, 196)
        features[i:i+batch_size, :, :] = feat
        #features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

print("feature extraction complete")
print(features.shape)

#for i in tqdm.tqdm(range(1, 73001)):
for i, kid in tqdm.tqdm(enumerate(training_img_idx), total=len(training_img_idx)):
    with open(f"/fast/seagie/images_vgg16_cnn_out/KID_{kid}.npy", "wb") as f:
        np.save(f, features[i,:])


print("done.")
