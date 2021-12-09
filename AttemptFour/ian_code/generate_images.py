from nsd_access import NSDAccess
import PIL as pw
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

"""
    saves the NSD images to disk
"""

df_one = pd.read_csv("./TrainData/subj01_conditions.csv")
df_two = pd.read_csv("./TrainData/subj02_conditions.csv")

nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

keys_one = df_one['nsd_key']
keys_two = df_two['nsd_key']

keys = np.concatenate((keys_one, keys_two))

print(keys.shape)

new_size = (75, 75)

images = np.zeros((20000, new_size[0] * new_size[1]), dtype=np.float32)
print("images:", images.shape)

imgs = np.zeros((20000, 425, 425, 3), dtype=np.float32)

for i, key in tqdm(enumerate(keys)):
    imgs[i, :, :] = nsd_loader.read_images(int(key)-1)

for i, key in tqdm(enumerate(keys)):
    #img = nsd_loader.read_images(int(key) - 1) # np.array
    img = imgs[i,:,:,:]
    img = cv2.resize(img, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(new_size[0] * new_size[1])
    images[i, :] = img

for i, key in tqdm(enumerate(keys)):
    loc = f'/fast/seagie/data/images/img_KID{key}.npy'
    with open(loc, 'wb') as f:
        np.save(f, images[i, :])
