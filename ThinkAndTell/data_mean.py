from nsd_access import NSDAccess
from load_dataset import load_dataset
import pandas as pd
import tensorflow as tf
import numpy as np


"""
Raw betas are (327684,) sized (approx. 78GB with 64 bit encoding)
Visual cortex is 'only' (62756,) (approx. 16GB)
"""


with open('masks/visual_mask_lh.npy', 'rb') as f, open('masks/visual_mask_rh.npy', 'rb') as g:
    visual_mask_lh = np.load(f)
    visual_mask_rh = np.load(g)
    print("> visual region masks loaded from file") 

visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))


def apply_mask(x, mask):
    # Apply the visual area mask to the verticies
    return x[mask == 1]



dataset_unq = load_dataset("subj02", "unique", nparallel=54)#tf.data.experimental.AUTOTUNE)
dataset_shr = load_dataset("subj02", "shared", nparallel=54)#tf.data.experimental.AUTOTUNE)

# Apply the mask to keep only the visual cortex (62756,) instead of (327684,)
# The full betas would take up a little less than 40GB of space 
dataset_unq = dataset_unq.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_unq = dataset_unq.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))
# Apply mask to shared data
dataset_shr = dataset_shr.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_shr = dataset_shr.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))

BUFFER_SIZE = 1000
BATCH_SIZE = 512

dataset_test = dataset_shr.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

shr_data = np.zeros((3000, 62756))
shr_indicies = []

c = 0
for i in dataset_test:
    betas, NSD_idx = i
    # betas = (256, 327684)
    # NSD_idx = (256,)
    shr_data[c:c+betas.shape[0]] = betas
    shr_indicies.append(NSD_idx)     

    c += betas.shape[0]
    print(f"Shr: {c}", end='\r')
print()

mean_ = np.mean(shr_data, axis = 0)
mean_2 = np.mean(shr_data, axis=1)

print(mean_.shape)
print(mean_2.shape)

std_ = np.std(shr_data, axis =0)
std_2 = np.std(shr_data, axis = 1)

print(std_.shape)
print(std_2.shape)
