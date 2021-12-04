## Compute the n=4096 most active vertices in the betas
# Take an average across the training (unique) set and keep only the top n vertices
# Compare these to doing the same on the validation (shared) set and see if they contain the same vertices
#

import os, sys
import numpy as np
import pandas as pd
import re
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import collections 
import yaml
import nibabel as nb
import load_avg_betas as loader

##### Parameters

betas_path = "/fast/seagie/data/subj_2/betas_averaged_no_zscore/"
guse_path  = "/fast/seagie/data/subj_2/guse_averaged/"
GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

N = 4096

##### ------------

## Load NSD keys
df = pd.read_csv('./TrainData/subj02_conditions.csv')
unq_keys, shr_keys = loader.get_nsd_keys("", subject="subj02")


## Load guse captions
guse_unq = np.zeros((9000, 512), dtype=np.float32)
for i, key in enumerate(unq_keys):
    with open(f"{guse_path}/guse_embedding_KID{key}.npy", "rb") as f:
        guse_unq[i, :] = np.load(f)

guse_shr = np.zeros((1000, 512), dtype=np.float32)
for i, key in enumerate(shr_keys):
    with open(f"{guse_path}/guse_embedding_KID{key}.npy", "rb") as f:
        guse_shr[i, :] = np.load(f)
print(f"GUSE loaded")

# Load unq betas
betas_unq = np.zeros((9000, 327684), dtype=np.float32)
for i, key in enumerate(unq_keys):
    with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
        betas_unq[i, :] = np.load(f)
print("betas unq:", betas_unq.shape)

# Load shr betas
betas_shr = np.zeros((1000, 327684), dtype=np.float32)
for i, key in enumerate(shr_keys):
    with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
        betas_shr[i, :] = np.load(f)
print("betas shr:", betas_shr.shape)


avg_betas_unq = np.mean(np.abs(betas_unq), axis=0)
avg_betas_shr = np.mean(np.abs(betas_shr), axis=0)

sum_betas_unq = np.mean(np.abs(betas_unq), axis=0)

# sort the indices least to most active
idx_avg_betas_unq = np.argsort(avg_betas_unq)
idx_avg_betas_shr = np.argsort(avg_betas_shr)

idx_sum_betas_unq = np.argsort(sum_betas_unq)

#print("unq betas most active idx:", idx_avg_betas_unq[-N:])
#print("shr betas most active idx:", idx_avg_betas_shr[-N:])
most_active_unq = idx_avg_betas_unq[-N:]
most_active_shr = idx_avg_betas_shr[-N:]

most_active_unq_sum = idx_sum_betas_unq[-N:]

with open(f"./TrainData/sum_most_active_vert.txt", "w") as f:
    for i in most_active_unq_sum:
        f.write(f"{i}\n")

#print(avg_betas_unq[most_active_unq[-10:]])
#print(avg_betas_shr[most_active_shr[-10:]])

print(len(np.intersect1d(most_active_unq, most_active_shr)))
print(len(np.intersect1d(most_active_unq, most_active_unq_sum)))


