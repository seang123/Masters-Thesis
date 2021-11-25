"""

Compare betas by NSD key that have similar guse cosine similarity

"""
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
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
from tensorflow.keras.losses import CosineSimilarity, MeanSquaredError

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
print(f"GUSE: {guse_unq.shape}")

# Load unq betas
betas_unq = np.zeros((9000, 327684), dtype=np.float32)
for i, key in enumerate(unq_keys):
    with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
        betas_unq[i, :] = np.load(f)
print("betas unq:", betas_unq.shape)

cos_sim = CosineSimilarity()

target = 100
target_guse = guse_unq[target, :]

cos_sim_ls = []
for i, v in enumerate(unq_keys):
    cos_sim_ls.append( cos_sim(guse_unq[target,:], guse_unq[i,:]) * -1 )


fig = plt.figure()
plt.plot(cos_sim_ls)
plt.savefig(f"./cos_sim_target{target}.png")
plt.close(fig)

max_v = 0
max_i = 0
for i, v in enumerate(cos_sim_ls):
    if v > max_v and v <= 1.0:
        max_v = v
        max_i = i
print(f"{max_v} at index {max_i}") # 0.9286848902702332 at index 2069
cos_sim_ls = np.array(cos_sim_ls)

max_sim = np.argsort(cos_sim_ls)

samples = 100

np.random.seed(4)
rnd_sample = np.random.randint(1, 9000, samples)

mse = MeanSquaredError()

betas_target = mse(betas_unq[target,:], betas_unq[target,:])
betas_similar = []
for k, v in enumerate(max_sim[-samples:]):
    betas_similar.append( mse(betas_unq[v, :], betas_unq[target,:]) )
betas_sample = []
for k, v in enumerate(rnd_sample):
    betas_sample.append( mse(betas_unq[v, :], betas_unq[target,:]) )

fig = plt.figure(figsize=(15,10))
for k, v in enumerate(betas_similar):
    plt.plot(1, betas_similar[k], 'go', label='similar')
for k, v in enumerate(betas_sample):
    plt.plot(1, betas_sample[k], 'ro', label='random')
plt.plot(1, betas_target, 'bo', label='target')
#plt.title(f"NSD Key:\nTarget: {unq_keys[target]} | Similar: {unq_keys[max_sim[-20:]]} | Random: {unq_keys[rnd_sample]}")
plt.title(f"NSD Key:\nTarget: {unq_keys[target]}")
plt.legend()
plt.savefig(f"./betas_comparison_samples{samples}.png")
plt.close(fig)


