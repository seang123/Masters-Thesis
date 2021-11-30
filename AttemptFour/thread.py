import os, sys
import concurrent.futures
import numpy as np
import pandas as pd
import re
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import collections 
import yaml
import nibabel as nb
import DataLoaders.load_avg_betas as loader
import multiprocessing

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
keys = np.concatenate((unq_keys, shr_keys))
print("keys:", keys.shape)


def timeit(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}() - {(end - start):.2f} sec")
        return result
    return wrap 

@timeit
def singlethread():
    betas = np.zeros((10000, 327684), dtype=np.float32)
    for i, key in enumerate(keys):
        with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
            betas[i, :] = np.load(f)
    print("betas:", betas.shape)

@timeit
def multithread():
    def load_betas(key):
        with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
            return np.load(f)

    betas = np.zeros((10000, 327684), dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results_betas = executor.map(load_betas, keys)

    #start = time.time()
    #for k, v in enumerate(results_betas):
    #    betas[k,:] = v
    #print(f"elapsed: {(time.time() - start):.2f}")

def load_betas(key):
    with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
        return np.load(f)

@timeit
def parallel_read():

    pool = multiprocessing.Pool(processes=2)
    x_list = pool.map(load_betas, keys)


if __name__ == '__main__':

    #parallel_read()

    #singlethread()

    multithread()

    """
    start = time.time()
    multithread2()
    print(f"elapsed: {(time.time() - start):.2f}")
    """
