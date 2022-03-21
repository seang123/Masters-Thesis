import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
import json
import numpy as np
from collections import defaultdict
import pandas as pd
from nsd_access import NSDAccess
import get_guse
from scipy.spatial.distance import cosine as cosine_distance
from tqdm import tqdm

# Perform a comparison between the inference output caption
# and the training captions.
# This is done by converting every caption to a guse embedding
# and computing the cosine distance between these and the inference 
# caption

def guse_comparison(captions: list ):
    """ Compare a list of captions to the training set captions

    Captions are converted to GUSE embeddings
    and a cosine distance is computed between them and all training
    captions
    """

    train_keys = pd.read_csv("./TrainData/subj02_conditions.csv")
    train_keys = train_keys.loc[train_keys['is_shared'] == 0]
    train_keys = train_keys.reset_index(drop=True)
    print(train_keys.head())

    train_guse = np.zeros((9000, 5, 512), dtype=np.float32)

    # 1. load training GUSE
    for i, v in enumerate(train_keys['nsd_key']):
        for j in range(5):
            with open(f"/fast/seagie/data/subj_2/guse/guse_embedding_KID{v}_CID{j}.npy", "rb") as f:
                train_guse[i, j, :] = np.load(f)
    print("Train GUSE loaded:", train_guse.shape)

    # 2. Convert candidate caption to guse
    test_guse = get_guse.embed_caption(captions)

    distance = {}
    for i in tqdm(range(9000)):
        for j in range(5):
            distance[i,j] = cosine_distance(test_guse[0,0,:], train_guse[i, j, :])

    min_dist = min(distance, key=distance.get)
    max_dist = max(distance, key=distance.get)
    distance = dict(sorted(distance.items(), key=lambda item: item[1]))

    for i, j in enumerate(distance):
        if i >= 3: break
        else:
            with open(f"/fast/seagie/data/subj_2/captions/SUB2_KID{train_keys['nsd_key'].iloc[min_dist[0]]}.txt", "r") as f:
                data = f.read().splitlines()
                print(f"Top {i} cosine distance: {distance[j]:.3f}\n\t{data[j[1]]}")

    with open(f"/fast/seagie/data/subj_2/captions/SUB2_KID{train_keys['nsd_key'].iloc[max_dist[0]]}.txt", "r") as f:
        data = f.read().splitlines()
        print("Max cosine distance:", max_dist, "\n\t", data[max_dist[1]])
    
    return
