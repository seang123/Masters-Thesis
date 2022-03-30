import nibabel as nb
import cortex
import json
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
import tqdm
from nsd_access import NSDAccess
import tensorflow as tf
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

import pylab
NUM_COLORS=15
cm = pylab.get_cmap('viridis')
## example of indexing into a colour map range
#for i in range(NUM_COLORS):
#    color = cm(1.*i/NUM_COLORS) 

log_dir = './Log/'
model = 'proper_split_sub2'
epoch = 98

#model = 'proper_split_sub2_attn_loss'
#epoch = 118

model_path = f'{log_dir}/{model}/eval_out/'
tokenizer_loc = f"{model_path}/tokenizer.json"
attention_path = f'{model_path}/attention_scores_{epoch}.npy'
captions_path  = f'{model_path}/output_captions_{epoch}.npy'

## Load tokenizer
with open(tokenizer_loc, "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
print("Tokenizer loaded ...")


def load(path):
    with open(path, 'rb') as f:
        data = np.load(f)
    return np.squeeze(data, axis=-1)

def remove_pad(x):
    if type(x) == str:
        x = x.split(" ")
    x = [i for i in x if i != '<pad>' and i != '<end>']
    return " ".join(x)

def compute_bleu(captions: np.array, targets: dict, keys: np.array, b: int):

    chencherry = SmoothingFunction()
    weights = [
        (1./1., 0., 0., 0.),
        (1./2., 1./2., 0., 0.),
        (1./3., 1./3., 1./3., 0.),
        (1./4., 1./4., 1./4., 1./4.)     
    ]
    scores = {}

    references = []
    hypothesis = []
    for i, key in enumerate(keys):
        ref = [k.split(" ") for k in targets[key]]
        hyp = list(captions[i])
        hyp = remove_pad(hyp).split(" ")
        references.append(ref)
        hypothesis.append(hyp)
    
    for i, key in enumerate(keys):
        b_score = sentence_bleu(references[i], hypothesis[i], weights=weights[b], smoothing_function=chencherry.method0)
        scores[key] = b_score

    return scores

def load_captions_dict(keys: list):
    path = '/fast/seagie/data/captions/'
    captions = defaultdict(list)
    for i, key in enumerate(keys):
        with open(f"{path}/KID{key}.txt", "r") as f:
            content = f.read()
            for i in content.splitlines():
                cap = i.replace(".", " ").replace(",", " ").strip().split(" ")
                cap = [i.lower() for i in cap if i != '']
                cap = " ".join(cap)
                captions[key].append(cap)

    return captions
                

def main():
    caps = load(captions_path) # -> (515, 15)
    captions = tokenizer.sequences_to_texts(caps)
    captions = np.array([i.split(" ") for i in captions])
    #captions = np.reshape(captions, captions.shape[0] * captions.shape[1]) # flatten captions
    print(captions.shape)

    train_set = pd.read_csv("./TrainData/test_conditions.csv")['nsd_key']
    df = pd.read_csv("~/NSD3/nsddata/ppdata/subj02/behav/responses.tsv", sep='\t')

    df = df.groupby('73KID', as_index=False).sum()
    iscorr = df['ISCORRECT'].values

    temp = dict(zip(df['73KID'], df['ISCORRECT'])) # {nsd_key: hit_rate}
    nsd_hit_rate = {}
    for k,v in temp.items():
        if k in set(train_set):
            nsd_hit_rate[k] = v

    ## Compute BLEU for each 
    targets = load_captions_dict(train_set.values)

    #blue = compute_bleu(captions, targets, train_set.values, b=0)

    hit_rate_nsd = defaultdict(list)
    for k,v in nsd_hit_rate.items():
        hit_rate_nsd[v].append(k)

    """
    fig = plt.figure(figsize=(16,9))
    for i in range(4):
        keys = hit_rate_nsd[i]
        for k in keys:
            plt.scatter(i, blue[k], color='darkgray', alpha=0.5)
    plt.xlabel("Hit rate")
    plt.ylabel("BLEU-4")
    plt.title("Hit rate vs. BLEU-4 score")
    plt.savefig("bleu_hit_rate.png", bbox_inches='tight')
    plt.close(fig)
    """

    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    b_scores = [1, 4]
    for s in range(len(b_scores)):
        data = []
        blue = compute_bleu(captions, targets, train_set.values, b=s)
        for i in range(4):
            keys = hit_rate_nsd[i]
            temp = []
            for k in keys:
                temp.append( blue[k] )
            data.append(temp)
        ax[s].boxplot(data, notch=False)
        ax[s].set_title(f"BLEU-{b_scores[s]}")
        ax[s].set_ylabel("BLEU score")
        ax[s].set_xticklabels([i for i in range(4)])
        ax[s].set_xlabel("hit rate")
    plt.suptitle("Hit rate vs. BLEU score")
    plt.savefig("bleu_hit_rate.png", bbox_inches='tight')
    plt.close(fig)


def compare_subjects():

    df1 = pd.read_csv("~/NSD3/nsddata/ppdata/subj01/behav/responses.tsv", sep='\t')
    df2 = pd.read_csv("~/NSD3/nsddata/ppdata/subj02/behav/responses.tsv", sep='\t')
   
    iscorr1 = df1.groupby('73KID', as_index=False).sum()['ISCORRECT'].values
    iscorr2 = df2.groupby('73KID', as_index=False).sum()['ISCORRECT'].values

    print(sum(iscorr1))
    print(sum(iscorr2))
    

    return

if __name__ == '__main__':
    #main()
    compare_subjects()
