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

from matplotlib import colors as mcolors
#colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colours = [k for (k,v) in mcolors.CSS4_COLORS.items()]

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
#attention_path_weights = f'{model_path}/attention_weights_{epoch}.npy'
captions_path  = f'{model_path}/output_captions_{epoch}.npy'

## Load tokenizer
with open(tokenizer_loc, "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

def load(path):
    with open(path, 'rb') as f:
        data = np.load(f)
    return np.squeeze(data, axis=-1)


def tsne(data, perplexity=30, init='pca'):

    #x, y, z = data.shape
    #data = np.reshape(data, (x*y, z))

    s = time.time()
    X = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            learning_rate='auto', 
            init=init, 
            method='barnes_hut', 
            metric='euclidean'
    ).fit_transform(data) # minkowski metric
    #with open("attn_maps_tsne.npy", "rb") as f:
    #    X = np.load(f)
    #    np.save(f, X)
    print(f"tSNE - time elapsed - {(time.time() - s):.2f}")
    return X


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
            counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    
def cluster(data, n_clusters):
    """ Clustering 
    
    data : ndarray
        3d attention scores
    """

    x,y,z = data.shape
    data = data.reshape(x * y, z)

    #linkage: ward’, ‘complete’, ‘average’, ‘single’

    s = time.time()
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='ward',
        compute_distances=True,
    ).fit(data)
    #clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    print(f"clustering - time elapsed - {(time.time() - s):.2f}")
    return clustering

def plot_tsne(X, captions):

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X[:,0], X[:,1], color='darkgray')

    # --
    """
    for k in range(15):
        plt.scatter(X[k,0], X[k,1], color='red')
    print("red:", captions[0:15])
    # --
    for k in range(15):
        plt.scatter(X[k+15,0], X[k+15,1], color='green')
    print("green:", captions[15:30])
    # --
    for k in range(15):
        plt.scatter(X[k+30,0], X[k+30,1], color='orange')
    print("orange:", captions[30:45])
    # --
    for k in range(15):
        plt.scatter(X[k+45,0], X[k+45,1], color='blue')
    print("blue:", captions[45:60])
    # --
    for k in range(15):
        plt.scatter(X[k+60,0], X[k+60,1], color='cyan')
    print("cyan:", captions[60:75])
    """
    def find_idx_word(word):
        idx = []
        for i in range(captions.shape[0]):
            if captions[i] == word:
                idx.append( i )
        return idx

    end = find_idx_word('<end>')
    pad = find_idx_word('<pad>')
    a   = find_idx_word('a')
    giraffe = find_idx_word('giraffe')

    plt.scatter(X[end,0], X[end,1], label='<end>')
    plt.scatter(X[pad,0], X[pad,1], label='<pad>')
    plt.scatter(X[giraffe,0], X[giraffe,1], label='giraffe')
    plt.scatter(X[a,0], X[a,1], label='a')

    #for k in range(15):
    #    xx = range(k, 515*15, 15)
    #    plt.scatter(X[xx,0], X[xx,1], label=f"word:{k}")

    plt.legend()
    plt.title(f"2D tSNE of attention maps")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.savefig(f"attention_maps_tsne.png", bbox_inches='tight')
    plt.close(fig)

def tsne_no_pads(data, captions):
    """ remove the <pad> tokens before doing tsne """

    x,y,z = data.shape
    data = data.reshape(x * y, z)

    captions = tokenizer.sequences_to_texts(captions)
    captions = np.array([i.split(" ") for i in captions])
    captions = np.reshape(captions, captions.shape[0] * captions.shape[1]) # flatten captions

    data = data[np.where(np.logical_and(captions != '<pad>', captions != 'a'))]
    captions = captions[np.where(np.logical_and(captions != '<pad>', captions != 'a'))]
    print(data.shape)

    X = tsne(data, 50)

    def word_idx(word):
        ls = []
        for i in range(len(captions)):
            if captions[i] == word:
                ls.append(i)
        return ls

    data_ls = list(captions)
    data_set = set(data_ls)
    print("unique words in captions:", len(data_set))

    """
    greater = np.where(X[:,0] > 75)
    print(list(set(captions[greater])))
    print("----") 
    top = np.where(np.logical_and(X[:,0]<=0, X[:,0]<=20))
    print(top)
    top2 = np.where(X[:,1] > 55)
    print(top2)
    top = np.intersect1d(top, top2)
    print(top)
    print(list(set(captions[top])))
    """

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X[:,0], X[:,1], alpha = 0.5, color='darkgray')
    for i in ['man', 'woman', 'riding', 'surfboard', 'the']:
        word_i = word_idx(i)
        plt.scatter(X[word_i,0], X[word_i,1], alpha=0.5, label=i)
    plt.legend()
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.savefig("no_pads_or_a.png", bbox_inches='tight')
    plt.close(fig)

    return

def across_time_tsne():

    data = load(attention_path) # -> (515, 15, 360)

    x,y,z = data.shape
    data = data.reshape((x * y, z))
    X = tsne(data, 50, 'pca')
    #with open("attn_maps_tsne.npy", "rb") as f:
    #    X = np.load(f)

    ii = X[list(range(0, x*y, 15)),0]
    kk = X[list(range(1, x*y, 15)),0]
    print(ii[:10])
    print(kk[:10])

    print("15//2 =", 15//2)
    fig, ax = plt.subplots(8, 1, figsize=(20,40))

    ii = 0
    for i in range(0,15,2):
        pos = list(range(i, x*y, 15))
        ax[ii].scatter(X[:,0], X[:,1], color = 'darkgray')
        ax[ii].scatter(X[pos,0], X[pos,1], color='firebrick')
        ax[ii].set_title(f'timestep: {i}')
        ii += 1
    
    plt.savefig("time_tsne_attnweights.png", bbox_inches='tight')
    plt.close(fig)

def word_count(captions):

    from collections import Counter
    caps = list(captions)
    print("nr of words:", len(caps))
    counts = Counter(caps)
    print("nr of unique words:", len(counts))
    return
    

def main():
    
    data = load(attention_path) # -> (515, 15, 360)
    caps = load(captions_path) # -> (515, 15)
    print(data.shape)
    print(caps.shape)
    captions = tokenizer.sequences_to_texts(caps)
    captions = np.array([i.split(" ") for i in captions])
    captions = np.reshape(captions, captions.shape[0] * captions.shape[1]) # flatten captions

    word_count(captions)

    #across_time_tsne(data)
    #raise

    tsne_no_pads(data, caps)
    raise

    ## compute tSNE
    X = tsne(data, 50)

    ## clustering
    n_clusters = 15
    clusters = cluster(data, n_clusters)

    fig = plt.figure()
    plot_dendrogram(clusters, truncate_mode="level", p=3)
    plt.savefig("dendogram.png")
    plt.close(fig)

    #print(set(captions[np.where(clusters.labels_ == 0)[0]]))

    ## Plot tSNE
    plot_tsne(X, captions)
    sys.exit(0)

    ## Plot Clustered tSNE
    fig = plt.figure(figsize=(16,9))

    for i in range(n_clusters):
        idx = np.where(clusters.labels_ == i)
        c = cm(1.*i/n_clusters)
        plt.scatter(X[idx,0], X[idx,1], color = c)
    
    plt.title(f"2D tSNE of attention maps clustered into {n_clusters} groups")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.savefig(f"attention_maps_clustered_{n_clusters}.png", bbox_inches='tight')
    plt.close(fig)


def attn_variance():

    data = load(attention_path) # -> (515, 15, 360)
    print("attn:", data.shape)

    y = []
    for i in range(515):
        y.append( np.var(data[i,:,:], axis=1) )
    y = np.array(y)

    y_mean = np.mean(y, axis=0) # mean variance across timesteps

    y2 = np.std(y, axis=0) # std deviation error
    print(np.array(y).shape)
    print(y2.shape)

    fig = plt.figure(figsize=(16,9))
    for i in range(len(y)):
        plt.plot(y[i], label=i, color='darkgray', alpha = 0.75)
#    plt.legend()
    plt.errorbar(np.arange(len(y_mean)), y_mean, yerr=y2, capsize=10, color='black')
    #plt.plot(y_mean, label='mean', color='black')
    plt.title("Variance of a brain region across trials for each word")
    plt.xlabel("brain region")
    plt.ylabel("variance")
#    plt.ylim(ymax=0.5e-5)
    plt.savefig("attn_variance.png", bbox_inches='tight')
    plt.close(fig)

    """
    fig = plt.figure(figsize=(16,9))
    plt.errorbar(np.arange(len(y_mean)), y_mean, yerr=y2, capsize=10)
    plt.title("Variance between brain distributions at each word position")
    plt.xlabel("word position")
    plt.ylabel("variance")
    plt.savefig("some_other_var.png")
    plt.close(fig)
    """
    return

def diff_perplexity():
    print("path:", attention_path)
    data = load(attention_path) # -> (515, 15, 360)
    print("attn:", data.shape)

    x,y,z = data.shape
    data = np.reshape(data, (x*y, z))

    caps = load(captions_path) # -> (515, 15)
    captions = tokenizer.sequences_to_texts(caps)
    captions = np.array([i.split(" ") for i in captions])
    captions = np.reshape(captions, captions.shape[0] * captions.shape[1]) # flatten captions
    print("caps:", captions.shape)
    print(" ".join(captions[0:15]))

    per = 2
    X = tsne(data, perplexity=per, init='random')

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X[:,0], X[:,1], color='darkgray')

    for i in range(15):
        plt.scatter(X[i,0], X[i,1], color='red')

    def find_idx_word(word):
        idx = []
        for i in range(captions.shape[0]):
            if captions[i] == word:
                idx.append( i )
        return idx
    ida = find_idx_word('man')
    idb = find_idx_word('wave')
    idc = find_idx_word('surfboard')
    plt.scatter(X[ida,0], X[ida,1], color='green', label='man')
    plt.scatter(X[idb,0], X[idb,1], color='blue', label='wave')
    plt.scatter(X[idc,0], X[idc,1], color='orange', label='surfboard')

    plt.legend()
    plt.title("tSNE of attention maps")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.savefig(f"attn_tsne_p{per}_rnd.png", bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    across_time_tsne()
    #attn_variance()
    #main()
    #diff_perplexity()
