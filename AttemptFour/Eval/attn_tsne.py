import nibabel as nb
import cortex
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn import manifold
from tensorflow.keras.preprocessing.text import tokenizer_from_json


model = 'no_attn_loss_const_lr2'
model_path = f'./Log/{model}/eval_out/'
tokenizer_loc = f"./Log/{model}/eval_out/tokenizer.json"

with open(tokenizer_loc, "r") as f:
    tokenizer = tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

with open(f"{model_path}/attention_scores_41.npy", "rb") as f:
    data = np.squeeze(np.load(f), axis=-1)
    print("data:", data.shape)

with open(f"{model_path}/output_captions_41.npy", "rb") as f:
    caps = np.squeeze(np.load(f), axis=-1)
    print("caps", caps.shape)
    caps = tokenizer.sequences_to_texts(caps) # -> list
    caps = [i.split(" ") for i in caps]
    caps = np.array(caps)
    caps = np.reshape(caps, (caps.shape[0] * caps.shape[1]))
    print(caps.shape)

x,y,z = data.shape
data = np.reshape(data, (x*y, z))
print(data.shape)

def tsne():
    print("-- computing tSNE --")
    for p in [30]:
        print("perplexity:", p)
        X_emb = TSNE(n_components=3, perplexity=p, learning_rate='auto', init='pca').fit_transform(data)

        print(X_emb.shape)

        idx = []
        for k, v in enumerate(caps):
            if v == 'giraffe':
                idx.append(k)


        fig = plt.figure(figsize=(16,9))
        plt.scatter(X_emb[:,0], X_emb[:,1], c=X_emb[:,2], cmap=plt.cm.viridis)

#        for i in idx:
#            plt.scatter(X_emb[i,0], X_emb[i,1], color = 'green')

        plt.savefig(f'./attn_tsne_{p}.png')
        plt.close(fig)

def lle():
    method = 'standard'

    X_emb = manifold.LocallyLinearEmbedding(
            n_neighbors=10, n_components=2, method=method
    ).fit_transform(data)

    print(X_emb.shape)

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X_emb[:,0], X_emb[:,1])#, c=X_embedded[:,2], cmap=plt.cm.viridis)
    plt.savefig(f'./attn_lle.png')
    plt.close(fig)


if __name__ == '__main__':
    tsne()
    #lle()


