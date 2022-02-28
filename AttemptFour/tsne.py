import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from DataLoaders import load_avg_betas as loader


def plot():
    from mpl_toolkits import mplot3d
    from scipy import stats

    with open(f"./train_betas_tnse_perplexity_{30}_3d.npy", "rb") as f:
        data = np.load(f)
        print("data:", data.shape)

    z = np.abs(stats.zscore(data))
    z = np.where(z < 0.1)
    data = data[z[0]]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c = data[:,2], cmap='Greens')
    plt.savefig("./3d_tnse.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16,9))
    plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=plt.cm.viridis)
    plt.savefig(f'./2d_tnse.png')
    plt.close(fig)
    

perplexities = [30] # [30, 50, 100]

# Load caption ints
with open(f"./Log/no_attn_loss_const_lr2/eval_out/output_captions.npy", "rb") as f:
    X = np.squeeze(np.load(f), axis=-1)
print(X.shape)

def load_betas():
    # Load betas
    betas_path = "/fast/seagie/data/subj_2/betas_averaged/"
    train_keys, val_keys = loader.get_nsd_keys('2')
    betas = np.zeros((9000, 327684), dtype=np.float32)
    betas_val = np.zeros((1000, 327684), dtype=np.float32)
    for i, key in enumerate(train_keys):
        with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
            betas[i,:] = np.load(f)
    for i, key in enumerate(val_keys):
        with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
            betas_val[i,:] = np.load(f)
    return betas, betas_val

X, Y = load_betas()

for p in perplexities:
    print(f"Computing TSNE with perplexity: {p}")
    X_embedded = TSNE(n_components=2, perplexity=p, learning_rate='auto', init='random').fit_transform(X)

    print(X_embedded.shape)

    with open(f"./pca/train_betas_tnse_perplexity_{p}_1d.npy", "wb") as f:
        np.save(f, X_embedded)

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X_embedded[:,0], X_embedded[:,1])#, c=X_embedded[:,2], cmap=plt.cm.viridis)
    plt.savefig(f'./pca/train_betas_tsne_perplexity_{p}_1d.png')
    plt.close(fig)
