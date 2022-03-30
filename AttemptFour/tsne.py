import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import time
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
    

def compute_tnse():

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

def captions(target: str):
    """ loop through caps and see if a target word is present in the caption """
    train_keys, val_keys, test_keys = loader.get_nsd_keys('2')
    pairs_t = loader.create_pairs(train_keys)

    pairs_t = [i[1] for i in pairs_t]
    pairs_t = [pairs_t[i:i+5] for i in range(0, 45000, 5)] # list(list(str))

    hits_idx = []
    for i in range(len(pairs_t)):
        hit = 0
        for k in range(5):
            cap = pairs_t[i][k].split(" ")
            for word in cap:
                if word == target:
                    hits_idx.append(i)
                    hit = 1
            if hit > 0:
                break
                    
    return hits_idx
    
def compute_pca(X):
    from sklearn.decomposition import PCA, TruncatedSVD
    start = time.time()
    #pca = PCA(n_components=1000, svd_solver='randomized')
    pca = TruncatedPCA(n_components=2).fit_transform(X)
    print(f"Time elapsed: {(time.time() - start):.3f}")
    return pca


def tnse_caption_analysis():
    """ """

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
    #X_embedded = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random').fit_transform(X)
    #print(X_embedded.shape)
    #X_embedded = compute_pca(X)

    # Load old tsne components
    X_embedded = np.load(open(f"./pca/train_betas_tnse_perplexity_30.npy", "rb"))

    print(X_embedded.shape)

    cap_idx = captions('giraffe') 
    cap_idx_surf = captions('surfboard')
    cap_idx_broc = captions('broccoli')
    cap_idx_veg = captions('vegetables')
    cap_idx_zebra = captions('zebra')

    fig = plt.figure(figsize=(16,9))
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c='darkgray')

    ## giraffe
    #for i in cap_idx:
    #    plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='orange')
    #plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'orange', label='giraffe')
    ## surf
    #for i in cap_idx_surf:
    #    plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='dodgerblue')
    #plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'dodgerblue', label='surfboard')
    ## broc
    for i in cap_idx_broc:
        plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='lime')
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'lime', label='broccoli')
    ## veg
    for i in cap_idx_veg:
        plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='yellow')
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'yellow', label='vegetables')
    # car
    for i in captions('car'):
        plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='orangered')
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'orangered', label='car')
    # bus
    for i in captions('bus'):
        plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='lightsalmon')
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'lightsalmon', label='bus')
    ## plane
    for i in captions('plane'):
        plt.scatter(X_embedded[i, 0], X_embedded[i,1], c ='steelblue')
    plt.scatter(X_embedded[i,0], X_embedded[i,1], c = 'steelblue', label='plane')

    plt.legend()
    plt.title("tSNE of fMRI betas into 2-D")
    plt.savefig(f'./tsne_giraffe.png', bbox_inches='tight')
    plt.close(fig)
    


if __name__ == '__main__':

    #captions("giraffe")
    tnse_caption_analysis()
