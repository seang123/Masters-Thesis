import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import time

def load_betas():
    from DataLoaders import load_avg_betas as loader
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

def attn_scores_pca():
    with open(f"./Log/no_attn_loss_const_lr2/eval_out/attention_scores.npy", "rb") as f:
        attn = np.squeeze(np.load(f), axis=-1) # (1000, 15, 360)
        attn = np.mean(attn, axis=1)
        print(attn.shape)

    pca = PCA(n_components=40)
    pca.fit(attn)
    print("expl. var. ratio:", pca.explained_variance_ratio_)
    print("expl. var. ratio cumsum:", np.cumsum(pca.explained_variance_ratio_))

def compute_pca():
    X, _ = load_betas()
    start = time.time()
#pca = PCA(n_components=1000, svd_solver='randomized')
    pca = TruncatedSVD(n_components=5000)
    pca.fit(X)
    print(f"Time elapsed: {(time.time() - start):.3f}")

    np.save("./pca/pca_sing_values.npy", pca.singular_values_)
    np.save("./pca/pca_expl_var_ratio.npy", pca.explained_variance_ratio_)
    np.save("./pca/pca_expl_var.npy", pca.explained_variance_)
    np.save("./pca/pca_components.npy", pca.components_)
    print("PCA data saved")

def var_explained():
    x = np.load("./pca/pca_expl_var_ratio.npy")

    cs = np.cumsum(x)
    print(f"1 components: {cs[0]}") 
    print(f"5 components: {cs[5]}") 
    print(f"500 components: {cs[500]}") 
    print(f"5000 components: {cs[-1]}") 

    fig = plt.figure()
    plt.plot(np.cumsum(x))
    plt.savefig("./cumsum_pca.png")
    plt.close(fig)

if __name__ == '__main__':
    #compute_pca()
    #var_explained()
    attn_scores_pca()




