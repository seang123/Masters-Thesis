import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os, sys
sys.path.append('~/NSD/Code/Masters-Thesis/ThinkAndTell/')
sys.path.append('~/NSD/Code/MAsters-Thesis/')
from load_dataset import load_dataset
import cortex
from sklearn.decomposition import PCA
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# Singular value decomposition of the betas
# Or maybe PCA :)

"""
https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
"""

if os.path.exists("../masks/visual_mask_rh.npy"):
    with open('../masks/visual_mask_lh.npy', 'rb') as f, open('../masks/visual_mask_rh.npy', 'rb') as g:
        visual_mask_lh = np.load(f)
        visual_mask_rh = np.load(g)
        print(" > visual region masks loaded from file") 


visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))


def apply_mask(x, mask):
    # Apply the visual area mask to the verticies
    # before mask betas = (327684,)
    return x[mask == 1]


#### LOAD DATASET ####

#dataset_shr = load_dataset("subj02", "shared", nparallel=tf.data.experimental.AUTOTUNE)

#dataset_shr = dataset_shr.map(lambda a,b,c: (apply_mask(a, visual_mask), b,c))
#dataset_shr = dataset_shr.map(lambda a,b,c: (tf.ensure_shape(a, shape=(DIM,)),b,c))

#betas_ls = np.zeros((3000,62756))

#c = 0
#for element in dataset_shr.as_numpy_iterator():
#    c += 1
#    b = element[0]
#    betas_ls[c-1, :] = b
#    print(c, end='\r')
    
with open("/fast/seagie/betas_subj02_shr.npy", "rb") as f:
    betas_shr = np.load(f)

with open("/fast/seagie/betas_subj02_unq.npy", "rb") as f:
    betas_unq = np.load(f)

print("betas", betas_unq.shape, betas_shr.shape)

n_components = 5000
print(f"Fitting on {n_components} using randomized svd solver")

start = time.time()
pca = PCA(n_components=n_components, svd_solver="randomized")
pca.fit(betas_unq)

print(f"elapsed time: {(time.time() - start):.2f} sec")

print(f"{n_components} components maintain {np.sum(pca.explained_variance_ratio_)} of the variance")

np.save("./data/pca_sing_values.npy", pca.singular_values_)
np.save("./data/pca_expl_var_ratio.npy", pca.explained_variance_ratio_)
np.save("./data/pca_expl_var.npy", pca.explained_variance_)
np.save("./data/pca_components.npy", pca.components_)
print("PCA data saved")

print("transforming data...")

new_betas_shr = pca.transform(betas_shr)

new_betas_unq = pca.transform(betas_unq)

with open("./data/pca_subj02_betas_shr_vc.npy", "wb") as f:
    np.save(f, new_betas_shr)

with open("./data/pca_subj02_betas_unq_vc.npy", "wb") as f:
    np.save(f, new_betas_unq)

print("PCA reduced betas data saved to disk")

"""
Conversion process

Keep : 5000 components

27000 * 5000 + 3000 * 5000 (unq, shr) datasets
"""




print("done.")
