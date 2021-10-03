
from nsd_access import NSDAccess
from load_dataset import load_dataset
import pandas as pd
import tensorflow as tf
import numpy as np
import umap
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


"""
Raw betas are (327684,) sized (approx. 78GB with 64 bit encoding)
Visual cortex is 'only' (62756,) (approx. 16GB)
"""


with open('masks/visual_mask_lh.npy', 'rb') as f, open('masks/visual_mask_rh.npy', 'rb') as g:
    visual_mask_lh = np.load(f)
    visual_mask_rh = np.load(g)
    print("> visual region masks loaded from file") 

visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
DIM = int(np.sum(visual_mask))


def apply_mask(x, mask):
    # Apply the visual area mask to the verticies
    return x[mask == 1]



dataset_unq = load_dataset("subj02", "unique", nparallel=54)#tf.data.experimental.AUTOTUNE)
dataset_shr = load_dataset("subj02", "shared", nparallel=54)#tf.data.experimental.AUTOTUNE)

# Apply the mask to keep only the visual cortex (62756,) instead of (327684,)
# The full betas would take up a little less than 40GB of space 
dataset_unq = dataset_unq.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_unq = dataset_unq.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))
# Apply mask to shared data
dataset_shr = dataset_shr.map(lambda a,b: (apply_mask(a, visual_mask),b))
dataset_shr = dataset_shr.map(lambda a,b: (tf.ensure_shape(a, shape=(DIM,)),b))

BUFFER_SIZE = 1000
BATCH_SIZE = 1024 #512

# Batch the data for quicker loading
dataset_test = dataset_shr.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
dataset_train = dataset_unq.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

shr_data = np.zeros((3000, 62756))
shr_indicies = []

c = 0
for i in dataset_test:
    betas, NSD_idx = i
    # betas = (256, 327684)
    # NSD_idx = (256,)
    shr_data[c:c+betas.shape[0]] = betas
    shr_indicies.append(NSD_idx)     

    c += betas.shape[0]
    print(f"Shr: {c}", end='\r')
print()

# flatten
shr_indicies = [i for sublist in shr_indicies for i in sublist]

"""
#with open(f"./betas/betas_shr_vc_nozscore.npy", "wb") as f:
with open(f"/huge/seagie/betas/betas_shr_vc_nozscore.npy", "wb") as f:
    np.save(f, shr_data)

#with open(f"./betas/betas_shr_vc_keys_nozscore.txt", "w+") as f:
with open(f"/huge/seagie/betas/betas_shr_vc_keys_nozscore.txt", "w+") as f:
    for k in shr_indicies:
        f.write(f"{k}\n")
"""

## Save the training set | Unique betas
 
"""
unq_data = np.zeros((27000, 62756))
unq_indicies = []
c = 0
for i in dataset_train:
    betas, NSD_idx = i
    # betas = (256, 327684)
    # NSD_idx = (256,)
    #unq_data.append(betas) 
    unq_data[c:c+betas.shape[0]] = betas
    unq_indicies.append(NSD_idx)     

    c += betas.shape[0]
    print(f"Unq: {c}", end='\r')
print()

# flatten
unq_indicies = [i for sublist in unq_indicies for i in sublist]
"""

"""
#with open(f"./betas/betas_unq_vc_nozscore.npy", "wb") as f:
with open(f"/huge/seagie/betas/betas_unq_vc_nozscore.npy", "wb") as f:
    np.save(f, unq_data)

#with open(f"./betas/betas_unq_vc_keys_nozscore.txt", "w+") as f:
with open(f"/huge/seagie/betas/betas_unq_vc_keys_nozscore.txt", "w+") as f:
    for k in unq_indicies:
        f.write(f"{k}\n")
"""

_min = min(shr_indicies)
_max = max(shr_indicies)

colours = np.array([ (i - _min)/(_max - _min) for i in shr_indicies ])
colours = np.expand_dims(colours, axis=1)

fit = umap.UMAP(n_neighbors = 15)
u = fit.fit_transform(shr_data)

fig = plt.figure()
plt.scatter(u[:,0], u[:,1], c = colours)
plt.savefig("./umap_shr_data_15.png")
plt.close(fig)



print("Done.")




