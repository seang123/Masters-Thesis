import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
from DataLoaders import load_avg_betas as loader
import numpy as np
import nibabel as nb
import cortex

captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path    = "/huge/seagie/data/subj_2/betas_averaged/"

GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

## Load glasser regions
glasser_lh = nb.load(GLASSER_LH).get_data()
glasser_rh = nb.load(GLASSER_RH).get_data()

glasser = np.vstack((glasser_lh, glasser_rh)).flatten() # (327684,)

print("glasser_lh", glasser_lh.shape)
print("glasser_rh", glasser_rh.shape)
print("glasser   ", glasser.shape)



unq_keys, shr_keys = loader.get_nsd_keys("")
print(unq_keys.shape)
print(shr_keys.shape) 

all_keys = np.concatenate((unq_keys, shr_keys))
print(all_keys.shape)

## Load the betas
betas = np.zeros((all_keys.shape[0], 327684), dtype=np.float32)
for i, key in enumerate(all_keys):
    betas[i,:] = np.load(open(f"{betas_path}/subj02_KID{key}.npy", "rb"))

#betas_sum = np.sum(betas, axis=0)
#betas_mean = np.mean(betas, axis=0)

betas_mag = np.sqrt(np.sum(np.power(betas, 2),axis=0))

def vert(betas):
    vert = cortex.Vertex(betas, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)
    return im

im = vert(betas_mag)


fig = plt.figure()
plt.imshow(im)
plt.savefig("./magnitude_betas.png")
plt.close(fig)


