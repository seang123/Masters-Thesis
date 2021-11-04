import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import sys
sys.path.append("/home/seagie/NSD/Code/Masters-Thesis/AttemptFour/")
from DataLoaders import load_avg_betas as loader
import numpy as np
import nibabel as nb
import cortex

captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path    = "/huge/seagie/data/subj_2/betas_averaged/"
out_dir = "./Visualization"

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
#betas = np.zeros((all_keys.shape[0], 327684), dtype=np.float32)
#for i, key in enumerate(all_keys):
#    betas[i,:] = np.load(open(f"{betas_path}/subj02_KID{key}.npy", "rb"))

# unique
betas_unq = np.zeros((unq_keys.shape[0], 327684), dtype=np.float32)
for i, key in enumerate(unq_keys):
    betas_unq[i,:] = np.load(open(f"{betas_path}/subj02_KID{key}.npy", "rb"))
# shared
betas_shr = np.zeros((shr_keys.shape[0], 327684), dtype=np.float32)
for i, key in enumerate(shr_keys):
    betas_shr[i,:] = np.load(open(f"{betas_path}/subj02_KID{key}.npy", "rb"))


betas_all = np.concatenate((betas_unq, betas_shr), axis=0)
print("betas_all", betas_all.shape)

#betas_sum = np.sum(betas_all, axis=0)
#betas_mean = np.mean(betas_all, axis=0)


#betas_mag_unq = np.sum(np.power(betas_unq, 2),axis=0)
#betas_mag_shr = np.sum(np.power(betas_shr, 2),axis=0)
#betas_mag_all = np.sum(np.power(betas_all, 2),axis=0)

#betas_mag_unq = np.divide(betas_mag_unq, betas_unq.shape[0])
#betas_mag_shr = np.divide(betas_mag_shr, betas_shr.shape[0])
#betas_mag_all = np.divide(betas_mag_all, betas_all.shape[0])

def vert(betas):
    vert = cortex.Vertex(betas, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)
    return im
"""
im_unq = vert(betas_mag_unq)
im_shr = vert(betas_mag_shr)
im_all = vert(betas_mag_all)
"""

im_single = vert(betas_all[10,:])

def save_fig(img, file_name):

    fig = plt.figure()
    plt.imshow(img)# cmap='hot')
    plt.colorbar()
    plt.savefig(file_name)
    plt.close(fig)

def ten_rnd_avg():
    """ Plot 10 random average betas from train/val """

    unq_sample = betas_unq[np.random.randint(0, betas_unq.shape[0], size=100), :]
    shr_sample = betas_shr[np.random.randint(0, betas_shr.shape[0], size=100), :]

    unq_mean = np.mean(np.power(unq_sample,2), axis=0)
    shr_mean = np.mean(np.power(shr_sample,2), axis=0)

    save_fig(vert(unq_mean), f"{out_dir}/unq_mean_10.png")
    save_fig(vert(shr_mean), f"{out_dir}/shr_mean_10.png")

if __name__ == '__main__':
    ten_rnd_avg()

    """
    save_fig(vert(betas_mag_unq), "./magnitude_betas_unq.png")
    save_fig(vert(betas_mag_shr), "./magnitude_betas_shr.png")
    save_fig(vert(betas_mag_all), "./magnitude_betas_all.png")
    save_fig(im_single, "./single_betas.png")
    """
