import nibabel as nb
import cortex
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse

"""

Look at which regions are active across time

- generate tables 

"""

parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=False)
parser.add_argument('--e', type=int, required=False)
parser.add_argument('--sub', type=str, required=False)
args = parser.parse_args()

model_name = 'proper_split_sub2'
epoch = 98
model_path = f"./Log/{model_name}/eval_out/attention_scores_{epoch}.npy"

if args.dir != None: model_name = args.dir
if args.e != None: epoch = args.e
if args.sub != None: subject = args.sub

region_names = pd.read_csv("./TrainData/unique_regions_list.csv")

import pylab
NUM_COLORS=15
cm = pylab.get_cmap('viridis')
## example of indexing into a colour map range
#for i in range(NUM_COLORS):
#    color = cm(1.*i/NUM_COLORS) 
#cmapgen = (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))

def load(fname):
    return np.squeeze(np.load(open(fname, "rb")), axis=-1)

def get_flatmap(glasser_regions: np.array):
    # Given a glasser_regions ndarray return the image to plot
    vert = cortex.Vertex(glasser_regions, subject='fsaverage')#, vmin=-8, vmax=8)
    im, extents = cortex.quickflat.make_flatmap_image(vert)
    return im, extents

def test_plot_fs(idx: list):

    GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
    GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
    VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

    glasser_lh = nb.load(GLASSER_LH).get_data()
    glasser_rh = nb.load(GLASSER_RH).get_data()
    glasser = np.vstack((glasser_lh, glasser_rh)).flatten() # (327684,)

    glasser_lh_flat = glasser_lh.flatten()
    glasser_rh_flat = glasser_rh.flatten()
    glasser_indices_rh = np.array(range(len(glasser_rh_flat)))
    groups_rh = []
    for i in set(glasser_rh_flat):
        groups_rh.append(glasser_indices_rh[glasser_rh_flat == i])
    glasser_indices_lh = np.array(range(len(glasser_lh_flat)))
    groups_lh = []
    for i in set(glasser_rh_flat):
        groups_lh.append(glasser_indices_lh[glasser_lh_flat == i])
    groups = groups_lh[1:] + groups_rh[1:]
    groups_lh = groups_lh[1:]
    groups_rh = groups_rh[1:]
    assert len(groups) == 360, "Using separate hemishere groups = 360"

    glasser_indices = np.array(range(len(glasser)))
    groups_concat = []
    for i in set(glasser):
        groups_concat.append(glasser_indices[glasser == i])
    groups_concat = groups_concat[1:]
    print("len(groups_concat):",  len(groups_concat))


    # Print regions
    for i in idx:
        print(i, "-", idx_to_region(i))
    idx = set(idx)

    # Plot
    glasser_regions_lh = np.zeros(glasser_lh.shape)
    glasser_regions_rh = np.zeros(glasser_rh.shape)
    glasser_regions_lh[:] = np.NaN
    glasser_regions_rh[:] = np.NaN
    c_l = 1
    c_r = 1
    for i, g in enumerate(groups_lh):
        if i in idx:
            glasser_regions_lh[g] = c_l
            c_l -= 0.1
        else:
            glasser_regions_lh[g] = 0
    for i, g in enumerate(groups_rh):
        if (i+180) in idx:
            glasser_regions_rh[g] = c_r
            c_r -= 0.1
        else:
            glasser_regions_rh[g] = 0

    glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

    fig = plt.figure(figsize=(16,9))
    im, _ = get_flatmap(glasser_regions)
    plt.imshow(im, cmap=plt.get_cmap('viridis'))
    plt.savefig("test_region_id.png")
    plt.close(fig)


def idx_to_region(idx) -> str:
    # Returns the appropriate region name for a given idx [0, 360]
    return region_names.iloc[idx].regionLongName


def active_regions(data):
    print(data.shape)

    data = np.log(data)
    data_mean_trials = np.mean(data, axis=0) # (15, 360)
    data_mean_trials_sort = np.argsort(data_mean_trials, axis=-1)
    print(data_mean_trials_sort.shape)

    #for i in range(5):
    #    print(data_mean_trials_sort[i, -10:-1])

    regions = []
    for i in range(1):
        print("Word:", i)
        for r in range(1, 16):
            idx = data_mean_trials_sort[i, -r] # 0 based idx 
            regions.append(idx)
            print("\t", r, "-", idx+1, "-", data_mean_trials[i, idx], "-",idx_to_region(idx))

    return regions

def region_to_idx(region: str) -> int:
    # Given a region name (str) return its idx [1, 360] inclusive
    return region_names['regionID'].loc[region_names['regionLongName'] == region].values[0]

def main():
    data = load(model_path)
    data1 = load(f"./Log/proper_split_sub1/eval_out/attention_scores_97.npy")
    #print("-- subject 1 --")
    #r = active_regions(data1)
    print("-- subject 2 --")
    r2 = active_regions(data)

    #print(set(r) - set(r2))

def test():
    data = load(model_path)
    data = np.log(data)
    data = np.mean(data, axis=0)
    data_idx = np.argsort(data[0, :])
    for i in range(len(data_idx)):
        print(i, data_idx[i], data[0, data_idx[i]])

if __name__ == '__main__':
    assert idx_to_region(0) == 'Primary_Visual_Cortex_L'
    assert idx_to_region(180) == 'Primary_Visual_Cortex_R'
    #test_plot_fs([0, 9, 10,180,189,190])
    main()
