import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import nibabel as nb
import cortex
from itertools import groupby

"""
Split betas into several masked regions
Previously I've been using jsut the Visual Cortex mask which outputs a (62756,) array.
Now I want to split that into separate glasser regions. 
"""

GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

glasser_lh = nb.load(GLASSER_LH).get_data()
glasser_rh = nb.load(GLASSER_RH).get_data()

glasser = np.vstack((glasser_lh, glasser_rh)).flatten() # (327684,)

print("glasser_lh", glasser_lh.shape)
print("glasser_rh", glasser_rh.shape)
print("glasser   ", glasser.shape)

visual_parcels = pd.read_csv(VISUAL_MASK, index_col=0)
visual_parcel_list = list(visual_parcels.values.flatten())

"""
# glasser has 181 unique values ranging from 0 - 180
groups = [list(grp) for k, grp in groupby(glasser)]
print("len groups", len(groups))
print(groups[0:10])

#groups = [i for i in groups if len(i) > 1]
#print("groups with length > 1", len(groups))


#raise Exception("stop")
"""

# All parcels
groups = []
glasser_indices = np.array(range(len(glasser)))
for i in set(glasser):
    group = glasser_indices[glasser == i]
    groups.append(group)

# Visual parcels
"""
groups = []
glasser_indices = np.array(range(len(glasser)))
for i in visual_parcel_list:
    group = glasser_indices[glasser==i]
    groups.append(group)
"""

# Remove group 0 which correspondes to some vertices along the corpus callosum 
groups = groups[1:]


ngroups = len(groups)
group_size = [len(g) for g in groups]
group_size.sort()
smallest = group_size[0]
largest = group_size[-1]

print("ngroups: ", ngroups)
print("smallest:", smallest)
print("largest: ", largest)
print("avgerage:", np.mean(group_size))
print("argmax : ", np.argmax(group_size))
print("group sizes: ", group_size[-10:])

fake_data = np.random.uniform(0, 1, (glasser.shape))

glasser_regions = np.zeros(glasser.shape)
#glasser_regions.fill(np.nan)
colour = 0 #10
for i, g in enumerate(groups):
    glasser_regions[g] = colour # fake_data[g] # colour
    colour += 1.4 #10

vert = cortex.Vertex(glasser_regions, 'fsaverage')
im, extents = cortex.quickflat.make_flatmap_image(vert)


fig = plt.figure()
plt.imshow(im, cmap=plt.get_cmap('viridis'))
plt.savefig("./glasser.png")
plt.close(fig)


def all_regions():
    """ Save an image showing every region separately """
    colour = 10
    for i, g in enumerate(groups):
        glasser_regions = np.zeros(glasser.shape)
        glasser_regions.fill(np.nan)
        glasser_regions[g] = colour # fake_data[g] # colour

        vert = cortex.Vertex(glasser_regions, 'fsaverage')
        im, extents = cortex.quickflat.make_flatmap_image(vert)

        fig = plt.figure()
        plt.imshow(im, cmap=plt.get_cmap('inferno'))
        plt.savefig(f"./glassers/region_i{i}_g{len(g)}.png")
        plt.close(fig)
        print(f"batch: {i}", end='\r')

#all_regions()
