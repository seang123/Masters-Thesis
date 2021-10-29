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

ngroups = len(groups)
group_size = [len(g) for g in groups]
group_size.sort()
smallest = group_size[0]
largest = group_size[-1]

print("ngroups: ", ngroups)
print("smallest:", smallest)
print("largest: ", largest)

fake_data = np.random.uniform(0, 1, (glasser.shape))

visual_glasser = np.zeros(glasser.shape)
visual_glasser.fill(np.nan)
colour = 10
for g in groups:
    visual_glasser[g] = colour # fake_data[g] # colour
    colour += 10

vert = cortex.Vertex(visual_glasser, 'fsaverage')
im, extents = cortex.quickflat.make_flatmap_image(vert)


fig = plt.figure()
plt.imshow(im)
plt.savefig("./glasser.png")
plt.close(fig)



