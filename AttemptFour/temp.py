import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import DataLoaders.load_avg_betas as loader
from scipy import stats
import sys
import nibabel as nb
import cortex

from nsd_access import NSDAccess



with open("./Log/separate_hemispheres_attn_loss/eval_out/attention_scores.npy", "rb") as f:
    attn_scores_loss = np.load(f)
with open("./Log/separate_hemispheres_no_attn_loss/eval_out/attention_scores.npy", "rb") as f:
    attn_scores_no_loss = np.load(f)

attn_scores_loss = np.log(np.squeeze(attn_scores_loss, axis=-1))
attn_scores_no_loss = np.log(np.squeeze(attn_scores_no_loss, axis=-1))
print(attn_scores_loss.shape)
print(attn_scores_no_loss.shape)

print("mean across trials")
attn_scores_loss = np.mean(np.mean(attn_scores_loss, axis=0), 0)
attn_scores_no_loss = np.mean(np.mean(attn_scores_no_loss, axis=0), 0)
print(attn_scores_loss.shape)
print(attn_scores_no_loss.shape)

t_zero_loss = np.argsort(attn_scores_loss[:])
t_zero_no_loss = np.argsort(attn_scores_no_loss[:])

print(t_zero_loss[:25])
print(t_zero_no_loss[:25])

score = 0
for i in range(len(t_zero_loss)):
    if t_zero_loss[i] == t_zero_no_loss[i]:
        score += 1
print("score:", score, score/len(t_zero_loss))

raise

conditions = pd.read_csv(f"./TrainData/subj02_conditions.csv")
val_keys = conditions.loc[conditions['is_shared'] == 1]
val_keys = val_keys.reset_index(drop=True)

key = val_keys['nsd_key'].iloc[0]
print("key:", key)




GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

glasser_lh = nb.load(GLASSER_LH).get_data()
glasser_rh = nb.load(GLASSER_RH).get_data()
glasser = np.vstack((glasser_lh, glasser_rh)).flatten() # (327684,)

groups = []
glasser_indices = np.array(range(len(glasser)))
for i in set(glasser):
    group = glasser_indices[glasser == i]
    groups.append(group)
groups = groups[1:]

ngroups = len(groups)
assert ngroups == 180

region_id = 0

glasser_regions = np.zeros(glasser.shape)
#for i, g in enumerate(groups):
#    glasser_regions[g] = data[i]
glasser_regions[groups[region_id]] = 1
glasser_regions[groups[1]] = 2
glasser_regions[groups[2]] = 3

vert = cortex.Vertex(glasser_regions, 'fsaverage')
im, extents = cortex.quickflat.make_flatmap_image(vert)

fig = plt.figure()
plt.imshow(im, cmap=plt.get_cmap('viridis'))
plt.title(f"region: {region_id}")
plt.savefig("./locate_glasser")
plt.close(fig)


sys.exit(0)



#
# Check coco info related to mscoco image_id
#

nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

# Captions
target = nsd_loader.read_image_coco_info([key-1])
print(target)
raise
print(target[0]['caption'])

# Image
img = nsd_loader.read_images(key-1)
fig = plt.figure()
plt.imshow(img)
plt.title(f"key: {key}")
plt.savefig(f"./temp_img_key{key}.png")
plt.close(fig)
