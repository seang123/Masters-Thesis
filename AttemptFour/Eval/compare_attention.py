import nibabel as nb
import cortex
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json

fileDir = os.path.dirname(os.path.realpath('__file__'))
print("relative path:", fileDir)

data_sub1 = './Log/no_attn_loss_subject_1/eval_out/output_captions_raw_41.npy'
data_sub2 = './Log/no_attn_loss_const_lr2/eval_out/output_captions_raw_41.npy'

attention_sub1_file = f'{fileDir}/Log/no_attn_loss_subject_1/eval_out/attention_scores_41.npy'
attention_sub2_file= f'{fileDir}/Log/no_attn_loss_const_lr2/eval_out/attention_scores_41.npy'

conditions = pd.read_csv(f"./TrainData/subj02_conditions.csv")
val_keys = conditions.loc[conditions['is_shared'] == 1].reset_index(drop=True)

## Load tokenizer
tokenizer_loc = f"./Log/proper_split_sub2/eval_out/tokenizer.json"
with open(tokenizer_loc, "r") as f:
    tokenizer = tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

## Load Glasser
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
#groups_concat = list(map(list.__add__, groups_lh, groups_rh))
groups_lh = groups_lh[1:]
groups_rh = groups_rh[1:]
assert len(groups) == 360, "Using separate hemishere groups = 360"

glasser_indices = np.array(range(len(glasser)))
groups_concat = []
for i in set(glasser):
    groups_concat.append(glasser_indices[glasser == i])
groups_concat = groups_concat[1:]
print("len(groups_concat):",  len(groups_concat))


def load_data(fname):
    with open(fname, "rb") as f:
        data = np.load(f)
    return np.squeeze(data, axis=-1)

def rank_transform(data):
    return np.log(data)

def get_flatmap(glasser_regions):
        vert = cortex.Vertex(glasser_regions, subject='fsaverage')#, vmin=-8, vmax=8)
        im, extents = cortex.quickflat.make_flatmap_image(vert)
        return im, extents

def compare_attn_time(sub1, sub2):
    sub1 = np.mean(sub1, axis=(0,1)) # (15, 360)
    sub2 = np.mean(sub2, axis=(0,1)) # (15, 360)
    print(sub1.shape, sub2.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    fig.tight_layout()#rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Attention maps averaged across trials and time")

    # Sub 1
    data = sub1
    glasser_regions_lh = np.zeros(glasser_lh.shape)
    glasser_regions_rh = np.zeros(glasser_rh.shape)
    glasser_regions_lh[:] = np.NaN
    glasser_regions_rh[:] = np.NaN
    for i, g in enumerate(groups_lh):
        glasser_regions_lh[g] = data[i]
    for i, g in enumerate(groups_rh):
        glasser_regions_rh[g] = data[i+180]
    glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

    im, _ = get_flatmap(glasser_regions)
    axes[0].imshow(im, cmap=plt.get_cmap('viridis'))
    axes[0].set_title(f"Subject 1")
    axes[0].axis('off')

    # Sub 2
    data = sub2
    glasser_regions_lh = np.zeros(glasser_lh.shape)
    glasser_regions_rh = np.zeros(glasser_rh.shape)
    glasser_regions_lh[:] = np.NaN
    glasser_regions_rh[:] = np.NaN
    for i, g in enumerate(groups_lh):
        glasser_regions_lh[g] = data[i]
    for i, g in enumerate(groups_rh):
        glasser_regions_rh[g] = data[i+180]
    glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

    im, _ = get_flatmap(glasser_regions)
    axes[1].imshow(im, cmap=plt.get_cmap('viridis'))
    axes[1].set_title(f"Subject 2")
    axes[1].axis('off')

    plt.savefig("./Eval/compare_attn_trial_time.png", bbox_inches='tight')
    plt.close(fig)


def compare_attn(sub1, sub2):

    sub1 = np.mean(sub1, axis=0) # (15, 360)
    sub2 = np.mean(sub2, axis=0) # (15, 360)

    fig, axes = plt.subplots(15, 2, figsize=(10, 40))
    #fig.subplots_adjust(wspace=0, top=0.5)#, hspace=0) top=0.85
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])#rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Attention maps averaged across trials")

    # Sub1 
    images_sub1 = []
    r = 0
    for j in range(15):
        data = sub1[j, :]
        glasser_regions_lh = np.zeros(glasser_lh.shape)
        glasser_regions_rh = np.zeros(glasser_rh.shape)
        glasser_regions_lh[:] = np.NaN
        glasser_regions_rh[:] = np.NaN
        for i, g in enumerate(groups_lh):
            glasser_regions_lh[g] = data[i]
        for i, g in enumerate(groups_rh):
            glasser_regions_rh[g] = data[i+180]
        glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

        im, _ = get_flatmap(glasser_regions)
        images_sub1.append(axes[r,0].imshow(im, cmap=plt.get_cmap('viridis')))
        #axes[r,0].set_title(f"Subject 1 - word {j}")
        axes[r,0].axis('off')
        r += 1
    axes[0, 0].set_title("Subject 1")
    # Sub2
    images_sub2 = []
    r = 0
    for j in range(15):
        data = sub2[j, :]
        glasser_regions_lh = np.zeros(glasser_lh.shape)
        glasser_regions_rh = np.zeros(glasser_rh.shape)
        glasser_regions_lh[:] = np.NaN
        glasser_regions_rh[:] = np.NaN
        for i, g in enumerate(groups_lh):
            glasser_regions_lh[g] = data[i]
        for i, g in enumerate(groups_rh):
            glasser_regions_rh[g] = data[i+180]
        glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

        im, _ = get_flatmap(glasser_regions)
        images_sub2.append(axes[r,1].imshow(im, cmap=plt.get_cmap('viridis')))
        #axes[r,1].set_title(f"Subject 2 - word {j}")
        axes[r,1].axis('off')
        r += 1
    axes[0, 1].set_title("Subject 2")

    plt.savefig("./Eval/attention_map_comparison.png")
    plt.close(fig)

    return


def main():
    sub1 = load_data("./Log/proper_split_sub1/eval_out/attention_scores_97.npy")
    sub2 = load_data("./Log/proper_split_sub2/eval_out/attention_scores_98.npy")
    print(sub1.shape)
    print(sub2.shape)

    sub1 = np.log(sub1)
    sub2 = np.log(sub2)

    timestep = 14

    sub1_mean = np.mean(sub1[:, timestep, :], axis=0) # -> (360,)
    sub2_mean = np.mean(sub2[:, timestep, :], axis=0) # -> (360,)
    print("sub1_mean:", sub1_mean.shape)

    ## Sub 1
    data = sub1_mean
    mask_lh = np.zeros(glasser_lh.shape)
    mask_rh = np.zeros(glasser_rh.shape)
    mask_lh[:] = np.NaN
    mask_rh[:] = np.NaN
    for i, g in enumerate(groups_lh):
        mask_lh[g] = data[i]
    for i, g in enumerate(groups_rh):
        mask_rh[g] = data[i+180]
    glasser_regions = np.vstack((mask_lh, mask_rh)).flatten() # (327684,)
    im1, _ = get_flatmap(glasser_regions)
    ## Sub 2
    data = sub2_mean
    mask_lh = np.zeros(glasser_lh.shape)
    mask_rh = np.zeros(glasser_rh.shape)
    mask_lh[:] = np.NaN
    mask_rh[:] = np.NaN
    for i, g in enumerate(groups_lh):
        mask_lh[g] = data[i]
    for i, g in enumerate(groups_rh):
        mask_rh[g] = data[i+180]
    glasser_regions = np.vstack((mask_lh, mask_rh)).flatten() # (327684,)
    im2, _ = get_flatmap(glasser_regions)


    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    #fig.subplots_adjust(wspace=0, top=0.5)#, hspace=0) top=0.85
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])#rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Attention maps averaged across trials at word position {timestep}")
    axes[0].imshow(im1, cmap=plt.get_cmap('viridis'))
    axes[0].set_title("Subject 1")
    axes[0].axis('off')
    axes[1].imshow(im2, cmap=plt.get_cmap('viridis'))
    axes[1].set_title("Subject 2")
    axes[1].axis('off')

    plt.savefig(f"./attn_maps_sub1_vs_sub2_t{timestep}.png", bbox_inches='tight')
    plt.close(fig)

def find_idx_word(word, captions):
    idx = []
    for i in range(captions.shape[0]):
        for k in range(captions.shape[1]):
            if captions[i,k] == word:
                idx.append( (i,k) )
    return idx

def compare_regions():
    data = load_data("./Log/proper_split_sub2/eval_out/attention_scores_98.npy")
    print(data.shape)
    caps = load_data("./Log/proper_split_sub2/eval_out/output_captions_98.npy")
    captions = tokenizer.sequences_to_texts(caps)
    captions = np.array([i.split(" ") for i in captions])
    #captions = np.reshape(captions, captions.shape[0] * captions.shape[1]) # flatten captions
    print(captions.shape)

    ida = find_idx_word('man', captions)
    idb = find_idx_word('woman', captions)
    idc = find_idx_word('people', captions)

    data_ = np.log(data)
    data_ = np.mean(data_, axis=0)
    data_ = np.mean(data_, axis=0)
    idx = np.argsort(data_)

    data = np.mean(data, axis=0)
    data = np.mean(data, axis=0)
    idx2 = np.argsort(data)[::-1]

    fig, ax = plt.subplots(1,1,figsize=(16,9))
    ax.plot(data[idx2], color='darkgray')
    #ax.scatter(315, data[idx[315]], marker='*', c = 'darkslategrey')
    plt.title("Attention across 360 regions. Averaged across trials and timesteps")
    plt.ylabel("Softmax attention")
    plt.xlabel("Region")
    plt.grid()
    plt.savefig("temp_delete.png", bbox_inches='tight')
    plt.close(fig)

    


if __name__ == '__main__':
    
    #main()
    compare_regions()

    #attn_sub1, attn_sub2 = rank_transform(attn_sub1), rank_transform(attn_sub2)
    #compare_attn(attn_sub1, attn_sub2)
    #compare_attn_time(attn_sub1, attn_sub2)





