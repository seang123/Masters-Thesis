import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import tensorflow as tf
import json
import numpy as np
import nibabel as nb
import cortex
from itertools import groupby
import pandas as pd
from nsd_access import NSDAccess
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
import argparse

parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

data_loc = os.path.join('Log', args.dir, 'eval_out')
output_loc = f"./Log/{args.dir}/eval_out/output_captions.npy"
attn_score_loc = f"./Log/{args.dir}/eval_out/attention_scores.npy"
tokenizer_loc = f"./Log/{args.dir}/eval_out/tokenizer.json"

nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

# NSD condition keys
conditions = pd.read_csv(f"./TrainData/subj02_conditions.csv")
val_keys = conditions.loc[conditions['is_shared'] == 1]
val_keys = val_keys.reset_index(drop=True)

## Setup glasser region groups
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
## glasser region groups == end

def load_data():
    with open(output_loc, "rb") as f:
        outputs = np.load(f)
    with open(attn_score_loc, "rb") as f:
        attn_scores = np.load(f)

    print(outputs.shape)
    print(attn_scores.shape)
    return outputs, attn_scores

def visualise_attention(idx, attention_scores, outputs):
    
    attn = np.squeeze(attention_scores[idx], axis=-1) # (13, 180)

    ## Caption tokens -> text
    output = outputs[idx,:] # (1000, 13, 1) -> (13, 1)
    captions = np.squeeze(output, axis=-1) # (13, 1) -> (13,)
    captions = np.expand_dims(captions, 0)
    with open(tokenizer_loc, "r") as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    caption = tokenizer.sequences_to_texts(captions)
    caption_split = caption[0].split(" ")

    nrows = 7 # 3
    ncols = 2 # 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 40), sharex=True) # for 7x2 figsize = (20,40) (w*h)
    fig.subplots_adjust(wspace=0)#, hspace=0)
    r = 0
    c = 0
    x = 0
    images = []
    glasser_regions_over_time = []
    for j in range(13):
        data = attn[j, :]
        glasser_regions = np.zeros(glasser.shape)
        glasser_regions[:] = np.NaN
        for i, g in enumerate(groups):
            glasser_regions[g] = data[i]

        glasser_regions_over_time.append(glasser_regions)

        vert = cortex.Vertex(glasser_regions, 'fsaverage')
        im, extents = cortex.quickflat.make_flatmap_image(vert)
        if r >= nrows:
            r = 0
            c += 1
        images.append(axes[r,c].imshow(im, cmap=plt.get_cmap('viridis')))
        axes[r,c].set_title(f"word: {caption_split[x]}")
        axes[r,c].axis('off')
        r += 1
        x += 1
    

    axes[r,c].axis('off')
    #fig.colorbar(images[-1], ax=axes, orientation='horizontal', fraction=.1)
    plt.savefig("./attn_glasser.png", bbox_inches='tight')
    plt.close(fig)
   
    # NSD Key
    key = val_keys['nsd_key'].iloc[idx]
    # Target caption
    target = nsd_loader.read_image_coco_info([int(key)-1])
    target = target[-1]['caption']
    # Plot image and candidate caption
    picture = nsd_loader.read_images(int(key)-1)
    fig = plt.figure()
    plt.imshow(picture)
    title_str = "".join(caption) + "\n" + target
    plt.title(title_str)
    plt.axis('off')
    plt.savefig("candidate_cap.png", bbox_inches='tight')
    plt.close(fig)



    return

def avg_attention_at_t(t, attention_scores):

    attn = np.squeeze(attention_scores, axis=-1) # (1000, 13, 180)

    attn_t = attn[:,t,:]
    attn_t = np.mean(attn_t, axis=0)
    print("attn_t:", attn_t.shape)

    glasser_regions = np.zeros(glasser.shape)
    glasser_regions[:] = np.NaN
    for i, g in enumerate(groups):
        glasser_regions[g] = attn_t[i]

    df = pd.read_csv(open(f"./glasser_viz/glasser_description.csv", "r"), index_col=0)
    print(f"top 5 regions for word: {t}")
    attn_sort = np.argsort(attn_t)
    attn_sort = attn_sort[-10:][::-1]
    for i in range(len(attn_sort)):
        print("\t", attn_sort[i], "-", df['Area Description'].iloc[attn_sort[i]])

    vert = cortex.Vertex(glasser_regions, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)

    fig = plt.figure()
    plt.imshow(im)
    plt.title(f"Average across word: {t}")
    plt.savefig("./time_average.png")
    plt.close(fig)


    return

def top_region_at_t(t, idx, attention_scores):

    attn = np.squeeze(attention_scores[idx], axis=-1) # (13, 180)

    df = pd.read_csv(open(f"./glasser_viz/glasser_description.csv", "r"), index_col=0)

    print(f"top 5 regions for word: {t}")
    attn_sort = np.argsort(attn[t,:])
    attn_sort = attn_sort[-5:][::-1]
    for i in range(len(attn_sort)):
        print("\t", attn_sort[i], "-", df['Area Description'].iloc[attn_sort[i]])


def top_region_over_time(idx, attention_scores):
    """ what is the top most active regions attn score over time """

    attn = np.squeeze(attention_scores[idx], axis=-1) # (13, 180)

    df = pd.read_csv(open(f"./glasser_viz/glasser_description.csv", "r"), index_col=0)

    top_region_at_t(0, idx, attention_scores)
    top_region_at_t(12, idx, attention_scores)

    for i in range(13):
        print(max(attn[i,:]), np.argmax(attn[i,:]))

    print("sum over time (r:139):", np.sum(attn[:,139]))

    fig = plt.figure()
    for i in range(13):
        plt.plot(attn[i,:], label = i)
    plt.legend()
    plt.savefig('./attn_over_time.png')
    plt.close(fig)

    return


if __name__ == '__main__':
    outputs, attention_scores = load_data()
    idx = 672
    top_region_over_time(idx, attention_scores)
    visualise_attention(idx, attention_scores, outputs)
    avg_attention_at_t(0, attention_scores)








