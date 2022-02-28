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
from collections import defaultdict
import pandas as pd
from nsd_access import NSDAccess
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk import pos_tag, pos_tag_sents, word_tokenize
#import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('universal_tagset')
import argparse
import scipy.stats as scipy_stats 
import guse_comparison as guse_comp

parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

data_loc = os.path.join('Log', args.dir, 'eval_out')
output_loc = f"./Log/{args.dir}/eval_out/output_captions.npy"
attn_score_loc = f"./Log/{args.dir}/eval_out/attention_scores.npy"
tokenizer_loc = f"./Log/{args.dir}/eval_out/tokenizer.json"
out_path = f"./Log/{args.dir}/eval_out/"
print(f"out path: {out_path}")

nsd_loader = NSDAccess("/home/seagie/NSD3/")
nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
print("NSDAccess loader initialized ... ")

with open(tokenizer_loc, "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

# Glasser Region CSV
df = pd.read_csv(open(f"./glasser_viz/glasser_description.csv", "r"), index_col=0)
#print(df.head(10))

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

#groups = []
#glasser_indices = np.array(range(len(glasser)))
#for i in set(glasser):
#    group = glasser_indices[glasser == i]
#    groups.append(group)
#groups = groups[1:]
#ngroups = len(groups)
#assert ngroups == 180

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

## glasser region groups == end

def load_data():
    with open(output_loc, "rb") as f:
        outputs = np.load(f)
    with open(attn_score_loc, "rb") as f:
        attn_scores = np.load(f)

    outputs = np.squeeze(outputs, axis=-1)
    attn_scores = np.squeeze(attn_scores, axis=-1)
    print("outputs:    ", outputs.shape)
    print("attn_scores:", attn_scores.shape)
    return outputs, attn_scores

def visualise_attention(idx: int, attention_scores, outputs):
    """ For a particular trial, visualise the attention for each word
    also plot the image and the candidate caption.
    Parameters:
    -----------
        idx
            trial idx [0, 1000)
    """
    
    attn = attention_scores[idx]

    caption, caption_split = get_caption(idx, outputs)
    target = get_target_caption(idx)

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
        glasser_regions_lh = np.zeros(glasser_lh.shape)
        glasser_regions_rh = np.zeros(glasser_rh.shape)
        glasser_regions_lh[:] = np.NaN
        glasser_regions_rh[:] = np.NaN
        for i, g in enumerate(groups_lh):
            glasser_regions_lh[g] = data[i]
        for i, g in enumerate(groups_rh):
            glasser_regions_rh[g] = data[i+180]

        glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

        glasser_regions_over_time.append(glasser_regions)

        vert = cortex.Vertex(glasser_regions, subject='fsaverage', vmin=-8, vmax=8)
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
    plt.savefig(f"{out_path}/attn_glasser.png", bbox_inches='tight')
    plt.close(fig)
    return

def get_caption(idx: int, output:np.array):
    """ reverse-Tokenizes the output into natural language """
    captions = tokenizer.sequences_to_texts(output)
    caption = captions[idx]
    caption_split = caption.split(" ")
    return caption, caption_split

def get_target_caption(idx):
    """ Return the target caption for a given idx """
    key = val_keys['nsd_key'].iloc[idx]
    # Target caption
    target = nsd_loader.read_image_coco_info([int(key)-1])
    target = target[-1]['caption']
    return target

def get_picture(idx):
    """ Get the NSD picture for a given idx """
    key = val_keys['nsd_key'].iloc[idx]
    return nsd_loader.read_images(int(key)-1)

def print_examples(samples: int, output: np.array):

    indices = np.random.randint(0, 1000, samples)

    for i in indices:
        cand = get_caption(i, output)[0]
        targ = get_target_caption(i)
        print(f"\n--- idx: {i} - NSD: {val_keys['nsd_key'].iloc[i]} --- ")
        print("\tCandidate:", cand)
        print("\tTarget:   ", targ)
 

def plot_image_caption(idx: int, output: np.array):
    """ Plot the image caption pair 
    Returns:
    --------
        caption : str
        caption_split : list(str)
        target : str
    """
    caption, _ = get_caption(idx, output)

    # NSD Key
    target = get_target_caption(idx)
    # Plot image and candidate caption
    picture = get_picture(idx)

    fig = plt.figure()
    plt.imshow(picture)
    title_str = remove_pad(caption, end='leave') + "\n" + target
    plt.title(title_str)
    plt.axis('off')
    plt.savefig(f"{out_path}/candidate_cap.png", bbox_inches='tight')
    plt.close(fig)


def avg_attention_at_t(t, attention_scores):
    """ Avg attention for a word at position t across trials """
    #attn = np.squeeze(attention_scores, axis=-1) # (1000, 13, 180)
    return np.mean(attention_scores[:,t,:], axis=0)

def avg_attention_across_trials(attention_scores):
    """ Plot the average attention maps for each word position across trials """

    words = attention_scores.shape[1]

    avgs = []
    for i in range(words):
        avgs.append(avg_attention_at_t(i, attention_scores))
    avgs = np.array(avgs) # (13, 180)

    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20, 40), sharex=True) # for 7x2 figsize = (20,40) (w*h)
    fig.subplots_adjust(wspace=0, top=0.85)#, hspace=0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    r = 0
    c = 0
    for w in range(words):
        glasser_regions_lh = np.zeros(glasser_lh.shape)
        glasser_regions_rh = np.zeros(glasser_rh.shape)
        glasser_regions_lh[:] = np.NaN
        glasser_regions_rh[:] = np.NaN
        for i, g in enumerate(groups_lh): # [0, 179]
            glasser_regions_lh[g] = avgs[w,i]
        for i, g in enumerate(groups_rh): # [180, 359]
            glasser_regions_rh[g] = avgs[w,i+180]

        glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

        attn_w = np.argsort(avgs[w,:])
        attn_w = attn_w[-3:][::-1]
        title_str = ""
        for i, region_idx in enumerate(attn_w):
            if region_idx >= 180:
                region_idx -= 180 
            title_str += f"\n{df['Area Description'].iloc[region_idx]}"
            break

        vert = cortex.Vertex(glasser_regions, 'fsaverage')
        im, extents = cortex.quickflat.make_flatmap_image(vert)
        if r >= axes.shape[0]: r = 0; c += 1 # col major
        axes[r,c].imshow(im, cmap=plt.get_cmap('viridis'))
        axes[r,c].set_title(f"t: {w}{title_str}")
        axes[r,c].axis('off')
        r += 1

    #volume = cortex.Volume.random(subject='S1', xfmname='fullhead')
    #print("-- volume created --")
    ##qmap = cortex.quickflat.make_figure(volume,with_curvature=True,with_sulci=True)
    #cortex.quickflat.make_png(f"{out_path}/cortex_png.png", volume, with_rois=True)

    fig.suptitle("Average attention at time t across trials")
    axes[r,c].axis('off')
    plt.savefig(f"{out_path}/avg_attn_at_t.png", bbox_inches='tight')
    plt.close(fig)
    return

def top_region_at_t(t, idx, attention_scores):
    """ Print the name of the _n_ most active regions at time t """

    attn = np.squeeze(attention_scores[idx], axis=-1) # (13, 180)
    n = 10

    attn_sort = np.argsort(attn[t,:])
    attn_sort = attn_sort[-n:][::-1]
    print(f"top {n} regions for word: {t}")
    for i, region in enumerate(attn_sort):
        if region >= 180: region -= 180
        print("\t", region, "-", df['Area Description'].iloc[region])

def top_region_over_time(idx, attention_scores):
    """ what is the top most active regions attn score over time """

    top_region_at_t(0, idx, attention_scores)
    top_region_at_t(12, idx, attention_scores)

    for i in range(13):
        max_val = max(attn[i,:])
        max_idx = np.argmax(attn[i,:])
        if max_idx > 180: max_idx -= 180
        print(max_val, max_idx)

    print("sum over time - Primary Visual Cortex LH (r:1):", np.sum(attn[:,0]))
    print("sum over time - Primary Visual Cortex RH (r:1):", np.sum(attn[:,180]))

    fig = plt.figure()
    for i in range(13):
        plt.plot(attn[i,:], label = i)
    plt.legend()
    plt.savefig(f'{out_path}/attn_over_time.png')
    plt.close(fig)

    return

def regions_count(attention_scores):
    """ Generate a dataframe describing region counts """

    trials, length, regions = attention_scores.shape
    attn = np.mean(attention_scores, axis=0)

    d = {}
    regions = range(1, 361)
    print("regions:", len(regions))

    d = np.zeros((15, 360), dtype=np.int32)
    
    for l in range(length):
        x = np.argsort(attn[l, :])
        d[l, :] = x

    df = pd.DataFrame(d, index=range(15), columns=range(1, 361))
    print(df.describe())
    

def top_regions_over_time_trial(attention_scores):
    """ Average across trials and across timesteps - then print the top 10 regions """
    #attn = attention_scores[:,:11, :]
    attn_mean = np.mean(np.mean(attention_scores, axis=0), axis=0) # (360,)

    attn_sort = np.argsort(attn_mean)
    attn_sort = attn_sort[-10:][::-1]
    print(f"Top 10 regions averaged across trials and time")
    for i, region in enumerate(attn_sort):
        temp_region = region
        hemisphere = 'lh'
        if region >= 180: region -= 180; hemisphere = 'rh'
        print(f"\t{region:3} - {attn_mean[temp_region]:.4f} - {hemisphere} - {df['Area Description'].iloc[region]}")

def temp(attention_scores):

    # Average and sort by attention
    attn_mean = np.mean(np.mean(attention_scores, axis=0), axis=0) # (360,)
    #attn_sort = np.argsort(attn_mean)
    #attn_sort = attn_sort[-7:][::-1]
    attn_sort = []
    print(attn_sort)

    glasser_regions_lh = np.zeros(glasser_lh.shape)
    glasser_regions_rh = np.zeros(glasser_rh.shape)
    glasser_regions_lh[:] = np.NaN
    glasser_regions_rh[:] = np.NaN
    for i, g in enumerate(groups_lh):
        if i in set(attn_sort):
            glasser_regions_lh[g] = np.NaN
        else:
            glasser_regions_lh[g] = attn_mean[i]
    for i, g in enumerate(groups_rh):
        if i+180 in set(attn_sort):
            glasser_regions_rh[g] = np.NaN
        else:
            glasser_regions_rh[g] = attn_mean[i+180]

    glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)

    vert = cortex.Vertex(glasser_regions, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)

    fig = plt.figure()
    plt.imshow(im)
    plt.savefig(f"{out_path}/norm_activity.png")
    plt.close(fig)

def correlation(attention_scores):
    """ Calculate correlation between left- and right-hemispheres """
    attn = np.mean(attention_scores, axis=0) # (13, 360)
    from scipy.stats.stats import pearsonr
    corr = pearsonr(attn[0,0:180], attn[0,180:])
    print(corr)
    corr = np.corrcoef(attn[0,:180], attn[0,180:])
    print(corr)

def visualise_regions(region_idx: int = 0):
    """ Visualise a region by idx [0-179] """
    glasser_regions = np.zeros(glasser.shape)
    glasser_regions[:] = np.NaN
    for i, g in enumerate(groups_concat): # [0, 179]
        if i == region_idx:
            glasser_regions[g] = 1
        else:
            glasser_regions[g] = 0

    vert = cortex.Vertex(glasser_regions, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)

    fig = plt.figure()
    plt.imshow(im)
    plt.savefig(f"{out_path}/region_{region_idx}.png")
    plt.close(fig)
    return

def remove_pad(caption: str, end='remove', replace_unk=True):
    """ Removes the padding tokens
    Parameters:
    -----------
        captions : str
        end : str
            whether to replace, remove, or leave the end token
        replace_unk : bool
            whether to replace the '<unk>' token
    """
    x = caption.split(" ")
    try:
        while True:
            x.remove('<pad>')
    except ValueError:
        # No pad token present so we can continue
        pass
    if end == 'replace':
        x = ['end' if i == '<end>' else i for i in x]
    elif end == 'remove':
        try:
            x.remove('<end>')
        except ValueError:
            pass
    elif end == 'leave':
        pass
    if replace_unk:
        x = ['unk' if i == '<unk>' else i for i in x] 
    return " ".join(x)

def ner(outputs):
    """ Named-entity recognition """
    import spacy
    from spacy import displacy
    from collections import Counter
    nlp = spacy.load("en_core_web_sm")

    captions = tokenizer.sequences_to_texts(outputs) # list of len 1000
    captions_clean = [remove_pad(i, end = 'remove') for i in captions] # remove <pad> <end> tokens

    for i, cap in enumerate(captions_clean):
        print("cap:", cap)
        doc = nlp((cap))
        # Analyze syntax
        print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
        print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

        # Find named entities, phrases and concepts
        for entity in doc.ents:
                print(entity.text, entity.label_)
        break
    

def tagging(outputs, tagset = 'universal'):
    """ Tag captions using the NLTK part-of-speech tagger
    https://www.nltk.org/book/ch05.html
    Parameters:
    -----------
        outputs : ndarray
            candidate captions
        tagset : str
            Tag set to use, 'universal', 'wsj', or 'brown'
    """
    
    print(pos_tag(['end']))
    print(pos_tag(['pad']))

    captions = tokenizer.sequences_to_texts(outputs) # list of len 1000
    captions_clean = [remove_pad(i, end = 'replace') for i in captions] # remove <pad> <end> tokens
    tagged_captions = pos_tag_sents([word_tokenize(i) for i in captions_clean], tagset=tagset) # list len 1000
    return tagged_captions

def attention_by_tag(outputs, attention_scores):
    """ Plot attention maps based on word tagging """

    tagged_captions = tagging(outputs)
    tags = ['NOUN', 'VERB', 'ADJ']
    tags_dict = defaultdict(list)

    for t in tags:
        for i, sent in enumerate(tagged_captions):
            for j, word in enumerate(sent):
                if word[1] == t:
                    tags_dict[t].append((i,j))

    attention_tag = defaultdict(list)
    for t in tags:
        for i, j in tags_dict[t]:
            attention_tag[t].append(attention_scores[i, j, :])
    
        print(f"{t} - {len(tags_dict[t])}")

    overall_mean = np.mean(np.mean(attention_scores, axis=0), axis=0)

    pop_mean = np.zeros((360,), dtype=np.float32)
    pop_count = 0
    # Get the pop mean excluding '<pad>'
    for i in range(len(tagged_captions)):
        for j in range(len(tagged_captions[i])):
            pop_mean += attention_scores[i,j]
            pop_count += 1
    pop_mean /= pop_count

    # Visualise
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 30), sharex=True) # for 7x2 figsize = (20,40) (w*h)
    fig.subplots_adjust(wspace=0)#, hspace=0)
    r = 0
    for tag in tags:
        attention_tag_i = np.array(attention_tag[tag])
        attention_tag_i = np.mean(attention_tag_i, axis=0) 
        print(f"{tag} - {np.std(attention_tag_i, axis=0):.5f}")
        attention_tag_i -= overall_mean

        glasser_regions_lh = np.zeros(glasser_lh.shape)
        glasser_regions_rh = np.zeros(glasser_rh.shape)
        glasser_regions_lh[:] = np.NaN
        glasser_regions_rh[:] = np.NaN
        for i, g in enumerate(groups_lh): # [0, 179]
            glasser_regions_lh[g] = attention_tag_i[i]
        for i, g in enumerate(groups_rh): # [180, 359]
            glasser_regions_rh[g] = attention_tag_i[i+180]

        glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)
        vert = cortex.Vertex(glasser_regions, 'fsaverage')
        im, extents = cortex.quickflat.make_flatmap_image(vert)

        pcm = axes[r].imshow(im, cmap=plt.get_cmap('viridis')) # 'RdBu', 'BrBG', 'viridis'
        plt.title("Average attention across occurances of NOUN")
        axes[r].set_title(f"Average attention for: {tag}")
        axes[r].axis('off')
        fig.colorbar(pcm, ax=axes[r], shrink=0.6)#, location='bottom')
        r += 1

    plt.savefig(f"{out_path}/attention_by_tag.png")
    plt.close(fig)

def attention_against_word(idx, outputs, attention_scores):
    """ Returns a square plot showing attention for each word - not very informative """
    captions = tokenizer.sequences_to_texts(outputs)
    caption = captions[idx]
    caption_split = caption.split(" ")

    print(attention_scores.shape)
    attn_concat = attention_scores[idx,:,:180] + attention_scores[idx,:,180:]

    fig, ax = plt.subplots(1,1)
    ax.pcolormesh(attn_concat)
    ax.set_title("Attention map", loc="left")
    ax.set_xlabel("Region")
    ax.set_ylabel("Word")
    ax.set_yticks(np.arange(len(caption_split)))
    ax.set_yticklabels(caption_split)

    plt.savefig(f"{out_path}/attn_map.png")
    plt.close(fig)

def guse_comparison(idx, outputs):
    """ Call the guse comparison code """
    captions = tokenizer.sequences_to_texts(outputs)
    caption = captions[idx]
    caption_split = caption.split(" ")
    caption = [remove_pad(caption, end='remove')]

    print("Inference caption:\n", caption)
    # Call the guse_comparison module that returns the similarity measure
    # between the inference and all training captions
    similarity = guse_comp.guse_comparison(caption)
    return

def rank_transform(data):
    #sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return np.log(data)

def attn_statistics(attn_scores):
    print("max:", np.max(attention_scores[idx]))
    print("min:", np.min(attention_scores[idx]))
    print("range:", np.ptp(attention_scores[idx]))
    print("mean:", np.mean(attention_scores[idx]))
    print("std:", np.std(attention_scores[idx]))
    return


def linear_attn_maps(idx, attn_scores):
    attn = np.mean(attn_scores[idx,:,:], axis=0)
    data_idx = np.argsort(attn)[::-1]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 30), sharex=True) # for 7x2 figsize = (20,40) (w*h)

    colours = np.arange(-180, 180)

    glasser_regions_lh = np.zeros(glasser_lh.shape)
    glasser_regions_rh = np.zeros(glasser_rh.shape)
    glasser_regions_lh[:] = np.NaN
    glasser_regions_rh[:] = np.NaN
    for i, idx in enumerate(data_idx):
        if idx < 180:
            glasser_regions_lh[groups[idx]] = colours[i]
        else:
            glasser_regions_rh[groups[idx]] = colours[i]
    glasser_regions = np.vstack((glasser_regions_lh, glasser_regions_rh)).flatten() # (327684,)
    vert = cortex.Vertex(glasser_regions, 'fsaverage')
    im, extents = cortex.quickflat.make_flatmap_image(vert)
    axes.imshow(im, cmap=plt.get_cmap('viridis_r')) # 'RdBu', 'BrBG', 'viridis'

    plt.savefig(f"{out_path}/linear_attn_maps.png")
    plt.close(fig)

if __name__ == '__main__':
    outputs, attention_scores = load_data()
    idx =  np.random.randint(0, 1000) # 672 - cattle, 2 - surfer, 946 - snowboarder
    print(f"--- trial: {idx} --- NSD: {val_keys['nsd_key'].iloc[idx]} ---")

    # Rank transform
    attention_scores = rank_transform(attention_scores)

    #print_examples(5, outputs)

    #plot_image_caption(idx, outputs)
    #attention_by_tag(outputs, attention_scores)

    #guse_comparison(idx, outputs)

    #attention_against_word(idx, outputs, attention_scores)
    
    #top_regions_over_time_trial(attention_scores)
    #temp(attention_scores)
    #correlation(attention_scores)
    
    #top_region_over_time(idx, attention_scores)
    #visualise_attention(idx, attention_scores, outputs)
#    avg_attention_across_trials(attention_scores)

    #regions_count(attention_scores)

    ner(outputs)





