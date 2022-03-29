import nibabel as nb
import cortex
import json
import os, sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from nsd_access import NSDAccess
import tensorflow as tf
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu

print("seed:", np.random.get_state()[1][0])
np.random.seed(67) # 6

fileDir = os.path.dirname(os.path.realpath('__file__'))
print("relative path:", fileDir)


data_sub1_file = './Log/no_attn_loss_subject_1/eval_out/output_captions_raw_41.npy'
data_sub2_file = './Log/no_attn_loss_const_lr2/eval_out/output_captions_raw_41.npy'

tokenizer_loc = f"./Log/no_attn_loss_const_lr2/eval_out/tokenizer.json"

# Val keys are the same for each subject
conditions = pd.read_csv(f"./TrainData/subj02_conditions.csv")
val_keys = conditions.loc[conditions['is_shared'] == 1].reset_index(drop=True)
val_keys_list = val_keys['nsd_key'].values
val_keys = pd.read_csv(f"./TrainData/test_conditions.csv") # overwrite val keys with test keys
val_keys_list = val_keys['nsd_key'].values


# Tokenizer is initalised on the whole 73k NSD so it should be the same for both subjects
with open(tokenizer_loc, "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
print("Tokenizer loaded ...")

def init_nsd():
    nsd_loader = NSDAccess("/home/seagie/NSD3/")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)
    print("NSDAccess loader initialized ... ")
    return nsd_loader

# Init NSD loader
nsd_loader = init_nsd()

def large_lstm():
    print("-- large_lstm --")
    with open(f"./Log/large_lstm/eval_out/output_captions_97.npy", "rb") as f:
        data = np.load(f)
        print(data.shape)

    data = np.squeeze(data, axis=-1)

    targets = load_captions_dict()
    
    all_bleu_scores(data, targets)

    np.random.seed(8)
    n_rows, n_cols = 3, 4
    indices = np.random.randint(0, 1000, (n_rows, n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,9))
    for r in range(n_rows):
        for c in range(n_cols):
            idx = indices[r,c]
            pic = get_picture(idx)
            caption, _ = get_caption(idx, data)
            caption = remove_pad(caption)
            axes[r,c].imshow(pic)
            axes[r,c].set_title(f"{caption}")
            axes[r,c].axis('off')
    plt.savefig("./large_lstm_out.png", bbox_inches='tight')
    plt.close(fig)


def load_data(file_name: str):
    with open(file_name, "rb") as f:
        data = np.load(f)
    print(data.shape)
    return data

def remove_pad(x):
    x = x.split(" ")
    x = [i for i in x if i != '<pad>' and i != '<end>']
    return " ".join(x)

def get_caption(idx: int, output:np.array):
    """ reverse-Tokenizes the output into natural language """
    captions = tokenizer.sequences_to_texts(output)
    caption = captions[idx]
    caption_split = caption.split(" ")
    return caption, caption_split

def untokenize(output:np.array):
    return tokenizer.sequences_to_texts(output)

def get_target_caption(idx: int):
    """ Return the target caption for a given idx """
    key = val_keys['nsd_key'].iloc[idx]
    # Target caption
    target = nsd_loader.read_image_coco_info([int(key)-1])
    target = target[-1]['caption']
    return target

def get_picture(idx: int):
    """ Get the NSD picture for a given idx """
    key = val_keys['nsd_key'].iloc[idx]
    return nsd_loader.read_images(int(key)-1)

def load_captions_dict():
    path = '/fast/seagie/data/captions/'
    captions = defaultdict(list)
    for i, key in enumerate(val_keys_list):
        with open(f"{path}/KID{key}.txt", "r") as f:
            content = f.read()
            for i in content.splitlines():
                cap = i.replace(".", " ").replace(",", " ").strip().split(" ")
                cap = [i.lower() for i in cap if i != '']
                cap = " ".join(cap)
                captions[key].append(cap)

    return captions
                


def bleu_score_sort(data_max: np.array, targets: defaultdict):
    """ Sort candidate captions by bleu score 
    
    Parameter
    ---------
        data_max : np.array  (1000, 15)
            argmax'd model output
        targets : dict of lists
            holds (nsd_key, [target caps]) 

    Return
    ------
        bleu_sort : list(tuple)
            list of ((idx, key), bleu_score)
    """

    chencherry = SmoothingFunction()

    scores = {}

    cand_captions = untokenize(data_max)
    
    for i, key in enumerate(val_keys_list):
        references = [i.split(" ") for i in targets[key]]
        hypothesis = cand_captions[i] 
        hypothesis = remove_pad(hypothesis).split(" ")

        b_score = sentence_bleu(references, hypothesis, weights=(1/4, 1/4, 1/4, 1/4), smoothing_function=chencherry.method2)
        scores[(i,key)] = b_score
    
    # Sort by val (bleu score)
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

    print("mean bleu:", np.mean([v for k,v in scores.items()]))

    # Conver the sorted dict to a sorted list of tuples [(key, val)]
    return [(k, v) for k,v in scores.items()]

def all_bleu_scores(data_max: np.array, targets: defaultdict):
    """ Compute Bleu 1-4 """

    chencherry = SmoothingFunction()
    weights = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1./1., 0., 0., 0.),
        (1./2., 1./2., 0., 0.),
        (1./3., 1./3., 1./3., 0.),
        (1./4., 1./4., 1./4., 1./4.)     
    ]

    cand_captions = tokenizer.sequences_to_texts(data_max)

    s = time.time()
    references = [] 
    hypothesis = []
    for i, key in enumerate(val_keys_list):
        ref = [i.split(" ") for i in targets[key]]
        hyp = cand_captions[i] 
        hyp = remove_pad(hyp).split(" ")
        references.append(ref)
        hypothesis.append(hyp)
    print(f"prepare caps: {(time.time() - s):.3f}")

    s = time.time()
    for w in weights[4:]:
        b_score = corpus_bleu(references, hypothesis, weights=w, smoothing_function=chencherry.method0)
        print(b_score)
    print(f"time to compute bleu: {(time.time() - s):.3f}")

    return


def sample_captions_choice(data_max: np.array, sorted_bleu_scores: list, fname='sorted_cap_img_pairs'):
    """
    Parameter
    ---------
        sorted_bleu_scores : [((idx, nsd_key), score)]
    """
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,9))

    keys = [i[0] for i in sorted_bleu_scores]
    idx_bad = keys[:n_cols]
    idx_med = keys[498:502]
    idx_good= keys[-n_cols:]


    # Good
    for i in range(n_cols):
        pic = get_picture(idx_good[i][0])
        caption, _ = get_caption(idx_good[i][0], data_max)
        caption = remove_pad(caption)
        axes[0,i].imshow(pic)
        axes[0,i].imshow(pic)
        axes[0,i].set_title(f"{caption}", fontsize=10)
        axes[0,i].axis('off')

    # Med
    for i in range(n_cols):
        pic = get_picture(idx_med[i][0])
        caption, _ = get_caption(idx_med[i][0], data_max)
        caption = remove_pad(caption)
        axes[1,i].imshow(pic)
        axes[1,i].imshow(pic)
        axes[1,i].set_title(f"{caption}", fontsize=10)
        axes[1,i].axis('off')

    # Bad
    for i in range(n_cols):
        pic = get_picture(idx_bad[i][0])
        caption, _ = get_caption(idx_bad[i][0], data_max)
        caption = remove_pad(caption)
        axes[2,i].imshow(pic)
        axes[2,i].imshow(pic)
        axes[2,i].set_title(f"{caption}", fontsize=10)
        axes[2,i].axis('off')

    plt.savefig(f"./Eval/caption_samples/{fname}.png", bbox_inches='tight')
    plt.close(fig)
    return


def sample_captions(data_max: np.array, fname= 'cap_img_pairs'):

    n_rows, n_cols = 3, 4

    indices = np.random.randint(0, 515, (n_rows, n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,9))
    
    for r in range(n_rows):
        for c in range(n_cols):
            idx = indices[r,c]
            pic = get_picture(idx)
            caption, _ = get_caption(idx, data_max)
            caption = remove_pad(caption)
            axes[r,c].imshow(pic)
            axes[r,c].set_title(f"{caption}", fontsize = 10)
            axes[r,c].axis('off')

    plt.savefig(f"./Eval/caption_samples/{fname}.png", bbox_inches='tight')
    plt.close(fig)
    
def plot_bleu(bleu):

    fig, ax = plt.subplots(1,1)
    ax.hist([i[1] for i in bleu])
    plt.title("Bleu scores for subject 2")
    plt.savefig("./Eval/caption_samples/bleu.png")
    plt.close(fig)

def perplexity(data):
    """ argmaxed data """
    print("-- perplexity --")
    print(data.shape)

    caps = untokenize(data) # list of captions
    caps = [i.split(" ") for i in caps]
    caps= [item for sublist in caps for item in sublist]
    print("nr. caps:", len(caps))


    with open("./TrainData/tokenizer_73k.json", 'r') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))
    words_tuple = list(tokenizer.word_counts.items())
    words_tuple = sorted(words_tuple, key=lambda x: x[1], reverse=True)
    vocab = words_tuple[:5000]

    zeros = 0
    probs = []
    for i, (word, _) in enumerate(vocab):
        count = caps.count(word)
        if count != 0:
            probs.append( count / 5000 )
        else:
            probs.append( 1.0e-10 )
            zeros += 1
    print("zeros:", zeros)

    log_probs = np.log(probs)

    entropy = 0
    for i in range(len(probs)):
        entropy += probs[i] * log_probs[i]
    print("entropy:", entropy)
    

    

def load_captions(keys):
    captions_path = "/fast/seagie/data/captions/"
    caps = []
    for count, key in enumerate(keys):
        with open(f"{captions_path}/KID{key}.txt", "r") as f:
            content = f.read()
            cid = 0
            for line in content.splitlines():
                cap = line.replace(".", " ").replace(",", " ").strip().split(" ")
                cap = [i.lower() for i in cap]
                #cap = remove_stop_words(cap)
                cap = ['<start>'] + cap + ['<end>']
                cap = " ".join(cap)
                caps.append(cap)

    return caps


def train_set_perplexity(data):
    """ Perplexity of subject 2 training set """
    with open("./TrainData/tokenizer_73k.json", 'r') as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.load(f))

    words_tuple = list(tokenizer.word_counts.items())
    words_tuple = sorted(words_tuple, key=lambda x: x[1], reverse=True)
    vocab = words_tuple[:5000]

    # Load keys
    df = pd.read_csv(f'./TrainData/subj02_conditions.csv')
    unq = df['nsd_key'].loc[df['is_shared']==0]
    shrd = df['nsd_key'].loc[df['is_shared']==1]
    train_keys, val_keys = unq.values, shrd.values
    all_keys = np.concatenate((train_keys, val_keys))
    
    # Load a [list] of all captions 
    captions = load_captions(all_keys)
    captions = [i.split(" ") for i in captions]
    captions = [item for sublist in captions for item in sublist]

    probs = []
    zeros = 0
    for i, (word, _) in enumerate(vocab):
        count = captions.count(word)
        if count != 0:
            probs.append( count / 5000 )
        else:
            zeros += 1

    print("zero counts:", zeros)

    log_probs = np.log2(probs)
    perp = np.sum(log_probs) / len(log_probs)
    print("perplexity:", perp)

    fig = plt.figure()
    plt.plot(probs)
    plt.savefig("vocab_probs.png")
    plt.close(fig)

    # Entropy
    entropy = 0
    for i in range(len(probs)):
        entropy += probs[i] * log_probs[i]
    print("entropy:", entropy)



def top_5(data_raw):
    """ data_raw == (trials, 15, 5001) """
    # 1. encode captions
    return

def bleu_histogram(bleu_scores):

    scores = [i[1] for i in bleu_scores]
    fig = plt.figure()
    plt.hist(scores, bins = np.arange(0, 1, 0.1)-0.05)

    plt.xlabel("BLEU-1 score")
    plt.ylabel("count")
    plt.title("BLEU-1 score for all 515 test samples")
    plt.savefig("./Eval/caption_samples/bleu_1_histogram.png", bbox_inches='tight')

def main(model_log):

    data = load_data(model_log)
    data_max =  np.argmax(data, axis=-1) # (1000, 15)
    print("data:", data.shape)
    print("data_max:", data_max.shape)

    targets = load_captions_dict()
    print(targets[3050])

    sample_captions(data_max, "no_teacher")
    #all_bleu_scores(data_max, targets)

    bleu_scores_sorted = bleu_score_sort(data_max, targets)
    sample_captions_choice(data_max, bleu_scores_sorted, "no_teacher_bleu")

    bleu_histogram(bleu_scores_sorted)

    return

if __name__ == '__main__':
#    nsd_loader = init_nsd()

    #main("./Log/proper_split_sub2/eval_out/output_captions_raw_98.npy")
    main('./Log/no_teacher_forcing/eval_out/output_captions_raw_84.npy')
    #main('./Log/no_teacher_forcing_no_stopwords/eval_out/output_captions_raw_116.npy')




    sys.exit(0)
    data_s1 = load_data(data_sub1_file)
    data_s2 = load_data(data_sub2_file)
    data_s1_max, data_s2_max =  np.argmax(data_s1, axis=-1), np.argmax(data_s2, axis=-1) # (1000, 15)

    #train_set_perplexity(data_s2)
    #perplexity(data_s2_max)
    
    # Plot some img-cap samples
    #sample_captions(data_s2_max)

    # Load target captions from disk
    targets = load_captions_dict()

    #all_bleu_scores(data_s1_max, targets)
    all_bleu_scores(data_s2_max, targets)

    bleu_scores_sorted = bleu_score_sort(data_s2_max, targets)
    sample_captions_choice(data_s2_max, bleu_scores_sorted)

    plot_bleu(bleu_scores_sorted)
