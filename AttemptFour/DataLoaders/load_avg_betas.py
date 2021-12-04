
import os, sys
sys.path.append("/home/seagie/NSD/Code/Masters-Thesis/soloist/Modified-Show-And-Tell-Keras")
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import my_utils as uu
import time
import re
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import collections 
from ian_code import nsd_get_data as ngd 
import yaml
import nibabel as nb
from concurrent.futures import ThreadPoolExecutor
#from data_generator_guse import DataGenerator


np.random.seed(42)

data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/fast/seagie/data/subj_2/captions/"
betas_path = "/fast/seagie/data/subj_2/betas_averaged/"
USE_ENTIRE_CORTEX = True

## ====== Glasser ======
GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

glasser_lh = nb.load(GLASSER_LH).get_data()
glasser_rh = nb.load(GLASSER_RH).get_data()

glasser = np.vstack((glasser_lh, glasser_rh)).flatten()

print("glasser_lh", glasser_lh.shape)
print("glasser_rh", glasser_rh.shape)
print("glasser   ", glasser.shape)

visual_parcels = pd.read_csv(VISUAL_MASK, index_col=0)
visual_parcel_list = list(visual_parcels.values.flatten())

if USE_ENTIRE_CORTEX == False:
    ## If using only visual cortex
    groups = []
    glasser_indices = np.array(range(len(glasser)))
    for i in visual_parcel_list:
        group = glasser_indices[glasser==i]
        groups.append(group)
else:
    ## If using entire cortex
    groups = []
    glasser_indices = np.array(range(len(glasser)))
    for i in set(glasser):
        group = glasser_indices[glasser == i]
        groups.append(group)

print("sum of groups sizes:", sum([len(g) for g in groups]))
print("Avg. group size:    ", np.mean([len(g) for g in groups]))
print("nr of groups        ", len([len(g) for g in groups]))

def get_groups(out_dim):
    return groups[1:], [out_dim for i in range(1,len(groups))]
    #return groups[1:], [len(g)//50 for g in groups[1:]]
## =====================

def remove_stop_words(list_of_words: list):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    return [i for i in list_of_words if i not in stop_words]

def build_tokenizer(subjects: list, top_k = 5000):
    """
    Parameters
    ----------
        captions_path : str
            path to folder with captions
        top_k : int
            vocab size

    Returns
    -------
        tokenizer : keras.Tokenizer
        all_captions : list 
    """

    all_captions = []
    avg_caption_length = 0

    num_files = 0

    for i, subj in enumerate(subjects):
        cap_path = f"/fast/seagie/data/subj_{subj}/captions/"
        for entry in os.scandir(cap_path):
            num_files += 1
            with open(entry.path, "r") as f:
                content = f.read()
                for i in content.splitlines():
                    cap = i.replace(".", " ")
                    cap = cap.replace(",", " ")
                    cap = cap.strip()
                    cap = cap.split(" ")
                    cap = [i.lower() for i in cap if i != '']
                    #cap = remove_stop_words(cap)
                    cap = ['<start>'] + cap + ['<end>']
                    avg_caption_length += len(cap)
                    cap = " ".join(cap)

                    all_captions.append(cap)
                
    print(f"num_files scanned: {num_files}")
    print(f"num captions read: {len(all_captions)}")
    print(f"avg caption length: {avg_caption_length/len(all_captions)}")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token = '<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer, all_captions

def get_nsd_keys(subj: str = '2') -> (list, list):
    """ Get the NSD keys for a subject 

    Parameter
    ---------
        subj : int | str
            the subject id
    Returns
    -------
        unq : ndarray 
            unique nsd keys
        shr : ndfarray
            shared nsd keys
    """

    df = pd.read_csv(f'./TrainData/subj0{subj}_conditions.csv')

    unq = df['nsd_key'].loc[df['is_shared']==0]
    shrd = df['nsd_key'].loc[df['is_shared']==1]
    
    assert len(unq) == 9000, "incorrect amount of unq keys"
    assert len(shrd) == 1000, "incorrect amount of shrd keys"

    return unq.values, shrd.values

def get_all_nsd_keys(subjects: list):

    nsd_keys = []
    for i, subj in enumerate(subjects):
        unq_keys, shr_keys = get_nsd_keys(subj)
        nsd_keys.append([unq_keys, shr_keys])
    return nsd_keys
        

def get_shr_nsd_keys(nsd_dir: str) -> list:
    """ Get the shared NSD keys """
    return ngd.get_1000(nsd_dir)

def create_pairs(keys: list, subj: str = '2'):
    """ returns NSD_key - caption pairs

    Parameters
    ----------
        nsd_keys : list
            list of unique nsd keys
        subj: int or str
            an id of the current subject - should match the passed keys 
        captions_path : str
            path to the caption files
    Returns
    -------
        pairs : list
            [NSD key, caption, caption id, key_idx, subj_id]
    """
    cap_path = f"/fast/seagie/data/subj_{subj}/captions/"

    pairs = []
    for count, key in enumerate(keys):
        with open(f"{cap_path}SUB{subj}_KID{key}.txt", "r") as f:
            content = f.read()
            cid = 0
            for line in content.splitlines():
                cap = line.replace(".", " ")
                cap = cap.replace(",", " ")
                cap = cap.strip()
                cap = cap.split(" ")
                cap = [i.lower() for i in cap]
                #cap = remove_stop_words(cap)
                cap = ['<start>'] + cap + ['<end>']
                cap = " ".join(cap)
                pairs.append( (key, cap, cid, count, subj) )
                cid += 1

    return pairs

def create_all_pairs(subjects: list):
    """ Create pairs for all subjects """
    pairs = []
    for i, subj in enumerate(subjects):
        unq_keys, shr_keys = get_nsd_keys(subj)
        train_pairs = create_pairs(unq_keys, subj)
        val_pairs   = create_pairs(shr_keys, subj)
        pairs.append([train_pairs, val_pairs])
    return pairs

def train_val_pairs_from_all(all_pairs):

    train_pairs = []
    val_pairs = []
    for i in range(len(all_pairs)):
        train_pairs.append(all_pairs[i][0])
        val_pairs.append(all_pairs[i][1])
    train_pairs = [i for sublist in train_pairs for i in sublist]
    val_pairs = [i for sublist in val_pairs for i in sublist]
    
    return train_pairs, val_pairs

def main():

    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
        print(f"Config file loaded.")

    start = time.time()
    tokenizer, _ = build_tokenizer(config['top_k'])
    print(f"Tokenizer built in {(time.time() - start):.2f} seconds")

    nsd_keys, shr_nsd_keys = get_nsd_keys('2') # (10_000,)
    #shr_nsd_keys = get_shr_nsd_keys(nsd_dir) # (1000,)

    print("len(set(nsd_keys))", len(list(set(nsd_keys))))
    print("len(set(shr_nsd_keys))", len(list(set(shr_nsd_keys))))

    train_keys = nsd_keys
    val_keys   = shr_nsd_keys

    print("train_keys:",len(train_keys))
    print("val_keys:  ",len(val_keys))

    train_pairs = np.array(create_pairs(train_keys, 2))
    val_pairs = np.array(create_pairs(val_keys, 2))

    print("train_pairs:", train_pairs.shape)
    print("val_pairs:  ", val_pairs.shape)

    
    all_pairs = create_all_pairs([1,2]) # [ [[train1], [val1]], [[train2], [val2]] ]

    train_pairs, val_pairs = train_val_pairs_from_all(all_pairs)
    print(len(train_pairs))
    print(len(val_pairs))

    raise


    #start = time.time()
    #print(f"Time to tokenize captions: {(time.time() - start):.2f}")

    generator = DataGenerator(
            train_pairs,
            batch_size = 32,
            tokenizer = tokenizer,
            units = 512,
            max_len = 10,
            vocab_size = 5001,
            nsd_keys = nsd_keys,
            pre_load_betas=False,
            shuffle=False,
            training=False)

    x = generator[0]
    print(x[0][0].shape)

    return

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time: {(time.time() - start):.2f}")


