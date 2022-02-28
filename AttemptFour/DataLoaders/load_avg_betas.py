
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


np.random.seed(42)

data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/fast/seagie/data/captions/"
betas_path = "/fast/seagie/data/subj_2/betas_averaged/"
USE_ENTIRE_CORTEX = True
SEPARATE_HEMISPHERES = True

## ====== Glasser ======
GLASSER_LH = '/home/danant/misc/lh.HCP_MMP1.mgz'
GLASSER_RH = '/home/danant/misc/rh.HCP_MMP1.mgz'
VISUAL_MASK = '/home/danant/misc/visual_parcels_glasser.csv'

glasser_lh = nb.load(GLASSER_LH).get_data() # 163_842 values in the range [0, 180]
glasser_rh = nb.load(GLASSER_RH).get_data()

glasser = np.vstack((glasser_lh, glasser_rh)).flatten()

print("glasser_lh", glasser_lh.shape)
print("glasser_rh", glasser_rh.shape)
print("glasser   ", glasser.shape)

visual_parcels = pd.read_csv(VISUAL_MASK, index_col=0)
visual_parcel_list = list(visual_parcels.values.flatten())

if USE_ENTIRE_CORTEX == False:
    ## If using only visual cortex
    print("-- visual area glasser regions --")
    groups = []
    glasser_indices = np.array(range(len(glasser))) # [0 to 163841]
    for i in visual_parcel_list:
        group = glasser_indices[glasser==i]
        groups.append(group)
elif SEPARATE_HEMISPHERES: 
    print("-- separate hemisphere glasser regions --")
    ## Separate Glasser regions into hemisphers - 360 regions
    glasser_lh = glasser_lh.flatten()
    glasser_rh = glasser_rh.flatten()
    # Right
    glasser_indices_rh = np.array(range(len(glasser_rh))) # [0 to 163_841]
    groups_rh = []
    for i in set(glasser_rh):
        groups_rh.append(glasser_indices_rh[glasser_rh == i])
    # Left 
    glasser_indices_lh = np.array(range(len(glasser_lh))) # [0 to 163_841]
    groups_lh = []
    for i in set(glasser_lh):
        groups_lh.append(glasser_indices_lh[glasser_lh == i])
    print("excluding group 0")
    print("groups_lh:", len(groups_lh[1:]))
    print("groups_rh:", len(groups_rh[1:]))
    print("Avg. group size lh:     ", np.mean([len(g) for g in groups_lh[1:]]))
    print("Avg. group size rh:     ", np.mean([len(g) for g in groups_rh[1:]]))
    print("max group size lh | rh: ", np.max([len(g) for g in groups_lh[1:]]), np.max([len(g) for g in groups_rh[1:]]))
    print("min group size lh | rh: ", np.min([len(g) for g in groups_lh[1:]]), np.min([len(g) for g in groups_rh[1:]]))
    groups = groups_lh[1:] + groups_rh[1:]
else:
    ## If using entire cortex
    print("-- full cortex glasser regions --")
    groups = []
    glasser_indices = np.array(range(len(glasser)))
    for i in set(glasser):
        group = glasser_indices[glasser == i]
        groups.append(group)
    groups = groups[1:]
    print("sum of groups sizes:", sum([len(g) for g in groups]))
    print("Avg. group size:    ", np.mean([len(g) for g in groups]))
    print("nr of groups        ", len([len(g) for g in groups]))

def get_groups(out_dim, separate_hemi=False):
    return groups, [out_dim for i in range(0,len(groups))]
        #return groups[1:], [len(g)//50 for g in groups[1:]]
        #return groups_lh[1:], groups_rh[1:], [out_dim for i in range(1, len(groups_lh))], [out_dim for i in range(1, len(groups_rh))]

## =====================

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"{func.__name__} - {(time.perf_counter() - start):.3f} sec")
        return out
    return wrapper


def remove_stop_words(list_of_words: list):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    return [i for i in list_of_words if i not in stop_words]

@timeit
def build_tokenizer(nsd_keys: list, top_k = 5000):
    """
    Build tokenizer from captions

    Ignore captions that aren't in the keys list

    Parameters
    ----------
        nsd_keys : list
            a list of NSD keys to load
        top_k : int
            vocab size

    Returns
    -------
        tokenizer : keras.Tokenizer
        all_captions : list 
    """

    all_captions = []
    avg_caption_length = 0

    keys = set([i for i in nsd_keys])

    num_files = 0
    for entry in os.scandir(captions_path):
        num_files += 1
        kid = re.search('(?<=KID)[0-9]+', entry.name)[0]
        if int(kid) in keys:
            with open(entry.path, "r") as f:
                content = f.read()
                for i in content.splitlines():
                    cap = i.replace(".", " ").replace(",", " ").strip().split(" ")
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
        shr : ndarray
            shared nsd keys
    """

    df = pd.read_csv(f'./TrainData/subj0{subj}_conditions.csv')

    unq = df['nsd_key'].loc[df['is_shared']==0]
    shrd = df['nsd_key'].loc[df['is_shared']==1]
    
    assert len(unq) == 9000, "incorrect amount of unq keys"
    assert len(shrd) == 1000, "incorrect amount of shrd keys"

    return unq.values, shrd.values
        

def get_shr_nsd_keys(nsd_dir: str) -> list:
    """ Get the shared NSD keys """
    return ngd.get_1000(nsd_dir)

@timeit
def create_pairs(keys: list, subj='2'):
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
            [NSD key, caption, caption id, count, subject id]
    """

    pairs = []
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
                pairs.append( (key, cap, cid, count, subj) )
                cid += 1

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

    train_keys, val_keys = get_nsd_keys('2') # (10_000,)
    print("len(set(nsd_keys))", len(list(set(train_keys))))
    print("len(set(shr_nsd_keys))", len(list(set(val_keys))))

    #train_keys = list(range(1, 73001))[:70000]
    #val_keys   = list(range(1, 73001))[70000:]

    train_pairs = np.array(create_pairs(train_keys))
    val_pairs = np.array(create_pairs(val_keys))
    print("train_pairs:", train_pairs.shape)
    print("val_pairs:  ", val_pairs.shape)

    start = time.time()
    tokenizer, _ = build_tokenizer(np.concatenate((train_keys, val_keys)), config['top_k'])
    print(f"Tokenizer built in {(time.time() - start):.2f} seconds")
    

    #start = time.time()
    #print(f"Time to tokenize captions: {(time.time() - start):.2f}")

    from data_generator_guse import DataGenerator
    generator = DataGenerator(
            train_pairs,
            batch_size = 64,
            tokenizer = tokenizer,
            units = 512,
            max_len = 13,
            vocab_size = 5001,
            pre_load_betas=True,
            shuffle=True,
            training=True
    )

    x = generator[0]
    print(x[0][0].shape)

    return

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Total time elapsed: {(time.time() - start):.2f}")


