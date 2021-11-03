
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


np.random.seed(42)


with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")


data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path = "/huge/seagie/data/subj_2/betas_meaned/"
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

def get_groups(out_dim):
    #return groups, [out_dim for i in range(len(groups))]
    return groups, [len(g)//100 for g in groups]
## =====================

def build_tokenizer(captions_path, top_k = 5000):
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

    num_files = 0
    for entry in os.scandir(captions_path):
        num_files += 1
        with open(entry.path, "r") as f:
            content = f.read()
            for i in content.splitlines():
                cap = i.split("\t")[1] # [betas_file_name + #, caption]
                cap = cap.split(" ")
                cap = [i.lower() for i in cap]
                cap = ['<start>'] + cap + ['<end>']
                cap = " ".join(cap)

                all_captions.append(cap)
                
    print(f"num_files scanned: {num_files}")
    print(f"num captions read: {len(all_captions)}")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token = '<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer, all_captions

def get_nsd_keys(nsd_dir: str, subject: str = 'subj02', n_sessions=40) -> (list, list):
    """ Get the NSD keys for a subject 

    Returns
    -------
        sample : list
            unique nsd keys shown
        conditions : list
            all nsd keys show ( sample keys repeated 3 times )
    """

    conditions = ngd.get_conditions(nsd_dir, subject, n_sessions)
    conditions = np.asarray(conditions).ravel() # (30_000,)
    conditions_bool = [True if np.sum(conditions == x) == 3 else False for x in conditions]
    conditions_sampled = conditions[conditions_bool]
    sample = np.unique(conditions[conditions_bool]) # (10_000,)
    n_images = len(sample) # 10_000
    all_conditions = range(n_images)
    print()

    return sample, conditions

def get_shr_nsd_keys(nsd_dir: str) -> list:
    """ Get the shared NSD keys """
    return ngd.get_1000(nsd_dir)

def load_betas(data_dir: str, file_name: str):
    """ Load the betas which have been zscored and averaged across trials """

    with open(f"{data_dir}{file_name}", "rb") as f:
        betas = np.load(f)
        print("betas:", betas.shape, "loaded.")

    return betas # (327684, 10000)

def create_pairs(keys: list, captions_path: str, seed: int = 42):
    """ NSD_key - caption pairs

    Parameters
    ----------
        nsd_keys : list
            list of unique nsd keys
        captions_path : str
            path to the caption files
    Returns
    -------
        pairs : list
            [NSD key, captions]
    """

    pairs = []
    for count, key in enumerate(keys):
        with open(f"{captions_path}SUB2_KID{key}.txt", "r") as f:
            content = f.read()
            for line in content.splitlines():
                cap = line.split("\t")[1]
                cap = cap.split(" ")
                cap = [i.lower() for i in cap]
                cap = ['<start>'] + cap + ['<end>']
                cap = " ".join(cap)
                pairs.append( (key, cap) )

    np.random.seed(seed)
    np.random.shuffle(pairs)

    return pairs



def temp_rename(nsd_keys: list, dst_location = "/huge/seagie/data/subj_2/betas_meaned/"):
    """
    Re-save the (300k, 10k) betas file as 10k seperate files
    """

    betas = load_betas(data_dir, betas_file_name)

    print("betas loaded")

    for i in range(0, betas.shape[1]):

        beta = betas[:,i]
        key = nsd_keys[i]

        file_name = f"{dst_location}betas_SUB2_KID{key}.npy"
        with open(file_name, "wb") as f:
            np.save(f, beta)

        print(i, end='\r')

def apply_mask(x: np.array, mask: np.array):
    """ Apply a mask to a set of betas
        Parameters
        ----------
            x : np.array
                betas (~320k, )
            mask : np.array
                the mask indicies
        Returns
        -------
            np.array
                array with some areas masked out
    """
    return x[mask == 1]

def batch_generator(
            pairs: list,
            betas_path: str,
            captions_path: str,
            tokenizer,
            batch_size: int = 32,
            max_length: int = 10,
            vocab_size: int = 5000,
            units: int = 512,
            training: bool = True
        ): 

    with open('/home/seagie/NSD/Code/Masters-Thesis/ThinkAndTell/masks/visual_mask_lh.npy', 'rb') as f, open('/home/seagie/NSD/Code/Masters-Thesis/ThinkAndTell/masks/visual_mask_rh.npy', 'rb') as g:
        visual_mask_lh = np.load(f)
        visual_mask_rh = np.load(g)
#print(" > visual region masks loaded from file") 

    visual_mask = np.vstack([visual_mask_lh, visual_mask_rh]).squeeze()
    DIM = int(np.sum(visual_mask))


    pairs = np.array(pairs)
    N = pairs.shape[0]

    #while True:
    for i in range(0, N, batch_size):

        # Load batch
        batch = pairs[i:i+batch_size,:]
        nsd_key, cap = batch[:, 0], batch[:,1]

        betas_data = np.zeros((nsd_key.shape[0], 62756), dtype=np.float32)
        
        # Load betas + apply mask
        for i in range(0, nsd_key.shape[0]):
            key = nsd_key[i]
            with open(f"{betas_path}/betas_SUB2_KID{key}.npy", "rb") as f:
                betas_data[i,:] = apply_mask(np.load(f), visual_mask)

        # Tokenize captions
        cap_seqs = tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = max_length, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target,vocab_size)
        target = target.astype(np.int32)

        # Init LSTM
        init_state = np.zeros([cap_vector.shape[0], units], dtype=np.float32)


        if training:
           yield ((betas_data, cap_vector, init_state, init_state), target)
        else:
           yield ((betas_data, cap_vector, init_state, init_state), target, nsd_key)


def apply_lc_mask(betas:np.array, groups:list):
    """ Apply multiple masks to the betas

    A (327684,) array of betas gets split into multiple masked regions
    using the mask list

    Parameters
    ----------
        betas : ndarray
            betas array (327684,)
        groups : list
            a list of indices corresponding to different regions

    Returns
    -------
        list 
            different masked beta regions
            [(N0,), (N1,) ...., (N40,)]
    """
    
    beta_regions = []
    for i in groups:
        region = betas[i]
        beta_regions.append( region )

    print("beta_regions:", len(beta_regions))
    print("beta_regions:", [g.shape for g in beta_regions])

    return beta_regions

def lc_batch_generator(
            pairs: list,
            betas_path: str,
            captions_path: str,
            tokenizer,
            batch_size: int = 32,
            max_length: int = 10,
            vocab_size: int = 5000,
            units: int = 512,
            training: bool = True
        ): 

    pairs = np.array(pairs)
    N = pairs.shape[0]

#    while True:
    for i in range(0, N, batch_size):

        # Load batch
        batch = pairs[i:i+batch_size,:]
        nsd_key, cap = batch[:, 0], batch[:,1]

        betas_data = np.zeros((nsd_key.shape[0], 327684), dtype=np.float32)

        # Load betas + apply mask
        for i in range(0, nsd_key.shape[0]):
            key = nsd_key[i]
            with open(f"{betas_path}/betas_SUB2_KID{key}.npy", "rb") as f:
                betas_data[i, :] = np.load(f)

        # Tokenize captions
        cap_seqs = tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = max_length, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, vocab_size)

        # Init LSTM
        init_state = np.zeros([cap_vector.shape[0], units], dtype=np.float32)

        if training:
            yield ((betas_data, cap_vector, init_state, init_state), target)
        else:
           yield ((betas_data, cap_vector, init_state, init_state), target, nsd_key)


def main():

    tokenizer, _ = build_tokenizer(captions_path, config['top_k'])

    nsd_keys, _ = get_nsd_keys(nsd_dir) # (10_000,)
    shr_nsd_keys = get_shr_nsd_keys(nsd_dir) # (1000,)

    print("len(set(nsd_keys))", len(list(set(nsd_keys))))
    print("len(set(shr_nsd_keys))", len(list(set(shr_nsd_keys))))

    train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
    val_keys = shr_nsd_keys

    print(len(train_keys))
    print(len(val_keys))

    train_pairs = create_pairs(train_keys, captions_path)
    val_pairs = create_pairs(val_keys, captions_path)


    #batch_generator(val_pairs, betas_path, captions_path, tokenizer, config['batch_size'], config['max_length'], config['top_k'], config['units'])

    lc_batch_generator(val_pairs, betas_path, captions_path, tokenizer, config['batch_size'], config['max_length'], config['top_k'], config['units'])



    return

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time: {(time.time() - start):.2f}")


