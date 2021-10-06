
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

np.random.seed(42)


data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path = "/huge/seagie/data/subj_2/betas_meaned/"

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

def create_pairs(keys: list, captions_path: str):
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

    print("> batch_generator:", pairs.shape)

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
        cap_seqs = tokenizer.texts_to_sequences(cap)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = max_length, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target,vocab_size)

        # Init LSTM
        init_state = np.zeros([cap_vector.shape[0], units], dtype=np.float32)


        if training:
           yield ([betas_data, cap_vector, init_state, init_state], target)
        else:
           yield ([betas_data, cap_vector, init_state, init_state], target)

def main():

    tokenizer, _ = build_tokenizer(captions_path, params_dir['top_k'])

    nsd_keys, _ = get_nsd_keys(nsd_dir) # (10_000,)
    shr_nsd_keys = get_shr_nsd_keys(nsd_dir) # (1000,)

    train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
    val_keys = shr_nsd_keys

    ic(len(train_keys))
    ic(len(val_keys))

    train_pairs = create_pairs(train_keys, captions_path)
    val_pairs = create_pairs(val_keys, captions_path)

    ic(len(train_pairs))
    ic(len(val_pairs))

    batch_generator(val_pairs, betas_path, captions_path, tokenizer, params_dir['batch_size'], params_dir['max_length'], params_dir['top_k'], params_dir['units'])

    return

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time: {(time.time() - start):.2f}")


