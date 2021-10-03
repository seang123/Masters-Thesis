
import os, sys
sys.path.append("/home/seagie/NSD/Code/Masters-Thesis/soloist/Modified-Show-And-Tell-Keras")
import numpy as np
from nsd_access import NSDAccess
import pandas as pd
import my_utils as uu
import time
from parameters import params_dir
import re
import keras
from keras.utils import to_categorical
import collections 

np.random.seed(42)

"""
    1. Loop through the caption files
        a. Save the captions and initialize the tokenizer
    2. Create (betas_file, trial, caption) pairs 
    3. These pairs go too the batch loader 
        a. Batch loader reads the betas, and relevant caption
        b. Caption is passed through the tokenizer
        c. Batch loader returns [beta, caption_vector, target] + some other info ie. init lstm state

"""

#response = pd.read_csv("/home/seagie/NSD2/nsddata/ppdata/subj02/behav/responses.tsv", sep='\t', header=0)
stim_info_merged = pd.read_csv("/home/seagie/NSD2/nsddata/experiments/nsd/nsd_stim_info_merged.csv") # (73000, n)



captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path    = "/huge/seagie/data/subj_2/betas/"


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
    print(all_captions[0])

    tokenizer = keras.preprocessing.text.Tokenizer(num_words = top_k, oov_token = '<unk>', filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')
    tokenizer.fit_on_texts(all_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer, all_captions

def get_nsd_keys(betas_path, split_percentage = 0.9):
    """ Get the NSD keys

    Parameters
    ----------
        betas_path : str
            path to betas folder
        split_percentage : float
            train / validation split ratio

    Returns 
    -------
        kids : list
            list of nsd keys (30,000)
        kids_left : list
            n % of the kids 
        kids_right : list
            1 - n % of the kids list

    """

    # A list containing the keys that are part of the shared 1000 set
    shared_idx = stim_info_merged.index[stim_info_merged.shared1000 == True].tolist()

    kids = []

    for entry in os.scandir(betas_path): # 30_000 files
        kid = re.search(r"(?<=([KID]))[0-9]+", entry.path).group(0) # 73k ID
        kids.append(kid)

    kids_set = list(set(kids)) # (10,000)
    kids_left = kids_set[:int(len(kids_set) * split_percentage)]
    kids_right = kids_set[int(len(kids_set) * split_percentage):]

    #kids_left = [i for i in kids if i in kids_left]
    #kids_right = [i for i in kids if i in kids_right]

    # TODO: os.scandir is not consistent in how it loads data -> One way to get consistency would be to sort the loaded KID's 

    return kids, kids_left, kids_right

def temp_rename(kids):
    """ Adds a index 0,1,2 to the end of the betas file name
    """

    set_kids = list(set(kids))


    ls = os.listdir(betas_path)

    c = 0
    for key in set_kids:
        idx = 0
        for count, entry in enumerate(ls):
            kid = re.search(r"(?<=([KID]))[0-9]+", entry).group(0) # 73k ID
            if int(kid) == int(key):
                dst = entry.split(".")
                dst = dst[0] + f"_{idx}.npy"
                os.rename(os.path.join(betas_path,entry), os.path.join(betas_path,dst))
                idx += 1
                del ls[count]
        idx = 0
        c += 1
        print(c, end='\r')


def create_pairs(nsd_keys: list, betas_path: str, captions_path: str, subject: str = "SUB2"):
    """ Given a set of nsd_keys create [nsd_key, trial, caption] pairs 

    Trial is between [0-3)

    Each nsd_key has 3 betas associated with it, and each beta has 5 captions 

    TODO: Need to rethink this: rather than storing just nsd_key, storing the associated betas path as is 
    listed inthe captions file would make more sense for later in the batch loader

    Parameters
    ----------
        nsd_keys : list
            the unique nsd 73kid keys .ie. set(of all trial keys)
        betas_path : str
            path to betas
        captions_path : str
            path to captions
        subject : str
            which subject to load

    Returns
    -------
        pairs : list of tuples
            NSD 73k ID - trial - caption    pairs
    """

    assert( len(nsd_keys) < 30000 ), "Should be using set(nsd_keys)"
    pairs = []
    
    # 30_000 Images shown
    for i, key in enumerate(nsd_keys):

        # Each image has 4 or 5 captions
        with open(f"{captions_path}/{subject}_KID{key}.txt", "r") as f:
            content = f.read()
            for line in content.splitlines():
                cap = line.split("\t")[1] # [betas_file_name + #, caption]
                cap = cap.split(" ")
                cap = [i.lower() for i in cap]
                cap = ['<start>'] + cap + ['<end>']
                cap = " ".join(cap)

                # Get betas path
                #for count, entry in enumerate(ls): 
                #    kid = re.search(r"(?<=([KID]))[0-9]+\_[0-9]", entry).group(0)

                pairs.append((key, 0, cap))
                pairs.append((key, 1, cap))
                pairs.append((key, 2, cap))

    # Loop through betas (30_000)
    #   For one betas file get the caption file
    #       Loop through the file - create (betas_file, caption) pair
    ls = os.listdir(betas_path)

    for count, entry in enumerate(ls)
    

    print(f"{len(pairs)} - key-idx-caption pairs generated")

    return pairs


def temp_2():
    print("temp_2")

    ls = os.listdir(betas_path)

    i = "40118"

    pattern = r"(?<=([KID]))%s" % i

    print(pattern)

    for count, entry in enumerate(ls): 
        if entry == "betas_SUB2_S37_R9_T9_KID40118_2.npy":
            print(entry)
        kid = re.search(pattern, entry) # 73k ID
#        kid = re.search(r"(?<=([KID]))[0-9]+\_[0-9]", entry).group(0)

        if not kid is None:
            print(kid)

        break
    return


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



def batch_generator(pairs: list, betas_path: str, captions_path: str, batch_size = 128 ):
    """ Given a list of 73KID-trial-caption, load the relevant betas file and yield one batch of data 


    Parameters
    ----------
        pairs : list
            NSD-trial-captions
        betas_path : str
            path to betas
        captions_path : str
            path to captions
        batch_size : int

    Returns
    -------
        
    """
    
    pairs = np.array(pairs) # (N, 3)
    print("> batch_generator: ", pairs.shape) 

    N = pairs.shape[0]

    beta_file_template = f"{betas_path}betas_"
    betas_SUB2_S37_R9_T9_KID40118_2.npy
    
    beta_files = os.listdir(betas_path)

    while True:

        for i in range(0, N, batch_size):

            batch = pairs[i:i+batch_size,:]
            key, trial, cap = batch[:,0], batch[:,1], batch[:,2]

            # Load betas (mask out VC)
            betas_data = np.zeros((batch_size, 327684))

            for i in range(0, len(key)):
                n_key   = key[i]
                n_trial = trial[i]
                print(n_key, n_trial)
                sys.exit(0)



            # Tokenize caption

            # Init LSTM state
            init_state = np.zeros([text.shape[0], unit_size])

            # Create target


    return


def main():
#    tokenizer, _ = build_tokenizer(captions_path, params_dir['top_k'])
#    print(tokenizer.num_words)

    nsd_keys_all, nsd_keys_set_train, nsd_keys_set_val = get_nsd_keys(betas_path)
    print("nsd_keys_all:", len(nsd_keys_all))
    print("nsd_keys_set_train:", len(nsd_keys_set_train))
    print("nsd_keys_set_val:", len(nsd_keys_set_val))

    train_pairs = create_pairs(nsd_keys_set_train, betas_path, captions_path)
    val_pairs   = create_pairs(nsd_keys_set_val, betas_path, captions_path)

#    batch_generator(val_pairs, betas_path, captions_path, 10)

    temp_2()


if __name__ == '__main__':
    main()







