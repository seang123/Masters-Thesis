import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging


loggerA = logging.getLogger(__name__ + '.data_generator')

data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path = "/huge/seagie/data/subj_2/betas_meaned/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, batch_size, tokenizer, units, max_len, vocab_size, load_to_memory=True, seed=42, shuffle=True, training=False):
        print("initialising DataGenerator")
        self.pairs = np.array(pairs)
        self.unq_pair_keys = list(set([int(i[0]) for i in pairs]))
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.load_to_memory = load_to_memory # whether to pre-load betas to RAM
        self.seed = seed
        self.shuffle = shuffle
        self.training = training
        np.random.seed(self.seed)
        self.on_epoch_end()

        if self.load_to_memory:
            self.betas_data, self.nsd_idx = self.load_betas_into_mem()
    
    def load_betas_into_mem(self):
        """ Load betas into memory

        Loads the 10k betas into memory and creates a nsd_key to idx dictionary map
        """
        print("loading betas into memory")

        beta_files = os.listdir(f"{betas_path}")
        betas_data = np.zeros((len(self.unq_pair_keys), 327684), dtype=np.float32) 
        nsd_idx = {} # nsd to memory idx map
        idx = 0
        for _, v in enumerate(beta_files):
            nsd_key = int(re.search(r"(?<=([KID]))[0-9]+", v).group(0))
            if nsd_key in self.unq_pair_keys:
                with open(f"{betas_path}/betas_SUB2_KID{nsd_key}.npy", "rb") as f:
                    betas_data[idx, :] = np.load(f)
                nsd_idx[str(nsd_key)] = idx
                idx += 1

        print("betas loaded into memory successfully")
        logging.debug("betas loaded into memory successfully")
        return betas_data, nsd_idx


    def __len__(self):
        """ Nr. of batches per epoch """
        return len(self.pairs)//self.batch_size


    def on_epoch_end(self):
        """ Shuffle data when epoch ends """
        if self.shuffle:
            idxs = np.arange(0, len(self.pairs))
            np.random.shuffle(idxs)
            self.pairs = self.pairs[idxs]

        logging.info("shuffling dataset")

    def __getitem__(self, index):
        """ Return one batch """

        batch = self.pairs[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch)

    def __data_generation(self, batch):
        """ Generates data cointaining batch_size samples

        Takes a batch from the pairs array and returns the appropriate data
        """

        nsd_key, cap = batch[:,0], batch[:,1]

        betas_batch = np.zeros((nsd_key.shape[0], 327684), dtype=np.float32)

        if self.load_to_memory:
            idxs = []
            for i, v in enumerate(nsd_key):
                idx = self.nsd_idx[str(v)] # for the nsd_key get the relevant memory idx
                idxs.append(idx)
                betas_batch[i, :] = self.betas_data[idx]
            #betas_batch[:,:] = self.betas_data[idxs]
        else:
            for i, v in enumerate(nsd_key):
                key = nsd_key[i]
                with open(f"{betas_path}/betas_SUB2_KID{key}.npy", "rb") as f:
                    betas_batch[i, :] = np.load(f)


        # Tokenize captions
        cap_seqs = self.tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = self.max_len, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, self.vocab_size)

        # Init LSTM
        init_state = np.zeros([cap_vector.shape[0], self.units], dtype=np.float32)

        if self.training:
            return ((betas_batch, cap_vector, init_state, init_state), target)
        else:
            return ((betas_batch, cap_vector, init_state, init_state), target, nsd_key)








