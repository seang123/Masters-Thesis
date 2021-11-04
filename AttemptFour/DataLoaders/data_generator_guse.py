import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging
import time


loggerA = logging.getLogger(__name__ + '.data_generator')

data_dir = '/huge/seagie/data_meaned/'
nsd_dir = '/home/seagie/NSD2/'
subject = "subj02"
n_sessions = 40
targetspace = 'fsaverage'
betas_file_name = "subj02_betas_fsaverage_averaged.npy"
captions_path = "/huge/seagie/data/subj_2/captions/"
betas_path = "/huge/seagie/data/subj_2/betas_meaned/"
guse_path = "/huge/seagie/data/subj_2/guse/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, batch_size, tokenizer, units, max_len, vocab_size, shuffle=True, training=False):
        print("initialising DataGenerator")
        self.pairs = np.array(pairs)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.training = training

        self.on_epoch_end()

    
    def load_guse(self):
        """ Load the guse embeddings into memory """
        self.guse = np.load(open(f"{guse_path}/guse_embeddings.npy", "rb"))
        logging.info("guse embeddings loaded")


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

        nsd_key, cap, guse_key = batch[:,0], batch[:,1], batch[:,2]

        betas_batch = np.zeros((nsd_key.shape[0], 327684), dtype=np.float32)
        guse_batch = np.zeros((nsd_key.shape[0], 512), dtype=np.float32)

        for i, key in enumerate(nsd_key):
            with open(f"{betas_path}/betas_SUB2_KID{key}.npy", "rb") as f:
                betas_batch[i, :] = np.load(f)
            guse_batch[i,:] = self.guse[guse_key[i]]



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








