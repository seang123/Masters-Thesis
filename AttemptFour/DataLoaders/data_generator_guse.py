import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging
import time
import warnings
import concurrent.futures


loggerA = logging.getLogger(__name__ + '.data_generator')

nsd_dir = '/home/seagie/NSD2/'
captions_path = "/fast/seagie/data/subj_2/captions/"
#betas_path    = "/fast/seagie/data/subj_2/betas_averaged/"
guse_path     = "/fast/seagie/data/subj_2/guse_averaged/"
vgg16_path    = "/fast/seagie/data/subj_2/vgg16/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, batch_size, tokenizer, units, max_len, vocab_size, pre_load_betas=False, shuffle=True, training=False):
        print("initialising DataGenerator")
        self.pairs = np.array(pairs)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.training = training
        self.pre_load_betas = pre_load_betas

        self.most_active_vert = np.loadtxt("./TrainData/avg_most_active_vert.txt", dtype=np.int32)

        #self.guse = self.load_guse()
        if pre_load_betas: 
            warnings.warn("Pre-loading betas... Make sure generator is properly configured. Doesn't work with more than 1 subject")
            #self.betas = self.load_all_betas(nsd_keys)

        self.on_epoch_end()

    def load_all_betas(self, keys):
        # Not corrently implemented when using more than one subject
        betas = np.zeros((keys.shape[0], 327684), dtype=np.float32)
        for i, key in enumerate(keys):
            with open(f"{betas_path}/subj02_KID{key}.npy", "rb") as f:
                betas[i,:] = np.load(f)

        loggerA.info("betas pre-loaded into memory")
        return betas

    def __len__(self):
        """ Nr. of batches per epoch """
        return len(self.pairs)//self.batch_size

    def on_epoch_end(self):
        """ Shuffle data when epoch ends """
        if self.shuffle:
            np.random.shuffle(self.pairs)
            loggerA.info("shuffling dataset")

    def __getitem__(self, index):
        """ Return one batch """

        batch = self.pairs[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch)

    def __data_generation(self, batch):
        """ Generates data cointaining batch_size samples

        Takes a batch from the pairs array and returns the appropriate data
        """

        nsd_key, cap, guse_key, count, sub_id = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]

        #count   = count.astype(np.int32)

        betas_batch = np.zeros((nsd_key.shape[0], 327684), dtype=np.float32)
        guse_batch  = None # np.zeros((nsd_key.shape[0], 512), dtype=np.float32)
        vgg_batch   = np.zeros((nsd_key.shape[0], 4096), dtype=np.float32)


        #if self.pre_load_betas: 
        #    betas_batch = self.betas[count,:]
        for i, key in enumerate(nsd_key):
            betas_path = f"/fast/seagie/data/subj_{sub_id[i]}/betas_averaged/"
            with open(f"{betas_path}/subj0{sub_id[i]}_KID{key}.npy", "rb") as f:
                betas_batch[i, :] = np.load(f)
            #    b_batch = np.load(f)
            #    betas_batch[i, :] = b_batch[self.most_active_vert]
            #with open(f"{vgg16_path}/SUB2_KID{key}.npy", "rb") as f:
            #    vgg_batch[i,:] = np.load(f)

        # Tokenize captions
        cap_seqs = self.tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = self.max_len, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, self.vocab_size)

        # Init LSTM
        init_state = np.zeros([nsd_key.shape[0], self.units], dtype=np.float32)


        if self.training:
            return ((betas_batch, cap_vector, init_state, init_state, vgg_batch), target)
        else:
            return ((betas_batch, cap_vector, init_state, init_state, vgg_batch), target, nsd_key)







