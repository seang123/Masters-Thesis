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
import tqdm

"""
    Data Generator for loading VGG16 / InceptionV3 features

"""

loggerA = logging.getLogger(__name__ + '.data_generator')

nsd_dir = '/home/seagie/NSD2/'
captions_path = "/fast/seagie/data/subj_2/captions/"
#betas_path    = "/fast/seagie/data/subj_2/betas_averaged/"
guse_path     = "/fast/seagie/data/subj_2/guse_averaged/"
vgg16_path    = "/fast/seagie/data/subj_2/vgg16/"
inception_v3_path = "/huge/seagie/data/inception_v3/"

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

        self.nsd_to_idx = {}

        keys = np.array(list(set([i[0] for i in self.pairs])))
        for i, k in enumerate(keys):
            self.nsd_to_idx[str(k)] = i

        #self.guse = self.load_guse()
        if pre_load_betas: 
            warnings.warn("Pre-loading betas... Make sure generator is properly configured. Doesn't work with more than 1 subject")
            self.features = self.preload_data(keys)

        self.on_epoch_end()

    def load_helper(self, key, i):
        self.nsd_to_idx[str(key)] = i
        with open(f"{inception_v3_path}/KID{key}.npy", "rb") as f:
            return np.reshape(np.load(f), (64, 2048)), i

    def preload_data(self, keys):

        features = np.zeros((keys.shape[0], 64, 2048), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for i, key in enumerate(keys):
                futures.append(executor.submit(self.load_helper, key, i))
            for future in concurrent.futures.as_completed(futures):
                data, ii = future.result()
                features[ii] = data

        loggerA.info("betas pre-loaded into memory")
        return features

    def preload_data2(self, keys):
        # Not corrently implemented when using more than one subject
        features = np.zeros((keys.shape[0], 64, 2048), dtype=np.float32)
        for i, key in tqdm.tqdm(enumerate(keys), desc = 'loading data into memory', total=keys.shape[0]):
            #self.nsd_to_idx[str(key)] = i
            with open(f"{inception_v3_path}/KID{key}.npy", "rb") as f:
                features[i,:] = np.reshape(np.load(f), (64, 2048))

        loggerA.info("betas pre-loaded into memory")
        return features

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

        nsd_key, cap, guse_key, count = batch[:,0], batch[:,1], batch[:,2], batch[:,3]

        #count   = count.astype(np.int32)

        betas_batch = None #np.zeros((nsd_key.shape[0], 327684), dtype=np.float32)
        guse_batch  = None # np.zeros((nsd_key.shape[0], 512), dtype=np.float32)
        vgg_batch   = None #np.zeros((nsd_key.shape[0], 4096), dtype=np.float32)
        inception_batch = np.zeros((nsd_key.shape[0], 64, 2048), dtype=np.float32)

        for i, key in enumerate(nsd_key):
            #with open(f"/huge/seagie/data/inception_v3/KID{key}.npy", "rb") as f:
            idx = self.nsd_to_idx[str(key)]
            inception_batch[i] = self.features[idx]

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
            return ((inception_batch, cap_vector, init_state, init_state), target)
        else:
            return ((inception_batch, cap_vector, init_state, init_state), target, nsd_key)







