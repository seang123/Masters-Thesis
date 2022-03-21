import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import re
import logging
import time
import tqdm
import warnings
from collections import defaultdict
import concurrent.futures


loggerA = logging.getLogger(__name__ + '.data_generator')

#subject = '2'
nsd_dir = '/home/seagie/NSD2/'
captions_path = "/fast/seagie/data/captions/"
#betas_path    = f"/fast/seagie/data/subj_{subject}/betas_averaged/"
#guse_path     = f"/fast/seagie/data/subj_{subject}/guse_averaged/"
#vgg16_path    = f"/fast/seagie/data/subj_{subject}/vgg16/"

class DataGenerator(keras.utils.Sequence):
    """ https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly """

    def __init__(self, pairs, batch_size, tokenizer, units, max_len, vocab_size, subject='2', pre_load_betas=False, shuffle=True, training=False):
        print("initialising DataGenerator")
        self.subject=subject
        self.betas_path    = f"/fast/seagie/data/subj_{self.subject}/betas_averaged/"
        self.pairs = np.array(pairs)
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.units = units
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.training = training
        self.pre_load_betas = pre_load_betas

        #self.most_active_vert = np.loadtxt("./TrainData/avg_most_active_vert.txt", dtype=np.int32)

        #self.word_to_index, self.index_to_embedding = self.load_glove()
        if pre_load_betas: 
            warnings.warn("Pre-loading betas... Make sure generator is properly configured. Doesn't work with more than 1 subject")
            self.nsd_to_idx = {}
            keys = np.array(list(set([i[0] for i in self.pairs])))
            self.betas = self.preload_data(keys)

        self.on_epoch_end()

    def preload_data(self, keys):
        # Not corrently implemented when using more than one subject
        betas = np.zeros((keys.shape[0], 327684), dtype=np.float32)
        for i, key in tqdm.tqdm(enumerate(keys)):
            self.nsd_to_idx[str(key)] = i
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

    def load_glove(self):
        word_to_index = dict()
        index_to_embedding = []
        data_dir = './chakin_embeddings/'
        file_name = 'glove.twitter.27B.25d'
        with open(f"{data_dir}/{file_name}.txt", "r") as f:
            for (i, line) in enumerate(f):
                split = line.split(' ')
                word = split[0]
                representation = split[1:]
                #representation = np.array([float(val) for val in representation])
                representation = np.array(representation, dtype=np.float32)

                word_to_index[word] = i
                index_to_embedding.append(representation)
        
        # Start token
        i = i + 1
        word_to_index['<start>'] = i
        temp = [0.0] * len(representation)
        temp[0] = 1.0
        index_to_embedding.append(temp)
        # End token
        i = i + 1
        word_to_index['<end>'] = i
        temp = [0.0] * len(representation)
        temp[-1] = 1.0
        index_to_embedding.append(temp)

        # Unknown word
        _word_not_found = [0.0] * len(representation)
        _last_index = i + 1
        #word_to_index = defaultdict(mod_level_lambda) # necessary for pickling (can't pickla lambda)
        word_to_index = defaultdict(lambda: _last_index, word_to_index)
        index_to_embedding = np.array(index_to_embedding + [_word_not_found])
        print(len(word_to_index))
        print(index_to_embedding.shape)
        return word_to_index, index_to_embedding

    def glove_embeddings(self, cap):
        max_len = self.max_len
        embedding = np.zeros((cap.shape[0], max_len, 25), dtype=np.float32)
        for i, c in enumerate(cap):
            for k, word in enumerate(c):
                if k >= max_len:
                    continue
                embedding[i, k] = self.index_to_embedding[self.word_to_index[word]]
        return embedding


    def __getitem__(self, index):
        """ Return one batch """

        batch = self.pairs[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(batch)

    def __data_generation(self, batch):
        """ Generates data cointaining batch_size samples

        Takes a batch from the pairs array and returns the appropriate data
        """
        nsd_key, cap, guse_key, count, sub = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,-1]
        batch_size = nsd_key.shape[0]
        #count   = count.astype(np.int32)

        # Pre-allocate memory
        betas_batch = np.zeros((batch_size, 327684), dtype=np.float32)
        guse_batch  = None # np.zeros((nsd_key.shape[0], 512), dtype=np.float32)
        vgg_batch   = None # np.zeros((nsd_key.shape[0], 4096), dtype=np.float32)

        # Load data 
        #if self.pre_load_betas: 
        #    for i, key in enumerate(nsd_key):
        #        betas_batch[i, :] = self.betas[self.nsd_to_idx[key]]
        #else:
        for i, key in enumerate(nsd_key):
            if sub[i] == '1':
                with open(f"/fast/seagie/data/subj_1/betas_averaged/subj01_KID{key}.npy", "rb") as f:
                    betas_batch[i, :] = np.load(f)
            else:
                with open(f"/fast/seagie/data/subj_2/betas_averaged/subj02_KID{key}.npy", "rb") as f:
                    betas_batch[i, :] = np.load(f)

        # Tokenize captions
        cap_seqs = self.tokenizer.texts_to_sequences(cap) # int32
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs, maxlen = self.max_len, truncating = 'post', padding = 'post')

        # Create target
        target = np.zeros_like(cap_vector, dtype=cap_vector.dtype)
        target[:,:-1] = cap_vector[:,1:]
        target = to_categorical(target, self.vocab_size)

        # Init LSTM
        init_state = tf.zeros([batch_size, self.units], dtype=np.float32)
        
        if self.training:
            return ((betas_batch, cap_vector, init_state, init_state), target)
        else:
            return ((betas_batch, cap_vector, init_state, init_state), target, nsd_key)







