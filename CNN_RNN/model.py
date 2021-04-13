# Holds the CNN-encoder and RNN-decoder models 

# imports
import numpy as np
import pickle
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import time
import json
import pandas as pd
import logging
import tensorflow as tf
import string
import time
import tqdm
from itertools import zip_longest
import h5py

# CNN-encoder
# takes the CNN features and passes them through a Fully-connected layer
class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        """
        in_shape - tuple representing the input shape
        """
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.Model):
    """
    Attention mechanism
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
            self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis = 1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis = 1)

        return context_vector, attention_weights



# RNN-decoder
# Recurrent Network (currently without attention) 
class RNN_Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # embed vocab into a continuous, distributed representation 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Recurrent layer
        self.gru = tf.keras.layers.GRU(self.units, return_sequences = True, return_state=True, recurrent_initializer='glorot_uniform')

        # Fully Connected layers 
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden = None, training = True):
        # x        - a caption (or batch of captions) - (64, 1)
        # features - the image features               - (64, 64, 256)
        # hidden   - previous recurrent state         - (64, 512)
        # context_vector                              - (64, 256)
        # x after embedding                           - (64, 1, 256)
        # x after concat                              - (64, 1, 512)

        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        #x = tf.concat([tf.expand_dims(tf.reduce_sum(features, 1), 1), x], axis = -1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
