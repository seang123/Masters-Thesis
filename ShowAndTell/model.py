
import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis')
import utils
import time
import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = tf.nn.relu(self.fc(x))
        return x


class Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size, tokenizer, optimizer, loss_object):
        super(Decoder, self).__init__()
        self.units = units

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_object = loss_object

        self.encoder = Encoder(embedding_dim)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)

        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        """
            x        - a caption (word) - (bs, 1)
            features - image feature    - (bs, 4096)
            hidden   - previous state   - (bs, 512)
        """
        
        # get the sentence embedding
        x = self.embedding(x)

        # concatenate with image embedding
        x = tf.concat([tf.expand_dims(features, 1), x], axis = -1)

        # pass through RNN
        output, state, carry = self.lstm(x)

        # return-shape = (bs, 1, units) = (64, 1, 512)
        x = self.fc1(output)


        # reshape to (64, 512)
        x = tf.reshape(x, (-1, x.shape[2]))

        # map RNN output to vocab
        x = self.fc2(x)

        return x, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    @tf.function
    def train_step(self, img_tensor, target, training=False):
        loss = 0

        hidden = self.reset_state(batch_size = target.shape[0]) 

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:

            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                predictions, hidden = self.call(dec_input, features, hidden)
                loss += self.loss_function(target[:,i], predictions)

                # teacher forcing
                dec_input = tf.expand_dims(target[:,i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.trainable_variables + self.encoder.trainable_variables 

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
