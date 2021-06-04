
import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis')
import utils
import time
import tensorflow as tf
from tensorflow.keras.activations import relu
#from dataclass import Dataclass

class Encoder(tf.keras.Model):
    """Encoder Model.
    Takes 2nd last layer of CNN and maps it to a embedding vector
    """
    def __init__(self, embedding_dim, l2_reg, init_method, dropout):
        super(Encoder, self).__init__()
        self.l2 = l2_reg
        regularizer = tf.keras.regularizers.L2(l2_reg)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.bn = tf.keras.layers.BatchNormalization()

        self.fc = tf.keras.layers.Dense(
                embedding_dim, 
                activation='relu',
                kernel_regularizer=regularizer, 
                kernel_initializer=init_method,
                #bias_regularizer=regularizer
                )

    def call(self, x, training = False):
        if training == True:
            return self.dropout(self.fc(x))
        return self.fc(x)

    def summary(self):
        #x = tf.keras.Input(shape=(62756,))
        x = tf.keras.Input(shape=(5000,))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size, l2_reg, init_method, dropout, use_stateful=False):
        super(Decoder, self).__init__()

        self.units = units
        self.embedding_dim = embedding_dim
        regularizer = tf.keras.regularizers.L2(l2_reg)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=init_method, embeddings_regularizer=regularizer)#mask_zero = True)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.bn = tf.keras.layers.BatchNormalization()

        # LSTM layer
        self.lstm = tf.keras.layers.LSTM(
                units, 
                return_sequences=True, 
                stateful=False, 
                return_state=True, 
                unit_forget_bias=True, 
                kernel_initializer=init_method,
                recurrent_initializer=init_method, 
                kernel_regularizer=regularizer,
        )

        # Fully connected layers to convert from embedding dim to vocab
        self.fc1 = tf.keras.layers.Dense(units, activation='relu')#, kernel_regularizer=regularizer)
        self.fc2 = tf.keras.layers.Dense(vocab_size, activation='relu')#, kernel_regularizer=regularizer)

    def call(self, data, training = False):
        """Main call method
        training - should be true except when evaluating model word for word
        """
        words, features = data

        ## feat = (None, 1, 512)
        feat = tf.expand_dims(features, 1)

        ## x = (None, words+1, 512)
        x = self.embedding(words)
        #mask = self.embedding.compute_mask(words) # (1, 15)

        ## x = (None, words+1, 512)
        x = tf.concat([feat, x], axis = 1)

        ## output = (None, words+1, 512)
        output, hidden, carry = self.lstm(x)#, mask = mask)

        ## x = (None, words+1, 512)
        if training == True:
            x = self.dropout(self.fc1(self.dropout(output)))
        else:
            x = self.fc1(output)

        ## x = (None, words+1, 5001)
        x = self.fc2(x)

        return x, hidden, carry

    def layer_summary(self):
        for l in self.layers:
            print(l.summary())


    def summary(self):
        x = tf.keras.Input(shape=(15)) # word embeddings
        y = tf.keras.Input(shape=(self.embedding_dim)) # image 
        model = tf.keras.Model(inputs=[x, y], outputs=self.call((x, y)))
        return model.summary()

class CaptionGenerator(tf.keras.Model):

    def __init__(self, encoder, decoder, tokenizer, max_length):
        super(CaptionGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.tokenizer = tokenizer
        self.max_length = max_length

    @tf.function
    def train_step(self, img_cap):
        """Train step 

        Train whole network on an image + target caption input 

        Need to be using stateful = False in the lstm init

        Parameter
        --------
            img_cap : tuple (tf.tensor, tf.tensor)

        Return
        ------
            loss : dict (l1, l2)
        """
        # decompose dataset item
        img_tensor, _, target = img_cap

        loss = 0 # scce loss
        model_loss = 0 # L2 loss

        with tf.GradientTape() as tape:

            ## feature embedding
            features = self.encoder(img_tensor, training = True)

            predictions, _, _ = self.decoder((target, features), training=True) # (None, 16, 5001)

            # Loop through the sentences to get the loss
            for i in range(1, target.shape[1]): # (None, 15)
                loss += self.loss_function(target[:,i], predictions[:,i-1]) # maybe predictions[:,i-1]

            # normalised sparse categorical cross entropy loss
            scce = (loss / int(target.shape[1]))

            if len(self.encoder.losses) != 0:
                model_loss += tf.add_n(self.encoder.losses) # fc reg loss
            if len(self.decoder.losses) != 0:
                model_loss += tf.add_n(self.decoder.losses) # lstm reg loss
            
            total_loss = scce + model_loss
            
        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            
        return {"scce": scce, "loss": total_loss}

    @tf.function
    def test_step(self, data):
        """ Testing step
        """

        img_tensor, _, target = data

        loss = 0
        total_loss = 0

        features = self.encoder(img_tensor)
        predictions, _, _ = self.decoder((target, features))

        for i in range(1, target.shape[1]):
            loss += self.loss_function(target[:,i], predictions[:,i-1])

        scce = (loss / int(target.shape[1]))

        total_loss += scce
        if len(self.encoder.losses) != 0:
            total_loss += tf.add_n(self.encoder.losses) # fc reg loss
        if len(self.decoder.losses) != 0:
            total_loss += tf.add_n(self.decoder.losses) # lstm reg loss

        return {"scce": scce, "loss": total_loss}

    def loss_function(self, real, pred):
        """ Loss function

        real - (bs, 1) the i-th word of a captions for all batches
        pred - (bs, vs) a value for all words in the vocab  
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.compiled_loss(real, pred)#, regularization_losses=self.losses)
        
        #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        #loss_ = loss_fn(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


