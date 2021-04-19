
import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis')
import utils
import time
import tensorflow as tf
from dataclass import Dataclass


class Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = tf.nn.relu(self.fc(x))
        return x


class Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, unit_forget_bias=True, recurrent_initializer='glorot_uniform') 

        # unit forget bias set forgetting bias to 1 at initilisation

        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, data, training = False):
        """
            x        - a caption (word) - (bs, 1)
            features - image feature    - (bs, 256)
            hidden   - previous state   - (bs, 512)

            I just realised I always feed the image into the lstm at every time step
            the show and tell paper only feed it in at the start 
        """
        x, hidden = data
        
        # get the sentence embedding
        #x = self.embedding(x)
        
        # input to lstm should == (bs, 1, 512)

        # concatenate with image embedding
        # x = tf.concat([tf.expand_dims(features, 1), x], axis = -1)

        # pass through RNN
        output, state, carry = self.lstm(x) #(64, 1, 512) (64,512) (64,512)

        # return-shape = (bs, 1, units) = (64, 1, 512)
        x = self.fc1(output)


        # reshape to (64, 512)
        x = tf.reshape(x, (-1, x.shape[2]))

        # map RNN output to vocab
        x = self.fc2(x)

        return x, state
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class CaptionGenerator(tf.keras.Model):

    def __init__(self, encoder, decoder, tokenizer, max_length):
        super(CaptionGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.tokenizer = tokenizer
        self.max_length = max_length

    #def compile(self, optimizer, loss_fn):
    #    # Overwrite Keras.compile method
    #    super(Decoder, self).compile()
    #    self.optimizer = optimizer
    #    self.loss_object = loss_fn
    #    #self.metrics = metric

    @tf.function
    def train_step(self, img_cap: tuple):
        """
        Overwrites the keras.fit train_step
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

        On the first step we want to pass in the image embedding.
        On every subsequent step we want to pass in the decoded word 

        TODO:
            can this step be done without forloop as in: www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt

        """

        # decompose dataset item
        img_tensor, target = img_cap

        loss = 0

        hidden = self.decoder.reset_state(batch_size = target.shape[0]) 

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:

            features = self.encoder(img_tensor)


            features = tf.expand_dims(features, 1)
            predictions, hidden = self.decoder((features, hidden))
            loss += self.loss_function(target[:,0], predictions)

            for i in range(0, target.shape[1]):

                x = self.decoder.embedding(dec_input)# pass the input throuhg the embedding layer now already (instead of in call) 
                predictions, hidden = self.decoder((x, hidden))
                loss += self.loss_function(target[:,i], predictions)

                #self.compiled_metrics.update_state(target[:,i], predictions)

                # teacher forcing
                dec_input = tf.expand_dims(target[:,i], 1)


        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables 

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


        #return {m.name: m.result() for m in self.metrics}
        #return loss, total_loss
        return {"loss": loss, "norm loss": total_loss}

    @tf.function
    def test_step(self, data):
        """
        Evaluation function
        """
        img_tensor, target = data

        loss = 0

        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        ## Pass image into LSTM - to init state
        features = self.encoder(img_tensor)
        features = tf.expand_dims(features, 1)
        _, hidden = self.decoder((features, hidden))
        #loss += self.loss_function(target[:,0], prediction)
        
        for i in range(0, self.max_length):
            x = self.decoder.embedding(dec_input)# pass the input throuhg the embedding layer now already (instead of in call) 
            prediction, hidden = self.decoder((x, hidden))
            loss += self.loss_function(target[:,i], prediction) 

            predicted_id = tf.random.categorical(prediction, 1, dtype=tf.int32)

            #dec_input = predicted_id #tf.expand_dims([predicted_id], 1)
            # TODO: Do we use teacher forcing in test??
            dec_input = tf.expand_dims(target[:,i], 1)

        total_loss = (loss / int(target.shape[1]))

        return {"loss": loss, "norm loss": total_loss}

    @tf.function
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        #loss_ = self.loss_object(real, pred)
        loss_ = self.compiled_loss(real, pred, regularization_losses=self.losses)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
