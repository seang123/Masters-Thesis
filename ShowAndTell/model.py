
import numpy as np
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis')
import utils
import time
import tensorflow as tf
from dataclass import Dataclass

class Encoder(tf.keras.Model):
    """Encoder Model.
    Takes 2nd last layer of CNN and maps it to a embedding vector
    """
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = tf.nn.relu(self.fc(x))
        return x


class Decoder(tf.keras.Model):

    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        #inp = tf.keras.layers.Input(shape=(1,512), batch_size = 128)
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero = True)
        # LSTM layer
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, stateful=False, return_state=True, unit_forget_bias=True, recurrent_initializer='glorot_uniform')

        # Fully connected layers to convert from embedding dim to vocab
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, data, training = False):
        """Main call method
        training - should be true except when evaluating model word for word
        """

        words, features = data

        ## feat = (128, 1, 512)
        feat = tf.expand_dims(features, 1)

        ## x = (128, 260, 512)
        x = self.embedding(words)
        mask = self.embedding.compute_mask(words)
        ## x = (128, 261, 512)
        if training:    
            x = tf.concat([feat, x], axis = 1)

        ## output = (128, 261, 512)
        output, hidden, carry = self.lstm(x, mask = mask)

        ## x = (128, 261, 512)
        x = self.fc1(output)

        ## x = (128, 261, 5001)
        x = self.fc2(x)

        return x, hidden, carry

    def old_call(self, data, training=False):
        x = data
        output, _, _ = self.lstm(x)
        x = self.fc1(output)
        x = self.fc2(x)
        return x

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class CaptionGenerator(tf.keras.Model):

    def __init__(self, encoder, decoder, tokenizer, max_length):
        super(CaptionGenerator, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.tokenizer = tokenizer
        self.max_length = max_length

    #@tf.function
    def old_train_step(self, img_cap):

        img_tensor, target = img_cap

        loss = 0

        hidden = self.decoder.reset_state(batch_size = target.shape[0])
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']]* target.shape[0], 1)

        with tf.GradientTape() as tape:

            features = self.encoder(img_tensor)
            features = tf.expand_dims(features, 1)
            prediction = self.decoder(features)
            loss += self.loss_function(target[:,0], prediction)

            for i in range(1, target.shape[1]):
                x = self.decoder.embedding(dec_input)
                prediction = self.decoder(x)
                loss += self.loss_function(target[:,i], prediction)

                dec_input = tf.expand_dims(target[:,i], 1)
            
        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables 

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # reset state
        self.decoder.lstm.reset_states()

        return {"loss": loss, "norm loss": total_loss}


    @tf.function
    def train_step(self, img_cap):
        """Main train step 
        This would make it easier in that we would no longer need the stateful=True arrtribute in the LSTM init 
        also would probably be faster 

        Instead of taking a single word(bs,1,5001) -> embedding it(bs,1,256) -> passing it into lstm (bs,1,512)
        Take all words(bs,260,5001) -> embed them(bs,260,256) -> lstm(bs,260,512)      -- 260=max_length (all captions are paddd to 260)
        """
        #print("## Tracing graph ##")
        # decompose dataset item
        img_tensor, target = img_cap

        loss = 0

        with tf.GradientTape() as tape:

            ## feature embedding
            features = self.encoder(img_tensor)

            predictions, _, _ = self.decoder((target, features), training=True)

            
            #ls = []
            #for i in target[0,:].numpy():
            #    ls.append(self.tokenizer.index_word[i])
            #print(ls)
            
            ## Loop through the sentences to get the loss
            for i in range(1, target.shape[1]):
                loss += self.loss_function(target[:,i], predictions[:,i])
        
        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables 

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
            
        return {"loss": loss, "norm loss": total_loss}

    @tf.function
    def test_step(self, data):

        img_tensor, target = data

        loss = 0

        features = self.encoder(img_tensor)
        predictions, _, _ = self.decoder((target, features), training=True)

        for i in range(1, target.shape[1]):
            loss += self.loss_function(target[:,i], predictions[:,i])

        total_loss = (loss / int(target.shape[1]))

        return {"loss": loss, "norm loss": total_loss}



    @tf.function
    def old_test_step(self, data):
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
        prediction = self.decoder(features)
        loss += self.loss_function(target[:,0], prediction)
        
        for i in range(1, self.max_length):
            x = self.decoder.embedding(dec_input)# pass the input throuhg the embedding layer now already (instead of in call) 
            prediction, hidden = self.decoder(x)
            loss += self.loss_function(target[:,i], prediction) 

            predicted_id = tf.random.categorical(prediction, 1, dtype=tf.int32)

            #dec_input = predicted_id #tf.expand_dims([predicted_id], 1)
            # TODO: Do we use teacher forcing in test??
            dec_input = tf.expand_dims(target[:,i], 1)

        total_loss = (loss / int(target.shape[1]))

        return {"loss": loss, "norm loss": total_loss}

    def loss_function(self, real, pred):
        """ Loss function

        real - (bs, 1) the i-th word of a captions for all batches
        pred - (bs, vs) a value for all words in the vocab  
        """
        #print("-----loss function------")
        #print("real", real.shape, real.numpy())
        #print("pred", pred.shape, pred.numpy())
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        #print("mask", mask.numpy())
        ##loss_ = self.loss_object(real, pred)
        loss_ = self.compiled_loss(real, pred, regularization_losses=self.losses)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        #print("mask cast", mask.numpy())
        #print("loss", loss_.numpy())

        return tf.reduce_mean(loss_)


