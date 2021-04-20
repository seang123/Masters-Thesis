
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
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, stateful=False, return_state=True, unit_forget_bias=True, recurrent_initializer='glorot_uniform')#(inp)

        # Fully connected layers to convert from embedding dim to vocab
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def embed(self, x):
        return self.embedding(x)

    def call(self, data, training = False):

        x, features = data

        ## feat = (128, 1, 512)
        feat = tf.expand_dims(features, 1)

        ## x = (128, 260, 512)
        x = self.embedding(x)
        ## x = (128, 261, 512)
        x = tf.concat([feat, x], axis = 1)

        ## output = (128, 261, 512)
        output, _, _ = self.lstm(x)

        ## x = (128, 261, 512)
        x = self.fc1(output)

        ## x = (128, 261, 5001)
        x = self.fc2(x)

        return x

    def old_call(self, data, training = False):
        """
            x        - a caption (word) - (bs, 1)
            features - image feature    - (bs, 256)
            hidden   - previous state   - (bs, 512)

            I just realised I always feed the image into the lstm at every time step
            the show and tell paper only feed it in at the start 

            TODO: HOLY SHIT IM NOT PASSING THE PREVIOSU HIDDEN STATE
        """
        x, hidden = data
        
        # get the sentence embedding
        #x = self.embedding(x)
        
        # input to lstm should == (bs, 1, 512)

        # concatenate with image embedding
        # x = tf.concat([tf.expand_dims(features, 1), x], axis = -1)

        # pass through RNN
        output, state, _ = self.lstm(x) #(64, 1, 512) (64,512) (64,512)

        print("---lstm---")
        print("output", output.shape)
        print("state", state.shape)
        print("carry", carry.shape)

        # return-shape = (bs, 1, units) = (64, 1, 512)
        x = self.fc1(output)

        print("x", x.shape)
        print("---------")

        # reshape to (64, 512)
        #x = tf.reshape(x, (-1, x.shape[2]))

        #print("x reshape", x.shape)

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
    def old_train_step(self, img_cap: tuple):
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

            ## Input features into LSTM
            features = tf.expand_dims(features, 1)
            predictions, hidden = self.decoder((features, hidden))
            loss += self.loss_function(target[:,0], predictions)

            for i in range(1, target.shape[1]):

                x = self.decoder.embedding(dec_input)# input -> embedding layer 
                predictions, hidden = self.decoder((x, hidden))
                loss += self.loss_function(target[:,i], predictions)

                #self.compiled_metrics.update_state(target[:,i], predictions)

                # teacher forcing
                dec_input = tf.expand_dims(target[:,i], 1)

        #self.decoder.lstm.reset_states(states=[np.zeros((target.shape[0],1,512)),np.zeros((target.shape[0],1,512))]) # reset lstm hidden state (used with stateful=True)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.decoder.trainable_variables + self.encoder.trainable_variables 

        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


        #return {m.name: m.result() for m in self.metrics}
        #return loss, total_loss
        return {"loss": loss, "norm loss": total_loss}

    @tf.function
    def train_step(self, img_cap):
        """Implemention of train_step but without for-loop (TODO: implement)
        This would make it easier in that we would no longer need the stateful=True arrtribute in the LSTM init 
        also would probably be faster 

        Instead of taking a single word(bs,1,5001) -> embedding it(bs,1,256) -> passing it into lstm (bs,1,512)
        Take all words(bs,260,5001) -> embed them(bs,260,256) -> lstm(bs,260,512)      -- 260=max_length (all captions are paddd to 260)
        """
        print("### TRACING ###")
        # decompose dataset item
        img_tensor, target = img_cap

        loss = 0

        #hidden = self.decoder.reset_state(batch_size = target.shape[0]) 

        #dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)
        #dec_input = target


        with tf.GradientTape() as tape:

            ## feature embedding
            features = self.encoder(img_tensor)
            #features = tf.expand_dims(features, 1)

            ## Word embedding | pass whole sentence into LSTM
            #x = self.decoder.embedding(dec_input)

            ## concat image to start of input (128, 261, 512)
            #x = np.concatenate((features, x), axis=1)

            ## Generate predictions (128, 261, 5001)
            #predictions, hidden = self.decoder((x, hidden))

            predictions = self.decoder((target, features))
            
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
        
        for i in range(1, self.max_length):
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
