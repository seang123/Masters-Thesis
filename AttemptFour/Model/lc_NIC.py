import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                            LSTM,
                            BatchNormalization,
                            Dropout,
                            Embedding,
                            Input,
                            Lambda,
                            TimeDistributed,
                            LeakyReLU,
                            ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform, GlorotNormal
from tensorflow.keras.regularizers import L2
from tensorflow_addons import seq2seq
from . import layers
from . import attention
from . import localDense
import sys
import numpy as np
from collections import defaultdict
import logging

loggerA = logging.getLogger(__name__ + '.lc_model')

class NIC(tf.keras.Model):
    """ Overall the same as NIC, except using Locally Connected input
    Mostly the same as normal NIC model, but with takes more patched fMRI data
    Not just visual cortex, but visual cortex split into 41 regions
    """

    def __init__(self, in_groups, out_groups, units, embedding_features, embedding_text, attn_units, vocab_size, max_length, dropout_input, dropout_features, dropout_text, dropout_attn, dropout_lstm, input_reg, attn_reg, lstm_reg, output_reg):
        """ Initialisation method. Builds the keras.Model """
        super(NIC, self).__init__()


        # L2 Regularizers
        self.l2_in = L2(input_reg)
        self.l2_attn = L2(attn_reg)
        self.l2_lstm = L2(lstm_reg)
        self.l2_out = L2(output_reg)
        self.dropout_input = Dropout(dropout_input)
        self.dropout = Dropout(dropout_features)
        self.dropout_text = Dropout(dropout_text)

        #self.relu = ReLU()
        #self.MSE = tf.keras.losses.MeanSquaredError()
        #self.cos_sim = tf.keras.losses.CosineSimilarity() 

        # For locally connected and concated output
        self.dense_in = localDense.LocallyDense(
            in_groups,
            out_groups,
            embed_dim=embedding_features,
            dropout = self.dropout,
            activation=None,
            kernel_initializer=RandomUniform(-0.08, 0.08), #'he_normal',
            bias_initializer='zeros',
            kernel_regularizer=self.l2_in,
        )

        """
        self.dropout_attn = Dropout(dropout_attn)
        # When using attention
        self.dense_in = layers.LocallyDense(
            in_groups, 
            out_groups, 
            activation=None,
            kernel_initializer='he_normal',
            kernel_regularizer=self.l2_in,
            #name = 'lc_dense_in'
        )

        self.attention = attention.Attention(
            units = attn_units,
            #use_bias = True,
            dropout = self.dropout_attn,
            kernel_initializer=GlorotNormal(),
            kernel_regularizer=self.l2_attn,
            #name = 'attention'
        )
        """

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))

        # Text input
        self.embedding = Embedding(vocab_size, 
            embedding_text, 
            embeddings_initializer=RandomUniform(-0.08, 0.08),
            mask_zero=True,
            #activity_regularizer=self.l2_in,
            name = 'emb_text',
        )

        # LSTM layer
        self.lstm = LSTM(units,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.l2_lstm,
            #kernel_initializer=RandomUniform(-0.08, 0.08),
            #recurrent_initializer=RandomUniform(-0.08, 0.08),
            dropout=dropout_lstm,
            name = 'lstm'
        )

        # Output dense layer
        self.dense_out = TimeDistributed(
            Dense(
                vocab_size,
                activation='softmax',
                kernel_regularizer=self.l2_out,
                kernel_initializer=GlorotNormal(),
                bias_initializer='zeros',
            ),
            name = 'time_distributed_softmax'
        )
        loggerA.debug("Model initialized")

    def call(self, data, training=False):
        #return self.call_attention(data, training)
        return self.call_lc(data, training)

    def call_attention(self, data, training=False):
        """ Forward pass | Attention model """
        img_input, text_input, a0, c0, _ = data

        img_input = self.dropout_input(img_input, training=training)

        # Features from regions
        features = self.dense_in(img_input, training) 
        features = self.dropout(features, training=training)

        # Embed the caption vector
        text = self.embedding(text_input) # (bs, max_len, embed_dim)
        text = self.dropout_text(text, training=training)

        # init state
        a = tf.convert_to_tensor(a0) # (bs, embed_dim)
        c = tf.convert_to_tensor(c0)

        output = []
        # Pass through LSTM
        for i in range(text.shape[1]):
            # compute attention context
            context, _ = self.attention(a, features, training)
            context = self.expand(context) # (bs, 1, embed_features)

            # combine context with word
            sample = tf.concat([context, tf.expand_dims(text[:, i, :], axis=1)], axis=-1) # (bs, 1, embed_features + embed_text)

            _, a, c = self.lstm(sample, initial_state=[a,c], training=training)
            output.append(a)

        output = tf.stack(output, axis=1) # (bs, max_len, embed_dim)

        # Convert to vocab
        output = self.dense_out(output) # (bs, max_len, vocab_size)

        return output, features


    def call_lc(self, data, training=False):
        """ Forward Pass | locally connected without attention """

        img_input, text_input, a0, c0, _ = data

        img_input = self.dropout_input(img_input, training=training)

        # Betas Encoding 
        features = self.dense_in(img_input, training=training) 
        features = self.dropout(features, training=training)
        # Attend to embeddings
        #features = self.attention(features, training)
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(text_input)
        text = self.dropout_text(text, training=training)

        a0 = tf.convert_to_tensor(a0)
        c0 = tf.convert_to_tensor(c0)

        # Pass through LSTM
        #A, _, _ = self.lstm(tf.concat([features, text], axis=1), initial_state=[a0, c0], training=training)
        _, a, c = self.lstm(features, initial_state=[a0, c0], training=training)
        A, _, _ = self.lstm(text, initial_state=[a, c], training=training)

        # Convert to vocab
        output = self.dense_out(A, training=training)

        return output


    @tf.function()
    def train_step(self, data):
        """ Single backprop train step 
        
        Parameters
        ----------
            data : tuple
                holds the features, caption, init_state, guse, and target 

        Returns
        -------
            dict
                loss/accuracy metrics
        """

        target = data[1] # (batch_size, max_length, 5000)
        guse = data[0][-1]

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0
        guse_loss = 0
        guse_sim = 0
        total_loss = 0

        #print("tf.executing_eagerly() ==", tf.executing_eagerly() )

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction = self(
                    (
                        data[0]
                    ), 
                    training=True
            ) # (bs, max-length, vocab_size)


            # GUSE loss

            prediction = tf.reshape(prediction, (-1, prediction.shape[2]))
            target = tf.reshape(target, (-1, target.shape[2]))
            cross_entropy_loss = self.loss_function(target, prediction)
            accuracy = self.accuracy_calculation(target, prediction)
            """
            # Cross-entropy loss & Accuracy
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence length
            cross_entropy_loss /= int(target.shape[1])
            accuracy /= int(target.shape[1])
            """

            # capture regularization losses
            l2_loss = tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss += cross_entropy_loss
            total_loss += l2_loss

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        
        #cc = 0
        #for grad in gradients:
        #    if cc % 2 == 0:
        #       print(">", grad.name, "--", grad.shape)
        #   else:
        #        print(grad.name, "--", grad.shape)
        #    cc += 1
        #raise Exception("stop")
        #grad_sum = []
        #cc = 0
        #for grad in gradients:
        #    #if cc % 2 == 0:
        #    grad_sum.append( tf.reduce_sum(tf.math.square(grad), axis=0).numpy() ) # first part of Euclidean norm
        #    #grad_sum.append(tf.reduce_mean(grad, axis=0).numpy())
        #    #cc += 1

        return {"loss": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy}#, grad_sum

    @tf.function
    def test_step(self, data):
        """ Called during validation 

        Parameters
        ----------
            data : tuple
                holds the features, caption, init_state, and target 

        Returns
        -------
            dict
                loss/accuracy metrics
        """
        
        target = data[1]
        guse = data[0][-1]

        l2_loss   = 0
        cross_entropy_loss = 0
        accuracy  = 0
        guse_loss = 0
        guse_sim  = 0

        # Call model on sample
        prediction = self(
                (
                    data[0]
                ),
                training=False
        )

        # GUSE loss

        prediction = tf.reshape(prediction, (-1, prediction.shape[2]))
        target = tf.reshape(target, (-1, target.shape[2]))
        cross_entropy_loss = self.loss_function(target, prediction)
        accuracy = self.accuracy_calculation(target, prediction)

        """
        # Cross-entropy & Accuracy
        for i in range(0, target.shape[1]):
            cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
            accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

        # Normalise across sentence length 
        cross_entropy_loss /= int(target.shape[1])
        accuracy /= int(target.shape[1])
        """

        # Regularization losses
        l2_loss = tf.add_n(self.losses)

        return {"loss": cross_entropy_loss, "L2": l2_loss, 'accuracy': accuracy}

    @tf.function
    def loss_function(self, real, pred):
        """ Call the compiled loss function """
        real = tf.convert_to_tensor(real)
        loss_ = self.compiled_loss(real, pred)
        return tf.reduce_mean(loss_)

    @tf.function
    def accuracy_calculation(self, real, pred):
        """ Compute categorical accuracy

        Parameters
        ----------
            real : ndarray - one-hot
                (batch-size, vocab-size) 
            pred : ndarray - float
                (batch-size, vocab-size)
        Returns
        -------
            count : float
                accuracy value across batches
        """
        real_arg_max = tf.math.argmax(real, axis = 1) 
        pred_arg_max = tf.math.argmax(pred, axis = 1)
        count = tf.reduce_sum(tf.cast(real_arg_max == pred_arg_max, tf.float32))
        return count / real_arg_max.shape[0]

    def cosine_similarity(self, x, y):
        """ Cosine similarity between two vectors
        1.0 if x==y
        """
        num = sum([i * j for (i,j) in zip(x, y)])
        den_x = np.sqrt(sum(np.power(x,2)))
        den_y = np.sqrt(sum(np.power(y,2)))
        return num / (den_x * den_y)

    def cosine_loss(self, x, y):
        """ Return the cosine similarity as a loss value 

        The value ranges between 1 and 0, and is 0 if x == y
        """
        return 1 - self.cosine_similarity(x, y)

    def greedy_predict(self, *args, **kwargs):
        #return self.greedy_predict_lc(*args, **kwargs)
        return self.greedy_predict_attention(*args, **kwargs)

    def greedy_predict_lc(self, img_input, a0, c0, start_seq, max_len, units, tokenizer):

        # Betas Encoding 
        features, latent = self.dense_in(img_input, training=False) 
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(start_seq)
        text = self.expand(text)

        # Initial LSTM call with brain data
        _, a, c = self.lstm(features, initial_state=[a0,c0], training=False)

        outputs = []
        for i in range(max_len):
            A, a, c = self.lstm(text, initial_state=[a,c], training=False)
            output = self.dense_out(A) # (bs, 1, 5001)

            word = np.argmax(output, axis=-1)
            # TODO: non-greedy
            outputs.append(output)

            # encode the new word
            text = self.embedding(word)

        return np.array(outputs)

    def greedy_predict_attention(self, img_input, a0, c0, start_seq, max_len, units, tokenizer):
        """ Make a prediction for a set of features and start token

        Should be fed directly from the data generator

        Parameters
        ----------
            img_input : ndarray
                the features (fMRI betas/CNN)
            start_seq : ndarray
                starting token to seed the caption output
            a0 : ndarray
                initial state hidden
            c0 : ndarray
                initial state carry
            max_len : int
                the max caption length to produce
            units : int
                nr of hidden units in the LSTM 

        Returns
        -------
            outputs : ndarray
                holds the words produced for each caption/batch

        """

        features = self.dense_in(img_input)

        text = self.embedding(start_seq)
        text = self.expand(text)

        a = a0
        c = c0

        outputs = []
        for i in range(max_len):
            context, _ = self.attention(a, features, training=False)
            context = self.expand(context)

            sample = tf.concat([context, text], axis=-1)
            A, a, c = self.lstm(sample, initial_state=[a,c], training=False)
            output = self.dense_out(A) # (bs, 1, 5001)

            word = np.argmax(output, axis=-1)
            # TODO: non-greedy
            outputs.append(output)

            # encode the new word
            text = self.embedding(word)

        return np.array(outputs)
        

    @staticmethod
    def select_topk(probability_vector, k: int = 5):

        idxs = np.argsort(probability_vector)
        return idxs[:k]

    @staticmethod
    def select_nucleus(probability_vector, p: float = 0.5):
        """ Selects top-p choices from a probability vector

        Similary to how top-k selects the top k elements, top-p
        selecets the top k elements such that their sum is >= p

        Note: shouldn't take batched data. ie. only single batch 
        """
        probability_vector = np.squeeze(probability_vector) 

        idxs = np.argsort(probability_vector)
        vals, probs, cumsum = [], [], 0.0
        for _, v in enumerate(idxs):
            vals.append(v)
            probs.append(probability_vector[v])
            cumsum += probability_vector[v]
            if cumsum > p:
                return vals, probs






