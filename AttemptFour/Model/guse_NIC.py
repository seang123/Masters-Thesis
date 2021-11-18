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
    """
    Use the GUSE embeddings as input to the LSTM instead of the brain
    """

    def __init__(self, in_groups, out_groups, units, embedding_dim, vocab_size, max_length, dropout_input, dropout, dropout_text, input_reg, lstm_reg, output_reg):
        """ Initialisation method. Builds the keras.Model
        Parameters
        ----------
        """
        super(NIC, self).__init__()


        # L2 Regularizers
        self.l2_in = L2(input_reg)
        self.l2_lstm = L2(lstm_reg)
        self.l2_out = L2(output_reg)
        #self.dropout = Dropout(dropout)
        self.dropout_input = Dropout(dropout_input)
        self.dropout_text = Dropout(dropout_text)

        #self.relu = ReLU()

        self.MSE = tf.keras.losses.MeanSquaredError()

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))

        # Text input
        self.embedding = Embedding(vocab_size, 
            embedding_dim, 
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
            #activity_regularizer=self.l2_lstm,
            #kernel_initializer=RandomUniform(-0.08, 0.08),
            #recurrent_initializer=RandomUniform(-0.08, 0.08),
            dropout=dropout,
            name = 'lstm'
        )

        # Output dense layer
        self.dense_out = TimeDistributed(
            Dense(
                vocab_size,
                activation='softmax',
                kernel_regularizer=self.l2_out,
                #kernel_initializer=RandomUniform(-0.08,0.08),
                kernel_initializer=GlorotNormal(),
                bias_initializer='zeros',
                #activity_regularizer=self.l2_out,
            ),
            name = 'time_distributed_softmax'
        )
        logging.debug("Model initialized")


    def call(self, data, training=False):
        """ Forward Pass
        
        Parameters
        ----------
            img_input : ndarray
                the features (betas/CNN) (327684,)
            text_input : ndarray
                caption, integer encoded
            a0 : ndarray
                LSTM hidden state
            c0 : ndarray
                LSTM carry state

        Returns
        -------
            output : ndarray
                a (batch_size, max_len, vocab_size) array holding predictions
        """

        _, text_input, a0, c0, guse = data[0], data[1], data[2], data[3], data[4]

        guse = self.dropout_input(guse, training=training)

        guse = self.expand(guse)

        # Embed the caption vector
        text = self.embedding(text_input)
        text = self.dropout_text(text, training=training)

        a0 = tf.convert_to_tensor(a0)
        c0 = tf.convert_to_tensor(c0)

        # Pass through LSTM
        #A, _, _ = self.lstm(tf.concat([features, text], axis=1), initial_state=[a0, c0], training=training)
        _, a, c = self.lstm(guse, initial_state=[a0, c0], training=training)
        A, _, _ = self.lstm(text, initial_state=[a, c], training=training)

        # Convert to vocab
        output = self.dense_out(A, training=training)

        return output


    def greedy_predict(self, img_input, a0, c0, start_seq, max_len, units, tokenizer):
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

        #features = self.dense_in(img_input)
        #features = self.expand(features)
        features = img_input

        text = self.embedding(start_seq)
        text = self.expand(text)

        print("features", features.shape)
        # Call LSTM on features
        whole, final, c = self.lstm(features, initial_state=[a0, c0])

        outputs = []
        for i in range(max_len):
            whole, final, c = self.lstm(text, initial_state=[final, c])

            output = self.dense_out(whole)
            outputs.append( output )
            text = tf.math.argmax(output, axis=2)
            text = self.embedding(text)

        return np.array(outputs)

    @tf.function()
    def train_step(self, data):
        """ Single backprop train step 
        
        Parameters
        ----------
            data : tuple
                holds the features, caption, init_state, and target 

        Returns
        -------
            dict
                loss/accuracy metrics
        """
        target = data[1] # (batch_size, max_length, 5000)

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        #print("tf.executing_eagerly() ==", tf.executing_eagerly() )

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction = self(
                    (
                        data[0]
                    ), 
                    training=True
            ) # (bs, max-length, vocab_size)

            # Get the loss
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence
            cross_entropy_loss /= int(target.shape[1])
            accuracy /= int(target.shape[1])

            #if len(self.losses) != 0:
            l2_loss = tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss = tf.add(cross_entropy_loss, l2_loss)

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

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        # Call model on sample
        prediction = self(
                (
                    data[0]
                ),
                training=False
        )

        # Get the loss
        for i in range(0, target.shape[1]):
            cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
            accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

        # Normalise across sentence
        cross_entropy_loss /= int(target.shape[1])
        accuracy /= int(target.shape[1])

        #if len(self.losses) != 0:
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
        """ Compute Accuracy

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







