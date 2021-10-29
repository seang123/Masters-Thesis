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
                            ReLU
                            )
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


class NIC(tf.keras.Model):
    """ Overall the same as NIC, except using Locally Connected input

    Mostly the same as normal NIC model, but with takes more patched fMRI data
    Not just visual cortex, but visual cortex split into 41 regions
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
        self.dropout = Dropout(dropout)
        self.dropout_input = Dropout(dropout_input)
        self.dropout_text = Dropout(dropout_text)

        self.relu = LeakyReLU(0.2)


        # For locally connected and concated output
        self.dense_in = localDense.LocallyDense(
                in_groups,
                out_groups,
                embed_dim = embedding_dim,
                activation=self.relu,
                kernel_initializer='he_normal',
                kernel_regularizer=self.l2_in,
                #kernel_constraint=tf.keras.constraints.MaxNorm(max_value=1), 
                #bias_constraint=tf.keras.constraints.MaxNorm(max_value=1),
        )

        # When using attention
        """
        self.dense_in = layers.LocallyDense(
                in_groups, 
                out_groups, 
                activation=self.relu,
                kernel_initializer='he_normal',
                kernel_regularizer=self.l2_in,
                #name = 'lc_dense_in'
        )

        self.attention = attention.Attention(
                use_bias = True,
                kernel_initializer=GlorotNormal(),
                name = 'attention'
        )
        """

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))

        # Text input
        #self.input2 = Input(shape=(max_length,))
        self.embedding = Embedding(vocab_size, 
                embedding_dim, 
                mask_zero=True,
                name = 'emb_text',
        )

        # LSTM layer
        self.lstm = LSTM(units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=self.l2_lstm,
                dropout=dropout,
                name = 'lstm'
        )

        # Output dense layer
        self.dense_out = TimeDistributed(
                Dense(vocab_size,
                    activation='softmax',
                    kernel_regularizer=self.l2_out,
                    #kernel_initializer=RandomUniform(-0.08, 0.08),
                    kernel_initializer=GlorotNormal(),
                ),
                name = 'time_distributed_softmax'
        )

    def attend(self, hidden_state, features):
        return

    def call_attention(self, data, training=False):
        img_input, text_input, a0, c0 = data

        if training:
            img_input = self.dropout_input(img_input)

        # Features from regions
        features = self.dense_in(img_input, training) 
        if training:
            features = self.dropout(features)

        # Embed the caption vector
        text = self.embedding(text_input) # (bs, max_len, embed_dim)
        if training:
            text = self.dropout_text(text)

        # init state
        a = tf.convert_to_tensor(a0) # (bs, embed_dim)
        c = tf.convert_to_tensor(c0)

        output = []
        # Pass through LSTM
        for i in range(text.shape[1]):
            # compute attention context
            context = self.attention(a, features, training)
            context = self.expand(context) # (bs, 1, embed_dim)

            # combine context with word
            sample = tf.concat([context, tf.expand_dims(text[:, i, :], axis=1)], axis=-1) # (bs, 

            _, a, c = self.lstm(sample, initial_state=[a,c], training=training)
            output.append(a)

        output = tf.stack(output, axis=1) # (bs, max_len, embed_dim)

        # Convert to vocab
        output = self.dense_out(output) # (bs, max_len, vocab_size)

        return output


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

        img_input, text_input, a0, c0 = data

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
        _, a, c = self.lstm(features, initial_state=[a0,c0], training=training)
        A, _, _ = self.lstm(text, initial_state=[a,c], training=training)

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

        features = self.dense_in(img_input)
        features = self.expand(features)

        text = self.embedding(start_seq)
        text = self.expand(text)

        # Call LSTM on features
        whole, final, c = self.lstm(features, initial_state=[a0, c0])

        outputs = []
        print("---")
        for i in range(max_len-1):
            whole, final, c = self.lstm(text, initial_state=[final, c])

            output = self.dense_out(whole)
            #outputs.append( output )
            text = tf.math.argmax(output, axis=2)
            outputs.append(text)
            text = self.embedding(text)


        return np.array(outputs)

    def nongreedy_predict(self, img_input, a0, c0, start_seq, max_len, units):
        """ Use non greedy methods when making predictions

        Returns more than one predicted sentence that we can then choose from
        """

        features = self.dense_in(img_input)
        features = self.attention(features)
        features = self.expand(features)

        text = self.embedding(start_seq)
        text = self.expand(text)

        whole, final, c = self.lstm(features, initial_state=[a0, c0])

        output = self.dense_out(whole)
        print("whole ", whole.shape)
        print("output", output.shape)
        top_p, probs = self.select_nucleus(output, p=0.5)

        if not isinstance(top_p, list):
            top_p = [top_p]

        outputs = []
        for k, p in top_p:
            outputs.append( _nongreedy_predict(p, a0, c0, max_len, [], 1) )

        



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

        img_input, text_input, a0, c0 = data[0]
        target = data[1] # (batch_size, max_length, 5000)

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction = self(
                    (
                        img_input, 
                        text_input, 
                        a0, 
                        c0
                    ), 
                    training=True
            ) # (64, 10, 5000) (bs, max-length, vocab_size)

            # Get the loss
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence
            cross_entropy_loss = (cross_entropy_loss / int(target.shape[1]))
            accuracy = accuracy / int(target.shape[1])

            if len(self.losses) != 0:
                l2_loss += tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss = tf.add(cross_entropy_loss, l2_loss)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    #   summed_squares = [tf.math.reduce_sum(tf.math.square(g)) for g in gradients]
    #   norm = tf.math.sqrt(sum(summed_squares))
        
        return {"loss": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy}

    @tf.function()
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
        
        img_input, text_input, a0, c0 = data[0]
        target = data[1]

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        # Call model on sample
        prediction = self(
                (
                    img_input, 
                    text_input, 
                    a0, 
                    c0
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

        if len(self.losses) != 0:
            l2_loss += tf.add_n(self.losses)

        return {"loss": cross_entropy_loss, "L2": l2_loss, 'accuracy': accuracy}

    def loss_function(self, real, pred):
        """ Call the compiled loss function """
        loss_ = self.compiled_loss(real, pred)
        return tf.reduce_mean(loss_)

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
        







