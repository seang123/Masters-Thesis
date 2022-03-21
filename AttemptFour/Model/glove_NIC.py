import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tensorflow.keras.layers import (Dense,
                            LSTM,
                            BatchNormalization,
                            Dropout,
                            Embedding,
                            Input,
                            Lambda,
                            TimeDistributed,
                            LeakyReLU,
                            PReLU,
                            ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform, GlorotNormal
from tensorflow.keras.regularizers import L2
from tensorflow_addons import seq2seq
from tensorflow_addons.rnn import LayerNormLSTMCell
from . import layers
from . import deep_layers
from . import attention
from . import localDense
from . import fullyConnected
from . import agc
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

    def __init__(self, groups, units, embedding_features, embedding_text, attn_units, vocab_size, max_length, dropout_input, dropout_features, dropout_text, dropout_attn, dropout_lstm, dropout_out, input_reg, attn_reg, lstm_reg, output_reg):
        """ Initialisation method. Builds the keras.Model """
        super(NIC, self).__init__()

        # L2 Regularizers
        self.l2_in = L2(input_reg)
        self.l2_attn = L2(attn_reg)
        self.l2_lstm = L2(lstm_reg)
        self.l2_out = L2(output_reg)
        self.dropout_input = Dropout(dropout_input, name='drop_in')
        self.dropout = Dropout(dropout_features, name='drop_feat')
        self.dropout_text = Dropout(dropout_text, name='drop_text')
        self.dropout_lstm = Dropout(dropout_lstm, name='drop_lstm')
        self.dropout_output = Dropout(dropout_out, name='drop_out')

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))
        self.MSE = tf.keras.losses.MeanSquaredError()

        self.batchnorm_in = tf.keras.layers.BatchNormalization(name = 'input_bn')

        """
        self.dense_in = fullyConnected.FullyConnected(
                embed_dim = embedding_features,
                dropout = self.dropout,
                activation = LeakyReLU(0.2),
                kernel_regularizer = self.l2_in
        )
        """

        """
        # For locally connected and concated output
        self.dense_in = localDense.LocallyDense(
            in_groups,
            out_groups,
            embed_dim=embedding_features,
            dropout = self.dropout,
            activation=None, #LeakyReLU(0.2),
            kernel_initializer=RandomUniform(-0.08, 0.08), #'he_normal',
            bias_initializer='zeros',
            kernel_regularizer=self.l2_in,
        )
        """

        # Locally connected input (in conjunction with attention)
        self.dense_in = layers.LocallyDense(
            groups,
            dropout = self.dropout,
            batch_norm = self.batchnorm_in,
            activation=LeakyReLU(0.2),
            kernel_initializer='he_normal',
            kernel_regularizer=self.l2_in,
            #name = 'lc_dense_in'
        )

        # Attention Layer
        self.dropout_attn = Dropout(dropout_attn)
        self.attention = attention.Attention(
            units=attn_units,
            dropout=self.dropout_attn,
            activation=LeakyReLU(0.2),
            kernel_initializer='he_normal',
            kernel_regularizer=self.l2_attn,
            #name = 'attention'
        )

        # LSTM layer
        self.lstm = LSTM(units,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.l2_lstm,
            dropout=dropout_lstm,
            name = 'lstm'
        )
        """
        #self.lstm.trainable=False # freeze layer 
        self.lnLSTMCell = LayerNormLSTMCell(units,
                kernel_regularizer=self.l2_lstm,
                )
        self.lstm = tf.keras.layers.RNN(
                self.lnLSTMCell, 
                return_sequences=True, 
                return_state=True,
                name='lstm'
        )
        """

        # Intermediary output dense layer
        self.dense_inter = TimeDistributed(
            Dense(
                256,
                activation=LeakyReLU(0.2),
                kernel_regularizer=self.l2_out,
                kernel_initializer=GlorotNormal(),
                bias_initializer='zeros',
            ),
            name = 'time_distributed_nonlinear'
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
        #self.dense_out.trainable=False # freeze layer 
        loggerA.debug("Model initialized")

    def call(self, data, training=False):
        return self.call_attention(data, training)
        #return self.call_naive_attention(data, training)
        #return self.call_lc(data, training)
        #return self.call_fc(data, training)

    def learn_init_state(self, features):
        """ Proposed by Xu et al. init the LSTM h,c state as MLP(mean(features))"""
        h = self.hidden_init(tf.reduce_mean(features, axis=1))
        c = self.carry_init(tf.reduce_mean(features, axis=1))
        return h, c

    def call_naive_attention(self, data, training=False):
        """ Same as call_attention() but without teacher forcing """
        img_input, text_input, a0, c0 = data
        img_input = self.dropout_input(img_input, training=training)

        # Features from regions
        features = self.dense_in(img_input, training) 
        features = self.dropout(features, training=training)

        # Embed the caption vector
        text_full = self.embedding(text_input) # (bs, max_len, embed_dim)
        text = tf.expand_dims(text_full[:,0,:], axis=1)
        text = self.dropout_text(text, training=training)

        # init state
        a = tf.convert_to_tensor(a0) # (bs, units)
        c = tf.convert_to_tensor(c0)
        
        attention_scores = []
        outputs = []
        for i in range(text_full.shape[1]):
            # compute attention context
            context, attn_scores = self.attention(a, features, training=training)
            context = self.expand(context) # (bs, 1, group_size)

            #attention_scores += attn_scores
            attention_scores.append( attn_scores )

            sample = tf.concat([context, text], axis=-1) # (bs, 1, embed_features + embed_text)
            seq, a, c = self.lstm(sample, initial_state=[a,c], training=training)

            seq = self.dropout_lstm(seq, training=training)

            # Dense-Decoder
            output = self.dropout_output(seq, training=training)
            output = self.dense_inter(output, training=training) # (bs, max_len, vocab_size)
            output = self.dense_out(output) # (bs, 1, 5001)
            outputs.append(output)

            # Greedy choice
            word = tf.argmax(output, axis=-1)

            # encode the new word
            text = self.embedding(word)

        outputs = tf.squeeze(tf.stack(outputs, axis=1)) # (bs, max_len, embed_dim)
        return outputs, tf.convert_to_tensor(attention_scores)

    def call_attention(self, data, training=False):
        """ Forward pass | Attention model """
        img_input, text, a0, c0 = data

        img_input = self.dropout_input(img_input, training=training)

        # Features from regions
        features = self.dense_in(img_input, training=training) 
        features = self.dropout(features, training=training)

        # Embed the caption vector
        #text = self.embedding(text_input) # (bs, max_len, embed_dim)
        text = self.dropout_text(text, training=training)

        # init state
        a = tf.convert_to_tensor(a0) # (bs, units)
        c = tf.convert_to_tensor(c0)

        #attention_scores = tf.zeros((features.shape[0], features.shape[1], 1), dtype=tf.float32)
        attention_scores = []

        output = []
        # Pass through LSTM
        for i in range(text.shape[1]):
            # compute attention context
            context, attn_scores = self.attention(a, features, training=training)
            context = self.expand(context) # (bs, 1, group_size)

            #attention_scores += attn_scores
            attention_scores.append( attn_scores )

            # combine context with word
            sample = tf.concat([context, tf.expand_dims(text[:, i, :], axis=1)], axis=-1) # (bs, 1, embed_features + embed_text)

            _, a, c = self.lstm(sample, initial_state=[a,c], training=training)
            out = self.dropout_lstm(a, training=training)
            output.append(out)

        output = tf.stack(output, axis=1) # (bs, max_len, embed_dim)

        # Convert to vocab
        output = self.dense_inter(output, training=training)
        output = self.dropout_output(output, training=training)
        output = self.dense_out(output, training=training) # (bs, max_len, vocab_size)

        return output, tf.convert_to_tensor(attention_scores)

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

        print("features:", features.shape)
        print("text    :", text.shape)

        # Pass through LSTM
        #A, _, _ = self.lstm(tf.concat([features, text], axis=1), initial_state=[a0, c0], training=training)
        _, a, c = self.lstm(features, initial_state=[a0, c0], training=training)
        A, _, _ = self.lstm(text, initial_state=[a, c], training=training)

        # Convert to vocab
        output = self.dense_out(A, training=training)

        return output, None

    def call_fc(self, data, training=False):
        img_input, text_input, a0, c0 = data

        img_input = self.dropout_input(img_input, training=training)

        features = self.dense_in(img_input, training=training)
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(text_input)
        text = self.dropout_text(text, training=training)

        a0 = tf.convert_to_tensor(a0)
        c0 = tf.convert_to_tensor(c0)

        sample = tf.concat([features, text], axis=1) # (bs, 1, embed_features + embed_text)

        # Pass through LSTM
        A, _, _ = self.lstm(features, initial_state=[a0, c0], training=training)

        # Convert to vocab
        output = self.dense_out(A)

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
        #guse = data[0][-1]

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0
        latent_loss = 0
        total_loss = 0
        attn_loss = 0

        #print("tf.executing_eagerly() ==", tf.executing_eagerly() )

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction, attention_scores = self(
                    (
                        data[0]
                    ), 
                    training=True
            ) # (bs, max-length, vocab_size), (bs, 13, 180, 1)

            # Attention loss
            attn_across_time = tf.reduce_sum(tf.squeeze(attention_scores,axis=-1), axis=1)
            attention_target = tf.ones(attn_across_time.shape, dtype=tf.float32)
            attn_loss = self.MSE(attention_target, attn_across_time)

            # Cross-entropy loss & Accuracy
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence length
            cross_entropy_loss /= int(target.shape[1])
            accuracy /= int(target.shape[1])

            # capture regularization losses
            l2_loss = tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss += cross_entropy_loss
            total_loss += l2_loss
            #total_loss += attn_loss

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        #gradients = agc.adaptive_clip_grad(trainable_variables, gradients, clip_factor=0.01, eps = 1e-3)
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

        return {"loss": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy, 'attention': attn_loss, 'lr':self.optimizer.lr}#, grad_sum

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
        #guse = data[0][-1]

        l2_loss   = 0
        cross_entropy_loss = 0
        accuracy  = 0
        latent_loss = 0
        attn_loss = 0

        # Call model on sample
        prediction, attention_scores = self(
                (
                    data[0]
                ),
                training=False
        )

        # Attention loss
        attn_across_time = tf.reduce_sum(tf.squeeze(attention_scores,axis=-1), axis=1)
        attention_target = tf.ones(attn_across_time.shape, dtype=tf.float32)
        attn_loss = self.MSE(attention_target, attn_across_time)

        # Cross-entropy & Accuracy
        for i in range(0, target.shape[1]):
            cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
            accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

        # Normalise across sentence length 
        cross_entropy_loss /= int(target.shape[1])
        accuracy /= int(target.shape[1])

        # Regularization losses
        l2_loss = tf.add_n(self.losses)

        return {"loss": cross_entropy_loss, "L2": l2_loss, 'accuracy': accuracy, 'attention': attn_loss}

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
        features, _ = self.dense_in(img_input, training=False) 
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(start_seq)
        text = self.expand(text)

        # Initial LSTM call with brain data
        _, a, c = self.lstm(features, initial_state=[a0,c0], training=False)
        
        outputs = []
        for i in range(0, max_len-1):
            A, a, c = self.lstm(text, initial_state=[a,c], training=False)
            output = self.dense_out(A) # (bs, 1, 5001)

            words = np.argmax(output, axis=-1)
            outputs.append(words)

            # encode the new word
            text = self.embedding(words)

        return np.array(outputs)

    def greedy_predict_attention(self, img_input, a0, c0, start_seq, max_len, units, tokenizer, training=False):
        """ Make a prediction for a set of features and start token

        Should be fed directly from the data generator

        Parameters
        ----------
        Returns
        -------
            outputs : ndarray
                holds the words produced for each caption/batch
            attention-scores : ndarray

        """

        features = self.dense_in(img_input, training=training)
        features = self.dropout(features, training=training)

        text = self.embedding(start_seq)
        text = self.dropout_text(text, training=training)
        text = self.expand(text)

        a = a0
        c = c0

        attention_scores = []
        outputs = []
        outputs_raw = []
        for i in range(max_len):
            context, attention_score = self.attention(a, features, training=training)
            context = self.expand(context)
            attention_scores.append( np.array(attention_score) )

            # Plot attention

            sample = tf.concat([context, text], axis=-1)
            _, a, c = self.lstm(sample, initial_state=[a,c], training=training)
            seq = self.expand(a)
            seq = self.dropout_lstm(seq, training=training)

            # Dense-Decoder
            output = self.dense_inter(seq, training=training)
            output = self.dropout_output(output, training=training)
            output = self.dense_out(output, training=training) # (bs, 1, 5001)
            outputs_raw.append(output)

            # Greedy choice
            word = np.argmax(output, axis=-1)
            outputs.append(word)

            # encode the new word
            text = self.embedding(word)

        # outputs -> np.array == (max_len, bs, 1)
        outputs = np.stack(outputs, axis=1)
        outputs_raw = np.concatenate(outputs_raw, axis=1)
        assert outputs.shape == (features.shape[0], max_len, 1)
        return outputs, outputs_raw, np.array(attention_scores)

    def beam_search(self, features, start_seq, max_len, units):

        features = self.dense_in(features, training=False)
        print("features:", features.shape)

        text = self.embedding(start_seq)
        text = tf.squeeze(text, axis=1)
        #text = self.expand(text)
        print("text:", text.shape)

        a = tf.zeros((features.shape[0], units))
        c = tf.zeros((features.shape[0], units))

        context, attention_score = self.attention(a, features, training=False)
        sample = tf.concat([context, text], axis=-1)
        seq, a, c = self.lstm(sample, initial_state=[a,c], training=False)
        output = self.dense_inter(seq, training=False)
        output = self.dense_out(output, training=False) # (bs, 1, 5001)
        #word = np.argmax(output, axis=-1)
        words, _ = self.select_nucleus2(output, p=0.5)

        frontier = []
        for w in words:
            frontier.append((seq, [w]))

        sequences = self._beam_search(frontier)
        print("-- Beam search complete --")


    def _beam_search(self, frontier: list, max_len: int, features: np.array):

        if frontier[0][0].shape[1] == max_len:
            return frontier
        else:
            new_frontier = []
            for i, seq in enumerate(frontier):
                output = self.dense_inter(seq, training=False)
                output = self.dense_out(output, training=False) # (bs, 1, 5001)
                #word = np.argmax(output, axis=-1)
                words, _ = self.select_nucleus2(output, p=0.5)

                for j, word in enumerate(words):
                    text = self.embedding(word)
                    context, _ = self.attention(seq[:,-1,:], features, training=False)
                    sample = tf.concat([context, text], axis=-1)

                    a = tf.zeros((features.shape[0], units))
                    c = tf.zeros((features.shape[0], units))
                    seq, a, c = self.lstm(seq, initial_state=[a,c], training=False)
                    seq, _, _ = self.lstm(context, initial_state=[a,c], training=False)
                    new_frontier.append((seq, frontier[i][1] + [word]))

            self._beam_search(new_frontier, max_len, features)

    def select_nucleus2(probability_vector, p: float = 0.5) -> (list, int):
        """ 
        Sample from probability_vector untill 
        total probability of those samples is greater than p 
        """
        samples = []
        samples.append( tf.random.categorical(probability_vector, 1) )

        sum_prob = 0
        for i in samples:
            sum_prob += probability_vector[i] 

        while sum_prob < p:
            samples.append( tf.random.categorical(probability_vector, 1) )
            sum_prob += probability_vector[samples[-1]]

        return samples, sum_prob


    @tf.function()
    def train_step_sam(self, data):
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
        rho = 0.05
        eps = 1e-12

        target = data[1] # (batch_size, max_length, 5000)
        guse = data[0][-1]

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0
        latent_loss = 0
        total_loss = 0
        attn_loss = 0

        #print("tf.executing_eagerly() ==", tf.executing_eagerly() )

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction, attention_scores = self(
                    (
                        data[0]
                    ), 
                    training=True
            ) # (bs, max-length, vocab_size), (bs, 180, 1)
            attention_target = tf.ones(attention_scores.shape, dtype=tf.float32)
            attn_loss += self.MSE(attention_target, attention_scores)

            # Cross-entropy loss & Accuracy
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence length
            cross_entropy_loss /= int(target.shape[1])
            accuracy /= int(target.shape[1])

            # capture regularization losses
            l2_loss = tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss += cross_entropy_loss
            total_loss += l2_loss
            total_loss += attn_loss

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)

        # first step
        e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        for i in range(len(trainable_variables)):
            print("---")
            print("\tgra", gradients[i])
            print("\tvar", trainable_variables[i])
            if type(gradients[i]) == tf.python.framework.indexed_slices.IndexedSlices:
                e_w = gradients[i].values * rho / (grad_norm + eps)
                print("\te_w", e_w)
                #e_w = tf.reduce_sum(e_w, axis=0)
                #trainable_variables[i].assign_add(e_w)
                #e_w = tf.tensor_scatter_nd_add(trainable_variables[i], tf.expand_dims(gradients[i].indices, axis=-1), e_w)
                e_w = tf.scatter_nd(tf.expand_dims(gradients[i].indices, axis=-1), e_w, (5001, 512))
                trainable_variables[i].assign_add(e_w)
                e_ws.append(e_w)
            else:
                e_w = gradients[i] * rho / (grad_norm + eps)
                print("\te_w", e_w)
                trainable_variables[i].assign_add(e_w)
                e_ws.append(e_w)

        
        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0
        latent_loss = 0
        total_loss = 0
        attn_loss = 0

        with tf.GradientTape() as tape:
            # Call model on sample
            prediction, attention_scores = self(
                    (
                        data[0]
                    ), 
                    training=True
            ) # (bs, max-length, vocab_size), (bs, 180, 1)

            attention_target = tf.ones(attention_scores.shape, dtype=tf.float32)
            attn_loss += self.MSE(attention_target, attention_scores)

            # Cross-entropy loss & Accuracy
            for i in range(0, target.shape[1]):
                cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
                accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

            # Normalise across sentence length
            cross_entropy_loss /= int(target.shape[1])
            accuracy /= int(target.shape[1])

            # capture regularization losses
            l2_loss = tf.add_n(self.losses)

            # Sum losses for backprop
            total_loss += cross_entropy_loss
            total_loss += l2_loss

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_sub(e_ws[i])
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return {"loss": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy, 'attention': attn_loss, 'lr':self.optimizer.lr}#, grad_sum



