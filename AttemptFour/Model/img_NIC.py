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
                            PReLU,
                            ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform, GlorotNormal
from tensorflow.keras.regularizers import L2
from tensorflow_addons import seq2seq
from tensorflow_addons.rnn import LayerNormLSTMCell
from . import layers
from . import attention
from . import localDense
from . import fullyConnected
from . import img_localDense
from . import agc
import sys
import numpy as np
from collections import defaultdict
import logging

loggerA = logging.getLogger(__name__ + '.lc_model')

class NIC(tf.keras.Model):
    """ Overall the same as NIC, except using Locally Connected input
            
            Used for training on image 
                LSTM and output layer should be same as lc_NIC model if pre-training
    """

    def __init__(self, group_size, units, embedding_features, embedding_text, attn_units, vocab_size, max_length, dropout_input, dropout_features, dropout_text, dropout_attn, dropout_lstm, dropout_out, input_reg, attn_reg, lstm_reg, output_reg):
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
        self.dropout_lstm = Dropout(dropout_lstm)
        self.dropout_out = Dropout(dropout_out)

        #self.relu = ReLU()
        self.MSE = tf.keras.losses.MeanSquaredError()

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))

        # For use with:  attention
        self.dense_in = img_localDense.LocallyDense(
            n_features = 512,
            embed_dim = group_size,
            dropout = self.dropout,
            activation=LeakyReLU(0.2),
            kernel_initializer='he_normal',
            kernel_regularizer=self.l2_in,
            #name = 'lc_dense_in'
        )

        """
        # No attention model 
        self.bn = BatchNormalization()
        self.dense_in_1 = Dense(
            512,
            activation=LeakyReLU(0.2),
            kernel_regularizer=self.l2_in,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            name = 'dense'
        )
        """

        self.dropout_attn = Dropout(dropout_attn)
        self.attention = attention.Attention(
            units = attn_units,
            dropout = self.dropout_attn,
            activation= LeakyReLU(0.2),
            kernel_initializer='he_normal',
            kernel_regularizer=self.l2_attn,
            #name = 'attention'
        )

        # Text input
        self.embedding = Embedding(
            vocab_size, 
            embedding_text, 
            embeddings_initializer=RandomUniform(-0.08, 0.08),
            mask_zero=True,
            #activity_regularizer=self.l2_in,
            name = 'emb_text',
        )

        # LSTM layer
        self.lstm = LSTM(
            units,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=self.l2_lstm,
            dropout = dropout_lstm,
            name = 'lstm'
        )
        """
        self.lnLSTMCell = LayerNormLSTMCell(units)
        self.lstm = tf.keras.layers.RNN(
                self.lnLSTMCell, 
                return_sequences=True, 
                return_state=True,
                #kernel_regularizer=self.l2_lstm,
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
        loggerA.debug("Model initialized")

    def call(self, data, training=False):
        return self.call_attention(data, training)
        #return self.call_lc(data, training)
        #return self.call_fc(data, training)


    def call_attention(self, data, training=False):
        """ Forward pass | Attention model """
        # Get input
        img_input, text_input, a0, c0 = data
        img_input = self.dropout_input(img_input, training=training)

        # Features from regions
        features = self.dense_in(img_input, training=training) 
        features = self.dropout(features, training=training)
    
        # Embed the caption vector
        text = self.embedding(text_input) # (bs, max_len, embed_dim)
        text = self.dropout_text(text, training=training)

        # init state
        a = tf.convert_to_tensor(a0) # (bs, embed_dim)
        c = tf.convert_to_tensor(c0)

        attention_scores = []
        output = []
        # Pass through LSTM
        for i in range(text.shape[1]):
            # compute attention context
            context, attn_scores = self.attention(a, features, training=training)
            context = self.expand(context) # (bs, 1, group_size)

            attention_scores.append(attn_scores)

            # combine context with word
            sample = tf.concat([context, tf.expand_dims(text[:, i, :], axis=1)], axis=-1) # (bs, 1, embed_features + embed_text)

            _, a, c = self.lstm(sample, initial_state=[a,c], training=training)
            out = self.dropout_lstm(a, training=training)
            output.append(out)

        output = tf.stack(output, axis=1) # (bs, max_len, embed_dim)

        # Convert to vocab
        output = self.dense_inter(output, training=training)
        output = self.dropout_out(output, training=training)
        output = self.dense_out(output, training=training) # (bs, max_len, vocab_size)

        return output, tf.convert_to_tensor(attention_scores)

    def call_fc(self, data, training=False):
        """ Call fully-connected model

        If pre-training on VGG16 FC for betas model, the 4096 need to be mapped to 32 units and passed to the LSTM on every time step
        """
        img_input, text_input, a0, c0  = data
        img_input = self.dropout_input(img_input, training=training)

        # (4096,) -> (512,)
        features = self.dense_in_1(img_input, training=training)
        features = self.bn(features, training=training)
        features = self.dropout(features, training=training)
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(text_input)
        text = self.dropout_text(text, training=training)

        a = tf.convert_to_tensor(a0)
        c = tf.convert_to_tensor(c0)

        #sample = tf.concat([features, text], axis=1) # (bs, max_len + 1, embed_dim)

        # Pass through LSTM
        _, a, c = self.lstm(features, initial_state=[a, c], training=training)
        seq, _, _ = self.lstm(text, initial_state=[a, c], training=training)
        seq = self.dropout_lstm(seq, training=training)

        # Convert to vocab
        seq = self.dense_inter(seq, training=training)
        seq = self.dropout_out(seq, training=training)
        output = self.dense_out(seq, training=training)

        return output, None


    @tf.function()
    def train_step(self, data):
        """ Single backprop train step 
        
        Parameters
        ----------
            data : tuple
                holds the ((features, captions, init_state, init_state), target)

        Returns
        -------
            dict
                loss/accuracy metrics
        """

        target = data[1] # (batch_size, max_length, 5000)

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0
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
            ) # (bs, max-length, vocab_size)

            # Attention Loss
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

        return {"loss": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy, "attention": attn_loss, 'lr': self.optimizer.lr}

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

        l2_loss   = 0
        cross_entropy_loss = 0
        accuracy  = 0
        attn_loss = 0

        # Call model on sample
        prediction, attention_scores = self(
                (
                    data[0]
                ),
                training=False
        )

        # Attention Loss
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
        return self.greedy_predict_fc(*args, **kwargs)
        #return self.greedy_predict_attention(*args, **kwargs)

    def greedy_predict_fc(self, img_input, a0, c0, start_seq, max_len, units, tokenizer):
        # Encoding 
        features = self.dense_in_1(img_input, training=False) 
        features = self.bn(features, training=False)
        features = self.expand(features)

        # Embed the caption vector
        text = self.embedding(start_seq)
        text = self.expand(text)
        print("text:", text.shape)

        # Initial LSTM call with brain data
        _, a, c = self.lstm(features, initial_state=[a0,c0], training=False)

        outputs = []
        for i in range(max_len-1):
            _, a, c = self.lstm(text, initial_state=[a,c], training=False)
            seq = self.expand(a)
            output = self.dense_out(seq) # (bs, 1, 5001)

            word = np.argmax(output, axis=-1) # (bs, 1)
            outputs.append(word)

            # encode the new word
            text = self.embedding(word)
            print("i:", i)

        outputs = np.stack(outputs, axis=1)
        #assert outputs.shape == (features.shape[0], 13, 1), f"{outputs.shape}"
        return outputs, None

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

        attention_scores = []
        outputs = []
        for i in range(max_len):
            context, attention_score = self.attention(a, features, training=False)
            context = self.expand(context)

            attention_scores.append(attention_score)

            sample = tf.concat([context, text], axis=-1)
            A, a, c = self.lstm(sample, initial_state=[a,c], training=False)
            output = self.dense_out(A) # (bs, 1, 5001)

            # Greedy choice
            word = np.argmax(output, axis=-1)
            # Non-greedy choice
            #word = np.random.choice(np.arange(1, 5000), p=output)
            outputs.append(word)

            # encode the new word
            text = self.embedding(word)

        return np.array(outputs), np.array(attention_scores)

    def non_greedy_word_select(output):
        """ Return an index of a vector based on its probability
        """
        output = np.squeeze(output)
        words = []
        for i in range(features.shape[0]):
            words.append(np.random.choice(np.arange(0, 5001), p = output[i]))
            
        words = np.expand_dims(np.array(words), axis=1)
        
        return words

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






