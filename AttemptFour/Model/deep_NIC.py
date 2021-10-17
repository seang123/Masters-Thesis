import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                            LSTM,
                            BatchNormalization,
                            Dropout,
                            Embedding,
                            Input,
                            Lambda,
                            TimeDistributed,
                            )
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import L2
import numpy as np
import sys

class NIC(tf.keras.Model):
    """ Class holding the neural network model """

    def __init__(self, input_size, units, embedding_dim, vocab_size, max_length, dropout_rate, input_reg, lstm_reg, output_reg):
        """ Initialisation method. Builds the keras.Model

        Parameters
        ----------
            input_size : int 
                the size of the input features. eg: (62756,) for betas visual cortex
            units : int
                nr of hidden units in LSTM
            embedding_dim : int
                nr of hidden in dense encoder
            vocab_size : int
                vocabulary size - nr of hidden units in the dense out layer
            max_length : int
                max caption length
            dropout_rate : float
                the dropout rate
            input_reg : float
                L2 lambda on input layer 
            lstm_reg : float
                L2 lambda on LSTM 
            output_reg : float
                L2 lambda on output layer

        """
        super(NIC, self).__init__()

        # Image input
        self.input1 = Input(shape=(input_size,))

        # L2 Regularizers
        self.l2_in = L2(input_reg)
        self.l2_lstm = L2(lstm_reg)
        self.l2_out = L2(lstm_reg)
        self.dropout = Dropout(dropout_rate)

        # Batch Norm
        #self.batch_norm = BatchNormalization(name='batch_norm')

        self.dense_in = Dense(embedding_dim, use_bias=True,
                kernel_initializer=RandomUniform(-0.08, 0.08),
                kernel_regularizer=self.l2_in,
                name = 'dense_img'
        )

        self.dense_2 = Dense(embedding_dim, use_bias=True,
                kernel_initializer=RandomUniform(-0.08, 0.08),
                kernel_regularizer=self.l2_in,
                name = 'dense_2'
        )

        self.expand = Lambda(lambda x : tf.expand_dims(x, axis=1))

        # Text input
        self.input2 = Input(shape=(max_length,))
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
                bias_regularizer=self.l2_lstm,
                name = 'lstm'
        )

        # Output dense layer
        self.dense_out = TimeDistributed(
                Dense(vocab_size,
                    activation='softmax',
                    kernel_regularizer=self.l2_out,
                    bias_regularizer=self.l2_out,
                    kernel_initializer=RandomUniform(-0.08, 0.08),
                ),
                name = 'time_distributed_softmax'
        )

    def call(self, img_input, text_input, a0, c0, training=False):
        """ Model call 
        
        Parameters
        ----------
            img_input : ndarray
                the features (betas/CNN)
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

        # Betas Encoding 
        features = self.dense_in(img_input) 
        if training:
            features = self.dropout(features)
        features = self.dense_2(features)
        if training:
            features = self.dropout(features)
        features = self.expand(features)
        #features = self.batch_norm(features)

        # Embed the caption vector
        text = self.embedding(text_input)
        if training:
            text = self.dropout(text)

        a0 = tf.convert_to_tensor(a0)
        c0 = tf.convert_to_tensor(c0)

        # Pass through LSTM
        _, a, c = self.lstm(features, initial_state=[a0,c0])

        A, _, _ = self.lstm(text, initial_state=[a,c])

        if training:
            A = self.dropout(A)

        # Convert LSTM output back to (5000,) probability vector
        output = self.dense_out(A)

        return output


    def greedy_predict(self, img_input, a0, c0, start_seq, max_len, units):
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
        #features = self.batch_norm(features)

        text = self.embedding(start_seq)
        text = self.expand(text)

        whole, final, c = self.lstm(features, initial_state=[a0, c0])
        final = tf.squeeze(whole, axis=1)

        outputs = []
        for i in range(max_len):
            whole, final, c = self.lstm(text, initial_state=[final,c])
            final = tf.squeeze(whole, axis=1)

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

        img_input, text_input, a0, c0 = data[0]
        target = data[1] # (batch_size, max_length, 5000)

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction = self(
                    img_input, 
                    text_input, 
                    a0, 
                    c0, 
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
        prediction = self(img_input, text_input, a0, c0, False)

        # Get the loss
        for i in range(0, target.shape[1]):
            cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
            accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

        # Normalise across sentence
        cross_entropy_loss /= int(target.shape[1])
        accuracy /= int(target.shape[1])

        if len(self.losses) != 0:
            l2_loss += tf.add_n(self.losses)

        return {"val_loss": cross_entropy_loss, "val_L2": l2_loss, 'val_accuracy': accuracy}

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
        










