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
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import L2
import numpy as np
import sys

class NIC(tf.keras.Model):
    """ Class holding the neural network model """

    def __init__(self, input_size, units, embedding_dim, vocab_size, max_length, dropout_rate, input_reg, lstm_reg, output_reg):
        super(NIC, self).__init__()

        # Image input
        self.input1 = Input(shape=(input_size,))

        # L2 Regularizers
        self.l2_in = L2(input_reg)
        self.l2_lstm = L2(input_reg)
        self.l2_out = L2(input_reg)
        self.dropout = Dropout(dropout_rate)

        self.dense_in = Dense(embedding_dim, use_bias=True,
                kernel_initializer=RandomUniform(-0.08, 0.08),
                kernel_regularizer=self.l2_in,
                name = 'dense_img'
        )

        self._lambda = Lambda(lambda x : tf.expand_dims(x, axis=1))

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
        """ Model call """

        # Betas Encoding 
        features = self.dense_in(img_input) 
        if training:
            features = self.dropout(features)
        features = self._lambda(features)

        # Embed the caption vector
        text = self.embedding(text_input)
        if training:
            text = self.dropout(text)

        a0 = tf.convert_to_tensor(a0)
        c0 = tf.convert_to_tensor(c0)

        #print("\nfeatures:", type(features), features.dtype, features.shape)
        #print("a0:", type(a0), a0.dtype, a0.shape)
        #print("c0:", type(c0), c0.dtype, c0.shape, "\n")

        # Pass through LSTM
        _, a, c = self.lstm(features, initial_state=[a0,c0])

        A, _, _ = self.lstm(text, initial_state=[a,c])

        # Convert LSTM output back to (5000,) probability vector
        output = self.dense_out(A)

        return output




class CaptionGenerator(tf.keras.Model):

    def __init__(self, model):
        super(CaptionGenerator, self).__init__()
        self.model = model
        print(f"Caption Generator initialised")

    @tf.function()
    def train_step(self, data):
        """ Single backprop train step """

        img_input, text_input, a0, c0 = data[0]
        target = data[1] # (batch_size, max_length, 5000)

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        with tf.GradientTape() as tape:

            # Call model on sample
            prediction = self.model(
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

            if len(self.model.losses) != 0:
                l2_loss += tf.add_n(self.model.losses)

            # Sum losses for backprop
            total_loss = tf.add(cross_entropy_loss, l2_loss)

        trainable_variables = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        
        return {"CE": cross_entropy_loss, 'L2': l2_loss, 'accuracy': accuracy}

    @tf.function()
    def test_step(self, data):
        """ Called during validation """
        
        img_input, text_input, a0, c0 = data[0]
        target = data[1]

        l2_loss = 0
        cross_entropy_loss = 0
        accuracy = 0

        # Call model on sample
        prediction = self.model(img_input, text_input, a0, c0)

        # Get the loss
        for i in range(0, target.shape[1]):
            cross_entropy_loss += self.loss_function(target[:,i], prediction[:,i])
            accuracy += self.accuracy_calculation(target[:,i], prediction[:,i])

        # Normalise across sentence
        cross_entropy_loss /= int(target.shape[1])
        accuracy /= int(target.shape[1])

        return {"CE": cross_entropy_loss, "L2": l2_loss, 'accuracy': accuracy}

    def loss_function(self, real, pred):
        """ Call the compiled loss function """
        loss_ = self.compiled_loss(real, pred)
        return tf.reduce_mean(loss_)

    def accuracy_calculation(self, real, pred):
        """ Compute Accuracy

        real : ndarray - int [0,1]
            (batch-size, vocab-size) 
        pred : ndarray - float
            (batch-size, vocab-size)
        """
        real_arg_max = tf.math.argmax(real, axis = 1) 
        pred_arg_max = tf.math.argmax(pred, axis = 1)

        count = tf.reduce_sum(tf.cast(real_arg_max == pred_arg_max, tf.float32))
        return count / real_arg_max.shape[0]
        

    def greedy_inference(self):
        pass

