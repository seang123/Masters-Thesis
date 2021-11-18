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
import logging

loggerC = logging.getLogger(__name__ + '.guse_model')

class GUSE(tf.keras.Model):
    """
    A model that linearly maps the brain betas to an embedding
    vector. 
    Then KNN is used to find a known GUSE vector as the output 
    """

    def __init__(self, n_hidden=512, dropout=0, l2=0):

        self.dense = Dense(
                n_hidden,
                activation = LeakyReLU(0.2),
                kernel_initializer = RandomUniform(-0.05, 0.05),
                bias_initializer='zeros',
                kernel_regularizer=L2(l2)
        )
