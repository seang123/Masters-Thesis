import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, Dense, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import L2

class FullyConnected(tf.keras.layers.Layer):
    '''
    '''
    def __init__(self, embed_dim, dropout, **kwargs):
        '''
        '''
        super(FullyConnected, self).__init__()

        #self.dense1 = Dense(1024, kernel_regularizer=kwargs.get('kernel_regularizer',0))
        self.dense2 = Dense(embed_dim, **kwargs)

        self.dropout = dropout
        self.bn = BatchNormalization()

    def call(self, x, training=False):
        """ Forward pass """

        out = self.dense2(x, training=training)
        out = self.bn(out, training=training)
        out = self.dropout(out, training=training)

        return out

