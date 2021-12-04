import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, Dense
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
        self.dense2 = Dense(512, **kwargs)
        self.dense3 = Dense(embed_dim)

        self.dropout = dropout

    def call(self, x, training=False):
        """ Forward pass """

        out = self.dense2(x)
        #out = self.dropout(out, training=training)

        #out = self.dense2(out)
        out = self.dropout(out, training=training)

        out = self.dense3(out)

        return out

