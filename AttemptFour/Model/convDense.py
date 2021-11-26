import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import L2

class LocallyDense(tf.keras.layers.Layer):
    '''
    '''
    def __init__(self, input_groups, output_groups, embed_dim, dropout, **kwargs):
        '''
        '''
        super(LocallyDense, self).__init__()

        self.input_groups = input_groups

        self.dense_layers = [
                tf.keras.layers.Dense(dim, **kwargs) for dim in output_groups]

        self.conv = tf.keras.layers.conv1d(512, 32, strides=1, padding='valid', activation=LeakyReLU(0.2))

        self.dropout = dropout

    def call(self, x, training=False):
        """ Forward pass """
        out = [layer(tf.gather(x, idx, axis=1), training=training) for (layer, idx) in zip(self.dense_layers, self.input_groups)] # 41 * (bs, embed_dim)

        ## concate method 
        out = tf.concat(out, axis=1)
        out = self.dropout(out, training=training)

        out = self.conv(out)

        print("out", out.shape)

        raise

        return out

