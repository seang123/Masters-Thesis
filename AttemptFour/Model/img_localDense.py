import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import L2

class LocallyDense(tf.keras.layers.Layer):
    '''
    A stack of FC layers 
        used for locally connected encoding of input
        ie. Assume an input of (batch_size, features, dimension)
            a new FC layer will be applied to each feature
    '''
    def __init__(self, n_features, embed_dim, dropout, **kwargs):
        '''
        Note: if each dense layer has the same name, then the checkpoint callback will fail. Best to just not give a name
        '''
        super(LocallyDense, self).__init__()

        self.n_features = n_features

        self.dense_layers = [
                tf.keras.layers.Dense(embed_dim, **kwargs) for dim in range(n_features)]

        self.bn = BatchNormalization()
        self.do = dropout


    def call(self, x, training=False):
        """ Forward pass """
        # x = (bs, 512, 196)
        out = [layer(x[:,i,:], training=training) for (layer,i) in zip(self.dense_layers, range(self.n_features))]
        # len out = 512, out[0] = (bs, group_size)
        out = tf.convert_to_tensor(out)
        out = tf.transpose(out, perm=[1,0,2]) # => (bs, 512, n_features)
        out = self.bn(out, training=training)
        out = self.do(out, training=training)

        return out
