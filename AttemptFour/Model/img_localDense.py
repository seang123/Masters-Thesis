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

        self.dropout = dropout
        self.bn = BatchNormalization()


    def call(self, x, training=False):
        """ Forward pass """
        # x = (bs, 64, 2048)
        #out = [layer(tf.gather(x, idx, axis=1), training=training) for (layer, idx) in zip(self.dense_layers, self.input_groups)] # 41 * (bs, embed_dim)
        out = [layer(x[:,i,:], training=training) for i in range(self.n_features)]
        # => (bs, 64, dim)

        out = tf.convert_to_tensor(out)
        out = tf.reshape(out, perm=[1,0,2])
        out = tf.bn(out)

        return out

