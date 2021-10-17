import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, ReLU
from tensorflow.keras.initializers import HeNormal, GlorotNormal

class HardAttention(tf.keras.layers.Layer):
    '''
    Hard Attention 
    --------------
        Betas split into 41 regions
        Each region is passed through a FC layer to get a equal-sized embedding
        Softmax is used to choose one embedding to pass along to the RNN
    '''
    def __init__(self, **kwargs):
        super(HardAttention, self).__init__()

        self.attention = tf.keras.layers.Dense(1, **kwargs)
        self.softmax = tf.keras.layers.Softmax(axis = -1)

    def call(self, x, training=False):
        """ Forward pass

        Parameters
        ----------
            x - patch embeddings
                (bs, 41, embed_dim)
        Returns
        -------
            x - attened embedding
                (bs, 1, embed_dim)
        """
        # Compute attention values
        attn_vals = self.attention(x) # (bs, 41, 1)
        attn_vals = self.softmax(attn_vals) # (bs, 41, 1)

        # Find the max attention region
        amax = tf.argmax(attn_vals, axis=1) #(bs, 1)

        # Keep only the max-attention region
        x = tf.gather(x, amax, batch_dims=1) # (bs, 1, embed_dim)
        x = tf.squeeze(x, axis=[1])
    
        return x 

class SoftAttention(tf.keras.layers.Layer):
    """ A soft attention layer
        
    This layer computes a softmax vector across all 41 embeddings 
    and then scales them by their respective probability
    before summing across embeddings
    """

    def __init__(self, **kwargs):
        super(SoftAttention, self).__init__()

        self.attention = tf.keras.layers.Dense(1, **kwargs)
        self.softmax = tf.keras.layers.Softmax(axis = -1)

    def call(self, x, training=False):
        """ Forward pass """
        
        attn_vals = self.attention(x)
        attn_vals = self.softmax(attn_vals)
        context = attn_vals * x
        x = tf.reduce_sum(context, axis = 1)

        return x # (bs, 1, 256)









