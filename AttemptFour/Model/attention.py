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


class Attention(tf.keras.layers.Layer):
    """ True attention as done in Show, Attend, and Tell 

    Given the hidden state of the lstm at t_i, and a set of features
    return the input for the LSTM on the next timestep t_i+1
    """

    def __init__(self, units, **kwargs):
        super(Attention, self).__init__()

        """ old
        self.attention = tf.keras.layers.Dense(1, **kwargs)
        self.softmax = tf.keras.layers.Softmax(axis = -1)
        """
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, hidden, features, training=False):

        hidden_with_time_axis = tf.expand_dims(hidden, 1) # (bs, 1, hidden_size)
        print("hidden_with_time_axis", hidden_with_time_axis.shape)

        attention_hidden_layer = tf.keras.activations.tanh(
                self.W1(features) +
                self.W2(hidden_with_time_axis)
                )
        print("attention_hidden_layer", attention_hidden_layer.shape)

        score = self.V(attention_hidden_layer)
        print("score", score.shape)

        attention_weights = self.softmax(score)
        print("attention_weights", attention_weights.shape)

        context_vector = attention_weights * features
        print("context_vector", context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        print("context_vector", context_vector.shape)


        raise Exception('stop')
        return
    #hidden_with_time_axis (64, 1, 512)
    #attention_hidden_layer (64, 181, 32)
    #score (64, 181, 1)
    #attention_weights (64, 181, 1)
    #context_vector (64, 181, 512)
    #context_vector (64, 512)




