import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU, ReLU
from tensorflow.keras.initializers import HeNormal, GlorotNormal

class Attention(tf.keras.layers.Layer):
    """ True attention as done in Show, Attend, and Tell 

    Given the hidden state of the lstm at t_i, and a set of features
    return the input for the LSTM on the next timestep t_i+1
    """

    def __init__(self, units, dropout, **kwargs):
        super(Attention, self).__init__()

        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.dropout = dropout

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.W1 = tf.keras.layers.Dense(units, **kwargs)
        self.W2 = tf.keras.layers.Dense(units, **kwargs)
        self.V  = tf.keras.layers.Dense(1)

    def call(self, hidden, features, training=False):
        """ Forward pass """

        hidden_with_time_axis = tf.expand_dims(hidden, 1) # (bs, 1, units)

        attention_hidden_layer = tf.keras.activations.tanh(
                self.W1(features) +
                self.W2(hidden_with_time_axis)
        ) # (bs, regions, attn_units)

        attention_hidden_layer = self.bn1(attention_hidden_layer)
        #attention_hidden_layer = self.dropout(attention_hidden_layer, training=training)

        score = self.V(attention_hidden_layer) # (bs, regions, 1)

        attention_weights = self.softmax(score) # (bs, regions, 1)

        context_vector = tf.reduce_sum(attention_weights * features, axis=1) # (bs, embed_dim)

        return context_vector, attention_weights



