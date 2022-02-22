import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras.initializers import HeNormal
import tensorflow_addons as tfa

class LocallyDense(tf.keras.layers.Layer):
    '''
    A stack of dense layers that map groups of dimensions
    in the input tensor to groups in the output tensor.
    Essentially a collection of parallel Dense layers that act on
    different dimensions of the input tensor. Their outputs are
    concatenated to a single tensor such that locality in the
    separate groups is conserved. Input groups may be overlapping.
    '''
    def __init__(self, groups, dropout, batch_norm, **kwargs):
        '''
        input_groups    -   list of n tensors. Each tensor contains indices
                            of the input dimensions belonging to goup n
        output_groups   -   list of n integers. The nth integer gives the number of
                            dimensions that group n is projected to. 
                            Eg. if there are 41 masked regions and you want each region to have a constant output from the dense layer call
                            then the output_groups == [256,256,....256]. Where 256 is the dimensional output size of the dense layer

        sets up the layer, initializes n dense layers with output dimensionality given
        in output_groups
        '''
        super(LocallyDense, self).__init__()

        in_groups, out_groups = groups
        assert len(in_groups) == len(out_groups), "Input groups don't match ouput groups"

        # Create a layer for each output group
        self.dense_layers = [tf.keras.layers.Dense(dim, **kwargs) for dim in out_groups]
        assert len(self.dense_layers) == 360, "Incorrect nr. of encoder layers"
        self.dense_layers2 = [tf.keras.layers.Dense(dim, **kwargs) for dim in out_groups]

        # Combine the input groups
        self.input_groups = in_groups

        self.dropout = dropout
        self.bn = batch_norm # axis = 1 or -1 seems to give equivalent results

    def call(self, x, training=False):
        """ Forward pass """
        out = [layer(tf.gather(x, idx, axis=1), training=training) for (layer, idx) in zip(self.dense_layers, self.input_groups)] 
        # out => regions * (bs, embed_dim)

        out = tf.convert_to_tensor(out)
        out = tf.transpose(out, perm=[1,0,2]) # (bs, n_regions, dim)
        out = self.bn(out, training=training)
        out = self.dropout(out, training=training)

        return out  

