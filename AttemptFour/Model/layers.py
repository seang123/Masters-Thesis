import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras.initializers import HeNormal

class LocallyDense(tf.keras.layers.Layer):
    '''
    A stack of dense layers that map groups of dimensions
    in the input tensor to groups in the output tensor.
    Essentially a collection of parallel Dense layers that act on
    different dimensions of the input tensor. Their outputs are
    concatenated to a single tensor such that locality in the
    separate groups is conserved. Input groups may be overlapping.
    '''
    def __init__(self, input_groups, output_groups, act_f, **kwargs):
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

        self.dense_layers = [tf.keras.layers.Dense(dim, **kwargs) for dim in output_groups]
        self.input_groups = input_groups

        self.act_f = act_f
        self.bn = tf.keras.layers.BatchNormalization()


    def call(self, x, training=False):
        """ Forward pass """
        out = [layer(tf.gather(x, idx, axis=1), training=training) for (layer, idx) in zip(self.dense_layers, self.input_groups)] # 41 * (bs, embed_dim)

        out = tf.convert_to_tensor(out)
        out = tf.transpose(out, perm=[1,0,2]) # (bs, 41, dim)
        out = self.act_f(out)
        out = self.bn(out)

        return out  

