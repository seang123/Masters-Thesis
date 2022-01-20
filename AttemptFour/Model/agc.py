import tensorflow as tf

def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5

def unitwise_norm(x):
    if len(x.shape) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(x.shape) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 3, 4]! {x}")
    return compute_norm(x, axis, keepdims)

def adaptive_clip_grad(parameters, gradients, clip_factor=0.01, eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        is_index_slice = False
        if type(grads) == tf.python.framework.indexed_slices.IndexedSlices:
            idx = grads.indices
            dense_shape = grads.dense_shape
            grads = grads.values
            is_index_slice = True
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, 1e-6))
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        if is_index_slice:
            new_grad = tf.IndexedSlices(new_grad, idx, dense_shape)
        new_grads.append(new_grad)
    return new_grads
