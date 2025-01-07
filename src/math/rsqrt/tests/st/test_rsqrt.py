import tensorflow as tf
import numpy as np

def rsqrt_test(x):
    """Compute the Swish activation function."""
    tensor = tf.convert_to_tensor(x)
    ori_dtype = tensor.dtype
    compute_dtype = tf.float32
    tensor = tf.cast(tensor, compute_dtype)
    result = tf.cast(tf.math.rsqrt(tensor), ori_dtype)
    return result.numpy()

def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = rsqrt_test(x["value"])
    return [res]
