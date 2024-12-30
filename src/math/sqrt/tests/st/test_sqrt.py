import tensorflow as tf
import numpy as np

def sqrt_test(x):
    """Compute the Swish activation function."""
    tensor = tf.convert_to_tensor(x)
    ori_dtype = tensor.dtype
    tensor = tensor.astype(tf.float32)
    result = tf.math.sqrt(tensor).astype(ori_type)
    return result.numpy()

def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = sqrt_test(x["value"])
    return [res]
