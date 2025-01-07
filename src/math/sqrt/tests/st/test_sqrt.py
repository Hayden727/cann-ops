import tensorflow as tf
import numpy as np

def sqrt_test(x):
    """Compute the Swish activation function."""
    tensor = tf.convert_to_tensor(x)
    sqrt_tensor = tf.math.sqrt(tensor)
    re = sqrt_tensor.numpy()
    return re

def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = sqrt_test(x['value'])
    return [res]
