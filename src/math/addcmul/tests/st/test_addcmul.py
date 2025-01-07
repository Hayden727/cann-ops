import tensorflow as tf
import numpy as np

def addcmul_test(input_data, x1, x2, value):
    """Compute the Swish activation function."""
    input_data_tensor = tf.convert_to_tensor(input_data)
    x1_tensor = tf.convert_to_tensor(x1)
    x2_tensor = tf.convert_to_tensor(x2)
    value_tensor = tf.convert_to_tensor(value)

    ori_dtype = input_data_tensor.dtype
    if ori_dtype == tf.int32:
        compute_dtype = tf.int32
    else:
        compute_dtype = tf.float32
    input_data_tensor = tf.cast(input_data_tensor, compute_dtype)
    x1_tensor = tf.cast(x1_tensor, compute_dtype)
    x2_tensor = tf.cast(x2_tensor, compute_dtype)
    value_tensor = tf.cast(value_tensor, compute_dtype)
    res = input_data + x1_tensor * x2_tensor * value_tensor
    result = tf.cast(res, ori_dtype)
    return result.numpy()

def calc_expect_func(input_data, x1, x2, value, y):
    """
    calc_expect_func
    """
    res = addcmul_test(input_data["value"], x1['value'], x2['value'], value['value'])
    return [res]
