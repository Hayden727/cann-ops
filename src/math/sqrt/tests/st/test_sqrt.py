import tensorflow as tf
import numpy as np

def calc_expect_func(x, y):
    """
    calc_expect_func
    """
    res = np.sqrt(x['value'])
    return [res]
