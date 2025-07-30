#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np


def reciprocal_test(x):
    tensor = tf.convert_to_tensor(x)
    result = tf.math.reciprocal(tensor)
    return result.numpy()
def calc_expect_func(x,y):
    """
    calc_expect_func
    """
    res = reciprocal_test(x["value"])
    return [res]
