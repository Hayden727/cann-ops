#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np


def swish_test(x, scale):
    """Compute the Swish activation function."""
    tensor = tf.convert_to_tensor(x)
    ori_dtype = tensor.dtype
    compute_dtype = tf.float32
    tensor = tf.cast(tensor, compute_dtype)
    swish_result = tensor * tf.sigmoid(tensor * scale)
    swish_result = tf.cast(swish_result, ori_dtype)
    return swish_result.numpy()


def calc_expect_func(x, y, scale=1.0):
    """
    calc_expect_func
    """
    res = swish_test(x["value"], scale)
    return [res]