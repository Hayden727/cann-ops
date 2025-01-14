#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np

def equal_test(x, y):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    res = tf.equal(x, y)
    return res.numpy().astype(np.bool_)

def calc_expect_func(x1, x2, y):
    """
    calc_expect_func
    """
    res = equal_test(x1["value"], x2["value"])
    return [res]
