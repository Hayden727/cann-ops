#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np

def swish_test(x, scale):
    tensor = tf.convert_to_tensor(x)
    swish_result = tensor * tf.sigmoid(tensor * scale)
    return swish_result.numpy()

def calc_expect_func(x, y, scale=1.0):
    res = swish_test(x["value"], scale)
    return [res]