#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np


def arange_test(x, y, z):
    arange_tensor = tf.range(x, y, z)
    re = arange_tensor.numpy()
    return re


def calc_expect_func(start, end, step, out):
    """
    calc_expect_func
    """
    res = arange_test(start['value'], end['value'], step['value'])
    return [res]
