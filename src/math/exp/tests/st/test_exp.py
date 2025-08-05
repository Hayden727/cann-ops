#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np


def exp_test(x,base,scale,shift):
    x = tf.convert_to_tensor(x)
    y = x * scale + shift
    if base != -1.0:
        if base < 0:
            return None
        y *= math.log(base)
    return y.numpy()


def calc_expect_func(x,base = 2.0,scale = 1.0,shift = 2.0):
    """
    calc_expect_func
    """
    res = exp_test(x["x"],base,scale,shift)
    return [res]
