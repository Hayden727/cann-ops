#!/usr/bin/python3
# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
import tensorflow as tf
import numpy as np

def gelu_grad_test(dy_np, x_np, soc='910b'):
    dy = tf.convert_to_tensor(dy_np)
    x = tf.convert_to_tensor(x_np)

    ori_dtype = dy.dtype
    compute_dtype = tf.float32
    dy = tf.cast(dy, compute_dtype)
    x = tf.cast(x, compute_dtype)
    if soc == '910b':
        x_square = tf.math.square(x)
        px = tf.math.multiply(x_square, -0.0713548162726002527220)
        px = tf.math.add(px, -1.595769121605730711759)
        px = tf.math.multiply(px, x)
        px = tf.math.exp(px)

        res0 = tf.math.multiply(x_square, 0.2140644488178007)
        res0 = tf.math.add(res0, 1.595769121605730711759)
        res0 = tf.math.multiply(res0, x)

        t = tf.math.add(px, 1.0)
        const_one = tf.ones(x.shape, dtype=compute_dtype)
        t = tf.math.divide(const_one, t)

        resp = tf.math.multiply(px, t)
        resp = tf.math.multiply(resp, res0)
        resp = tf.math.multiply(resp, t)

        mask_select = tf.equal(resp, resp)
        resp = tf.where(mask_select, resp, 0.0)
        resp = tf.math.add(resp, t)

        golden = tf.math.multiply(dy, resp)
        golden = tf.cast(golden, ori_dtype).numpy()
    else:
        x_square = tf.math.square(x)
        px = tf.math.multiply(x_square, -0.0713548162726002527220)
        px = tf.math.add(px, -1.5957691216057308)
        px = tf.math.multiply(px, x)
        px = tf.math.exp(px)
        px = tf.math.add(px, 1.0)
        const_one = tf.ones(x.shape, dtype=compute_dtype)
        px = tf.math.divide(const_one, px)

        res=tf.math.add(px, -1.0)
        res= tf.math.multiply(res, x)

        g2=tf.math.multiply(x_square, -0.21406444881780074632901625683959062)
        g2=tf.math.add(g2, -1.5957691216057307117597842397375274738)

        res = tf.math.multiply(res, g2)
        res= tf.math.add(res, 1.0)
        res = tf.math.multiply(res, px)
        golden = tf.math.multiply(dy, res)
        golden = tf.cast(golden, ori_dtype).numpy()
    return golden

def calc_expect_func(dy, x, y, z):
    """
    calc_expect_func
    """
    res = gelu_grad_test(dy["value"], x['value'], soc='910b')
    return [res]
