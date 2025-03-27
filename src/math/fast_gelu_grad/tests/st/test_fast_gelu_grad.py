#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import tensorflow as tf
import numpy as np


def fast_gelu_grad_test(dy, x):
    dy = tf.convert_to_tensor(dy)
    x = tf.convert_to_tensor(x)
    attr = 1.702
    attr_opp = 0 - attr
    attr_half = tf.math.divide(attr, 2)

    abs_x = tf.math.abs(x)
    mul_abs_x = tf.math.multiply(abs_x, attr_opp)
    exp_x = tf.math.exp(mul_abs_x)

    add_2 = tf.math.multiply(x, tf.math.multiply(exp_x, attr))
    temp1 = tf.math.subtract(x, abs_x)
    exp_pn_x = tf.math.exp(tf.math.multiply(temp1, attr))

    div_up = tf.math.add(exp_x, tf.math.add(add_2, exp_pn_x))
    div_down = tf.math.square(tf.math.add(exp_x, 1))

    result_temp = tf.math.divide(div_up, div_down)
    res = tf.math.multiply(dy, result_temp)
    return res.numpy()


def calc_expect_func(dy, x, z):
    res = fast_gelu_grad_test(dy["value"], x["value"])
    return [res]