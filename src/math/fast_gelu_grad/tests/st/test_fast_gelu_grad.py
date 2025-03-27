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
    original_dtype = x.dtype
    if original_dtype != np.float32:
        x = tf.cast(x, tf.float32).numpy()
        dy = tf.cast(dy, tf.float32).numpy()
    attr = 1.702
    attr_opp = 0 - attr
    attr_half = attr / 2

    abs_x = np.abs(x)
    mul_abs_x = abs_x * attr_opp
    exp_x = np.exp(mul_abs_x)

    add_2 = x * exp_x * attr
    exp_pn_x = np.exp((x - abs_x) * attr)

    div_up = exp_x + add_2 +exp_pn_x
    div_down = (exp_x + 1) ** 2

    result_temp = div_up / div_down
    res = dy * result_temp
    if original_dtype != np.float32:
        res = tf.cast(res, original_dtype).numpy()
    return res


def calc_expect_func(dy, x, z):
    res = fast_gelu_grad_test(dy["value"], x["value"])
    return [res]