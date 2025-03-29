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


def cross_test(x1, x2):
    x1_dtype = x1.dtype
    if x1_dtype != np.int8:
        x1 = tf.cast(x1, tf.int8).numpy()
    x2_dtype = x1.dtype
    if x2_dtype != np.int8:
        x2 = tf.cast(x2, tf.int8).numpy()    
    res = tf.linalg.cross(x1, x2)
    if x1_dtype != np.int8:
        res = tf.cast(res, x1_dtype).numpy()
    return res.numpy()


def calc_expect_func(x1, x2, y):
    res = cross_test(x1["value"], x2["value"])
    return [res]