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

def mse_loss_test(predict, label):
    predict_dtype = predict.dtype
    if predict_dtype != np.float32:
        predict = tf.cast(predict, tf.float32).numpy()
    label_dtype = label.dtype
    if label_dtype != np.float32:
        label = tf.cast(label, tf.float32).numpy()
    res =  np.mean((predict - label) ** 2)
    if predict_dtype != np.float32:
        res = tf.cast(res, predict_dtype).numpy()
    return res


def calc_expect_func(predict, label, y):
    res = mse_loss_test(predict["value"], label["value"])
    return [res]