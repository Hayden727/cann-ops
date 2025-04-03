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


def mse_loss_test(predict, label, reduction):
    tensor_predict = tf.convert_to_tensor(predict)
    tensor_label = tf.convert_to_tensor(label)
    reduction = "mean"
    res = tf.keras.losses.mean_squared_error(
        y_true=tensor_label,
        y_pred=tensor_predict
    )
    return res.numpy()


def calc_expect_func(predict, label, reduction, y):
    res = mse_loss_test(predict["value"], label["value"])
    return [res]