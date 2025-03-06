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


def addcmul_test(input_data, x1, x2, value):
    input_data_tensor = tf.convert_to_tensor(input_data)
    x1_tensor = tf.convert_to_tensor(x1)
    x2_tensor = tf.convert_to_tensor(x2)
    value_tensor = tf.convert_to_tensor(value)
    y = input_data_tensor + x1_tensor * x2_tensor * value_tensor
    return y.numpy()


def calc_expect_func(input_data, x1, x2, value, y):
    res = addcmul_test(input_data["value"], x1['value'], x2['value'], value['value'])
    return [res]
