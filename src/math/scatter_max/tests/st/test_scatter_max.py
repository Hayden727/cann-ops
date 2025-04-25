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

import numpy as np
import tensorflow as tf


def scatter_max_test(x_np, indices_np, updates_np, use_locking):
    # 转换为TensorFlow张量（保持原始数据类型）
    x_tensor = tf.convert_to_tensor(x_np)
    indices_tensor = tf.convert_to_tensor(indices_np, dtype=tf.int32)
    updates_tensor = tf.convert_to_tensor(updates_np)
    
    # 调整indices维度格式
    indices_nd = tf.expand_dims(indices_tensor, axis=-1)
    
    # 计算最大值并更新张量
    current_values = tf.gather_nd(x_tensor, indices_nd)
    max_values = tf.maximum(current_values, updates_tensor)
    result_tensor = tf.tensor_scatter_nd_update(
        x_tensor,
        indices_nd,
        max_values
    )
    
    return result_tensor.numpy()

def calc_expect_func(x, indices, updates, out, use_locking=False):
    """calc_expect_func"""
    res = scatter_max_test(
        x["value"],
        indices["value"],
        updates["value"],
        use_locking
    )
    return [res, ]