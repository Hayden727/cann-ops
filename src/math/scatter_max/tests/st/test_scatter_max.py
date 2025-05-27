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
import torch


def scatter_max_test(x_np, indices_np, updates_np, use_locking=False):
    # 转换为TensorFlow张量（保持原始数据类型）
    # x_tensor = tf.convert_to_tensor(x_np)
    # indices_tensor = tf.convert_to_tensor(indices_np, dtype=tf.int32)
    # updates_tensor = tf.convert_to_tensor(updates_np)
    x_tensor = torch.from_numpy(x_np)
    indices_tensor = torch.from_numpy(indices_np).to(torch.int32)
    updates_tensor = torch.from_numpy(updates_np)
    
    # x_var = tf.Variable(x_tensor)
    
    # 调用ScatterMax操作
    # result = tf.raw_ops.ScatterMax(
    #     ref=x_var,
    #     indices=indices_tensor,
    #     updates=updates_tensor,
    #     use_locking=use_locking
    # )
    result = torch.scatter_reduce(x_tensor, dim=0, index=indices_tensor,
                                src=updates_tensor, reduce="max", include_self=True)
    
    # 返回结果的NumPy数组
    return result.numpy()

def calc_expect_func(x, indices, updates, use_locking):
    """calc_expect_func"""
    res = scatter_max_test(
        x["value"],
        indices["value"],
        updates["value"],
        use_locking
    )
    return [res]