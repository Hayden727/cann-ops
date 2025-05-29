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


def radius_test(x, y, ptr_x, ptr_y, r, max_num_neighbors, ignore_same_index):
    ans_dtype = x.dtype
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if ptr_x is None and ptr_y is None:
        # 单示例情况
        n = x.shape[0]
        m = y.shape[0]
        out_vec = []
        for i in range(m):
            distances = np.linalg.norm(x - y[i], axis=1)
            neighbors = []
            for t, _ in enumerate(distances):
                if _ <= r:
                    neighbors.append(t)
            neighbors = np.array(neighbors)
            if ignore_same_index:
                neighbors = neighbors[neighbors != i]
            count = 0
            for neighbor in neighbors:
                if count < max_num_neighbors:
                    out_vec.extend([neighbor, i])
                    
                    count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out.astype(ans_dtype)
    else:
        # 批次情况
        out_vec = []
        for b in range(len(ptr_x) - 1):
            x_start, x_end = ptr_x[b], ptr_x[b + 1]
            y_start, y_end = ptr_y[b], ptr_y[b + 1]
            if x_start == x_end or y_start == y_end:
                continue
            for i in range(y_start, y_end):
                distances = np.linalg.norm(x[x_start:x_end] - y[i], axis=1)
                neighbors = []
                for t, _ in enumerate(distances):
                    if _ <= r:
                        neighbors.append(t + x_start)
                neighbors = np.array(neighbors)
                if ignore_same_index:
                    neighbors = neighbors[neighbors != i]
                count = 0
                for neighbor in neighbors:
                    if count < max_num_neighbors:
                        out_vec.extend([neighbor, i])
                        count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out.astype(ans_dtype)


def calc_expect_func(x, y, ptr_x = None, ptr_y = None, r = 1.0, max_num_neighbors = 32, ignore_same_index = False, out = None):
    """
    calc_expect_func
    """
    if ptr_x is None:
        res = radius_test(x['value'], y['value'], None, None, r, max_num_neighbors, ignore_same_index)
    else:
        res = radius_test(x['value'], y['value'], ptr_x['value'], ptr_y['value'], r, max_num_neighbors, ignore_same_index)
    return [res]
