#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import numpy as np


def radius_numpy(x, y, r, ptr_x=None, ptr_y=None, max_num_neighbors=32, ignore_same_index=False):
    """
    用numpy实现radius_cuda算子的功能
    :param x: 节点特征矩阵, shape为 [N, F]
    :param y: 节点特征矩阵, shape为 [M, F]
    :param r: 半径
    :param ptr_x: 可选的批次指针
    :param ptr_y: 可选的批次指针
    :param max_num_neighbors: 每个元素返回的最大邻居数
    :param ignore_same_index: 是否忽略相同索引的点
    :return: 邻接索引
    """
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
            for _, t in enumerate(distances):
                if distances[_] <= r:
                    neighbors.append(_)
            neighbors = np.array(neighbors)
            if ignore_same_index:
                neighbors = neighbors[neighbors != i]
            count = 0
            for neighbor in neighbors:
                if count < max_num_neighbors:
                    out_vec.extend([neighbor, i])
                    
                    count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out
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
                for _, t in enumerate(distances):
                    if distances[_] <= r:
                        neighbors.append(_ + x_start)
                neighbors = np.array(neighbors)
                if ignore_same_index:
                    neighbors = neighbors[neighbors != i]
                count = 0
                for neighbor in neighbors:
                    if count < max_num_neighbors:
                        out_vec.extend([neighbor, i])
                        count += 1
        out = np.array(out_vec).reshape(-1, 2).T
        return out


# 调用示例
if __name__ == "__main__":
    cdtype = os.getenv('COMPUTE_TYPE')
    if cdtype == 'float16':
        compute_dtype = np.float16
    elif cdtype == 'float32':
        compute_dtype = np.float32
    else:
        compute_dtype = np.int32

    x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
    y = np.random.uniform(-5, 5, [50, 2]).astype(compute_dtype)
    ptr_x = None
    ptr_y = None
    r = 1.0
    max_num_neighbors = 10
    ignore_same_index = False
    
    x = x.astype(compute_dtype)
    y = y.astype(compute_dtype)
    assign_index = radius_numpy(x, y, r, ptr_x, ptr_y, max_num_neighbors, ignore_same_index).astype(compute_dtype)
    
    for i in assign_index.shape:
        print(i, end=' ')

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.tofile("./input/input_x.bin")
    y.tofile("./input/input_y.bin")
    assign_index.tofile("./output/golden.bin")

