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
import numpy as np
import sys

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
        N = x.shape[0]
        M = y.shape[0]
        out_vec = []
        for i in range(M):
            distances = np.linalg.norm(x - y[i], axis=1)
            neighbors = []
            for _ in range(len(distances)):
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
        assert ptr_x is not None and ptr_y is not None
        out_vec = []
        for b in range(len(ptr_x) - 1):
            x_start, x_end = ptr_x[b], ptr_x[b + 1]
            y_start, y_end = ptr_y[b], ptr_y[b + 1]
            if x_start == x_end or y_start == y_end:
                continue
            for i in range(y_start, y_end):
                distances = np.linalg.norm(x[x_start:x_end] - y[i], axis=1)
                # neighbors = np.where(distances <= r)[0] + x_start
                neighbors = []
                for _ in range(len(distances)):
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

    case_id = int(os.getenv('CASE_ID'))

    if case_id == 1:
        x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [50, 2]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.0
        max_num_neighbors = 10
        ignore_same_index = False
    elif case_id == 2:
        x = np.random.uniform(-5, 5, [200, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [100, 3]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.5
        max_num_neighbors = 15
        ignore_same_index = True
    elif case_id == 3:
        x = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [30, 5]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 2.3
        max_num_neighbors = 5
        ignore_same_index = False
    elif case_id == 4:
        x = np.random.uniform(-5, 5, [150, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [80, 1]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 2.0
        max_num_neighbors = 20
        ignore_same_index = True
    elif case_id == 5:
        x = np.random.uniform(-5, 5, [1025, 2049]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [120, 2049]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 100.0
        max_num_neighbors = 12
        ignore_same_index = False
    elif case_id == 6:
        x = np.random.uniform(-1, 1, [1026, 2050]).astype(compute_dtype)
        y = np.random.uniform(-1, 1, [150, 2050]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 160.0
        max_num_neighbors = 18
        ignore_same_index = True
    elif case_id == 7:
        x = np.random.uniform(-5, 5, [1027, 2051]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [40, 2051]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 88.0
        max_num_neighbors = 8
        ignore_same_index = False
    elif case_id == 8:
        x = np.random.uniform(-5, 5, [120, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [60, 5]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.3
        max_num_neighbors = 13
        ignore_same_index = True
    elif case_id == 9:
        x = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [110, 1]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 2.2
        max_num_neighbors = 22
        ignore_same_index = False
    elif case_id == 10:
        x = np.random.uniform(-5, 5, [180, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [90, 4]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.6
        max_num_neighbors = 16
        ignore_same_index = True
    elif case_id == 11:
        x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.1
        max_num_neighbors = 11
        ignore_same_index = False
    elif case_id == 12:
        x = np.random.uniform(-5, 5, [200, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [200, 3]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.7
        max_num_neighbors = 17
        ignore_same_index = True
    elif case_id == 13:
        x = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 5.4
        max_num_neighbors = 6
        ignore_same_index = False
    elif case_id == 14:
        x = np.random.uniform(-5, 5, [150, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [150, 1]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 2.1
        max_num_neighbors = 21
        ignore_same_index = True
    elif case_id == 15:
        x = np.random.uniform(-5, 5, [250, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [250, 4]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.4
        max_num_neighbors = 14
        ignore_same_index = False
    elif case_id == 16:
        x = np.random.uniform(-5, 5, [300, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [300, 2]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.9
        max_num_neighbors = 19
        ignore_same_index = True
    elif case_id == 17:
        x = np.random.uniform(-5, 5, [80, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [80, 3]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 0.9
        max_num_neighbors = 9
        ignore_same_index = False
    elif case_id == 18:
        x = np.random.uniform(-5, 5, [120, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [120, 5]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.5
        max_num_neighbors = 15
        ignore_same_index = True
    elif case_id == 19:
        x = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 2.3
        max_num_neighbors = 23
        ignore_same_index = False
    elif case_id == 20:
        x = np.random.uniform(-5, 5, [180, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [180, 4]).astype(compute_dtype)
        ptr_x = None
        ptr_y = None
        r = 1.7
        max_num_neighbors = 17
        ignore_same_index = True
    elif case_id == 21:
        x = np.random.uniform(-5, 5, [150, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [80, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 150], dtype=np.int32)
        ptr_y = np.array([0, 80], dtype=np.int32)
        r = 0.7
        max_num_neighbors = 7
        ignore_same_index = False
    elif case_id == 22:
        x = np.random.uniform(-5, 5, [250, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [120, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 250], dtype=np.int32)
        ptr_y = np.array([0, 120], dtype=np.int32)
        r = 1.4
        max_num_neighbors = 14
        ignore_same_index = True
    elif case_id == 23:
        x = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [30, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 50], dtype=np.int32)
        ptr_y = np.array([0, 30], dtype=np.int32)
        r = 0.4
        max_num_neighbors = 4
        ignore_same_index = False
    elif case_id == 24:
        x = np.random.uniform(-5, 5, [200, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [100, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 200], dtype=np.int32)
        ptr_y = np.array([0, 100], dtype=np.int32)
        r = 2.4
        max_num_neighbors = 24
        ignore_same_index = True
    elif case_id == 25:
        x = np.random.uniform(-5, 5, [300, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [150, 4]).astype(compute_dtype)
        ptr_x = np.array([0, 300], dtype=np.int32)
        ptr_y = np.array([0, 150], dtype=np.int32)
        r = 1.1
        max_num_neighbors = 11
        ignore_same_index = False
    elif case_id == 26:
        x = np.random.uniform(-5, 5, [120, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [60, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 120], dtype=np.int32)
        ptr_y = np.array([0, 60], dtype=np.int32)
        r = 1.6
        max_num_neighbors = 16
        ignore_same_index = True
    elif case_id == 27:
        x = np.random.uniform(-5, 5, [180, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [90, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 180], dtype=np.int32)
        ptr_y = np.array([0, 90], dtype=np.int32)
        r = 0.8
        max_num_neighbors = 8
        ignore_same_index = False
    elif case_id == 28:
        x = np.random.uniform(-5, 5, [100, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 100], dtype=np.int32)
        ptr_y = np.array([0, 50], dtype=np.int32)
        r = 1.3
        max_num_neighbors = 13
        ignore_same_index = True
    elif case_id == 29:
        x = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [110, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 220], dtype=np.int32)
        ptr_y = np.array([0, 110], dtype=np.int32)
        r = 2.2
        max_num_neighbors = 22
        ignore_same_index = False
    elif case_id == 30:
        x = np.random.uniform(-5, 5, [280, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [140, 4]).astype(compute_dtype)
        ptr_x = np.array([0, 280], dtype=np.int32)
        ptr_y = np.array([0, 140], dtype=np.int32)
        r = 1.7
        max_num_neighbors = 17
        ignore_same_index = True
    elif case_id == 31:
        x = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [100, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 100], dtype=np.int32)
        ptr_y = np.array([0, 100], dtype=np.int32)
        r = 1.2
        max_num_neighbors = 12
        ignore_same_index = False
    elif case_id == 32:
        x = np.random.uniform(-5, 5, [200, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [200, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 200], dtype=np.int32)
        ptr_y = np.array([0, 200], dtype=np.int32)
        r = 1.8
        max_num_neighbors = 18
        ignore_same_index = True
    elif case_id == 33:
        x = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 50], dtype=np.int32)
        ptr_y = np.array([0, 50], dtype=np.int32)
        r = 8.0
        max_num_neighbors = 6
        ignore_same_index = False
    elif case_id == 34:
        x = np.random.uniform(-5, 5, [150, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [150, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 150], dtype=np.int32)
        ptr_y = np.array([0, 150], dtype=np.int32)
        r = 2.1
        max_num_neighbors = 21
        ignore_same_index = True
    elif case_id == 35:
        x = np.random.uniform(-5, 5, [250, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [250, 4]).astype(compute_dtype)
        ptr_x = np.array([0, 250], dtype=np.int32)
        ptr_y = np.array([0, 250], dtype=np.int32)
        r = 1.4
        max_num_neighbors = 14
        ignore_same_index = False
    elif case_id == 36:
        x = np.random.uniform(-5, 5, [300, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [300, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 300], dtype=np.int32)
        ptr_y = np.array([0, 300], dtype=np.int32)
        r = 1.9
        max_num_neighbors = 19
        ignore_same_index = True
    elif case_id == 37:
        x = np.random.uniform(-5, 5, [80, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [80, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 80], dtype=np.int32)
        ptr_y = np.array([0, 80], dtype=np.int32)
        r = 4.5
        max_num_neighbors = 9
        ignore_same_index = False
    elif case_id == 38:
        x = np.random.uniform(-5, 5, [120, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [120, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 120], dtype=np.int32)
        ptr_y = np.array([0, 120], dtype=np.int32)
        r = 1.5
        max_num_neighbors = 15
        ignore_same_index = True
    elif case_id == 39:
        x = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 220], dtype=np.int32)
        ptr_y = np.array([0, 220], dtype=np.int32)
        r = 2.3
        max_num_neighbors = 23
        ignore_same_index = False
    elif case_id == 40:
        x = np.random.uniform(-5, 5, [180, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [180, 4]).astype(compute_dtype)
        ptr_x = np.array([0, 180], dtype=np.int32)
        ptr_y = np.array([0, 180], dtype=np.int32)
        r = 1.7
        max_num_neighbors = 17
        ignore_same_index = True
    elif case_id == 41:
        x = np.random.uniform(-5, 5, [150, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [80, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 80, 150], dtype=np.int32)
        ptr_y = np.array([0, 40, 80], dtype=np.int32)
        r = 0.8
        max_num_neighbors = 8
        ignore_same_index = False
    elif case_id == 42:
        x = np.random.uniform(-5, 5, [250, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [120, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 120, 250], dtype=np.int32)
        ptr_y = np.array([0, 60, 120], dtype=np.int32)
        r = 1.3
        max_num_neighbors = 13
        ignore_same_index = True
    elif case_id == 43:
        x = np.random.uniform(-5, 5, [80, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [40, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 40, 80], dtype=np.int32)
        ptr_y = np.array([0, 20, 40], dtype=np.int32)
        r = 9.6
        max_num_neighbors = 6
        ignore_same_index = False
    elif case_id == 44:
        x = np.random.uniform(-5, 5, [200, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [100, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 100, 200], dtype=np.int32)
        ptr_y = np.array([0, 50, 100], dtype=np.int32)
        r = 2.0
        max_num_neighbors = 20
        ignore_same_index = True
    elif case_id == 45:
        x = np.random.uniform(-5, 5, [300, 4]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [150, 4]).astype(compute_dtype)
        ptr_x = np.array([0, 150, 300], dtype=np.int32)
        ptr_y = np.array([0, 75, 150], dtype=np.int32)
        r = 1.4
        max_num_neighbors = 14
        ignore_same_index = False
    elif case_id == 46:
        x = np.random.uniform(-5, 5, [120, 2]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [60, 2]).astype(compute_dtype)
        ptr_x = np.array([0, 60, 120], dtype=np.int32)
        ptr_y = np.array([0, 30, 60], dtype=np.int32)
        r = 1.1
        max_num_neighbors = 11
        ignore_same_index = True
    elif case_id == 47:
        x = np.random.uniform(-5, 5, [180, 3]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [90, 3]).astype(compute_dtype)
        ptr_x = np.array([0, 90, 180], dtype=np.int32)
        ptr_y = np.array([0, 45, 90], dtype=np.int32)
        r = 1.6
        max_num_neighbors = 16
        ignore_same_index = False
    elif case_id == 48:
        x = np.random.uniform(-5, 5, [100, 5]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [50, 5]).astype(compute_dtype)
        ptr_x = np.array([0, 50, 100], dtype=np.int32)
        ptr_y = np.array([0, 25, 50], dtype=np.int32)
        r = 0.7
        max_num_neighbors = 7
        ignore_same_index = True
    elif case_id == 49:
        x = np.random.uniform(-5, 5, [220, 1]).astype(compute_dtype)
        y = np.random.uniform(-5, 5, [110, 1]).astype(compute_dtype)
        ptr_x = np.array([0, 110, 220], dtype=np.int32)
        ptr_y = np.array([0, 55, 110], dtype=np.int32)
        r = 2.2
        max_num_neighbors = 22
        ignore_same_index = False
    elif case_id == 50:
        x = np.random.uniform(-1, 1, [2048, 1024]).astype(compute_dtype)
        y = np.random.uniform(-1, 1, [1024, 1024]).astype(compute_dtype)
        ptr_x = np.array([0, 2048], dtype=np.int32)
        ptr_y = np.array([0, 1024], dtype=np.int32)
        r = 100.8
        max_num_neighbors = 15
        ignore_same_index = True
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

