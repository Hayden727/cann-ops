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
import torch
import numpy as np

def bev_pool(depth, feat, ranks_depth, ranks_feat, ranks_bev,
             bev_feat_shape, interval_starts, interval_lengths):
    """
    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_depth: (N_points, ),
        ranks_feat:  (N_points, ),
        ranks_bev:   (N_points, ),
        bev_feat_shape: (B, D_Z, D_Y, D_X, C)
        interval_starts: (N_pillar, )
        interval_lengths: (N_pillar, )
    Returns:
        x: bev feature in shape (B, C, Dz, Dy, Dx)
    """
    B, D_Z, D_Y, D_X, C = bev_feat_shape
    bev_feat = np.zeros(bev_feat_shape, dtype=np.float32)

    N_pillar = len(interval_starts)
    for i in range(N_pillar):
        start = interval_starts[i]
        length = interval_lengths[i]
        end = start + length
        for j in range(start, end):
            rank_depth = ranks_depth[j]
            rank_feat = ranks_feat[j]
            rank_bev = ranks_bev[j]

            # 从 depth 和 feat 中提取对应元素
            b, n, d, fh, fw = np.unravel_index(rank_depth, depth.shape)
            b_, n_, fh_, fw_, c = np.unravel_index(rank_feat, feat.shape)
            assert b == b_ and n == n_ and fh == fh_ and fw == fw_

            # 计算 bev_feat 中的位置
            b__, dz, dy, dx, c_ = np.unravel_index(rank_bev, bev_feat_shape)
            assert b == b__ and c == c_

            # 累加特征
            bev_feat[b, dz, dy, dx, c] += depth[b, n, d, fh, fw] * feat[b, n, fh, fw, c]

    # 调整维度为 (B, C, Dz, Dy, Dx)
    x = np.transpose(bev_feat, (0, 4, 1, 2, 3))
    return x


# 调用示例
if __name__ == "__main__":
    # 定义输入参数的形状
    case_num = 50
    if case_num == 1:
        B, N, D, fH, fW = 2,4,6,8,10
        D_Z, D_Y, D_X, C = 4,6,8,12
        N_points, N_pillar = 20,10
    if case_num == 2:
        B, N, D, fH, fW = 4,6,8,10,12
        D_Z, D_Y, D_X, C = 6,8,10,14
        N_points, N_pillar = 30,12
    if case_num == 3:
        B, N, D, fH, fW = 2,2,4,6,8
        D_Z, D_Y, D_X, C = 2,4,6,10
        N_points, N_pillar = 16,8
    if case_num == 4:
        B, N, D, fH, fW = 6,8,10,12,14
        D_Z, D_Y, D_X, C = 8,10,12,16
        N_points, N_pillar = 40,14
    if case_num == 5:
        B, N, D, fH, fW = 4,4,6,8,10
        D_Z, D_Y, D_X, C = 4,6,8,12
        N_points, N_pillar = 24,10
    if case_num == 6:
        B, N, D, fH, fW = 8,10,12,14,16
        D_Z, D_Y, D_X, C = 10,12,14,18
        N_points, N_pillar = 50,16
    if case_num == 7:
        B, N, D, fH, fW = 2,6,8,10,12
        D_Z, D_Y, D_X, C = 6,8,10,14
        N_points, N_pillar = 36,12
    if case_num == 8:
        B, N, D, fH, fW = 6,6,8,10,12
        D_Z, D_Y, D_X, C = 6,8,10,14
        N_points, N_pillar = 44,14
    if case_num == 9:
        B, N, D, fH, fW = 4,8,10,12,14
        D_Z, D_Y, D_X, C = 8,10,12,16
        N_points, N_pillar = 48,16
    if case_num == 10:
        B, N, D, fH, fW = 10,12,14,16,18
        D_Z, D_Y, D_X, C = 12,14,16,20
        N_points, N_pillar = 60,18
    if case_num == 11:
        B, N, D, fH, fW = 2,4,4,6,8
        D_Z, D_Y, D_X, C = 4,4,6,10
        N_points, N_pillar = 20,10
    if case_num == 12:
        B, N, D, fH, fW = 4,6,6,8,10
        D_Z, D_Y, D_X, C = 6,6,8,12
        N_points, N_pillar = 30,12
    if case_num == 13:
        B, N, D, fH, fW = 6,8,8,10,12
        D_Z, D_Y, D_X, C = 8,8,10,14
        N_points, N_pillar = 40,14
    if case_num == 14:
        B, N, D, fH, fW = 8,10,10,12,14
        D_Z, D_Y, D_X, C = 10,10,12,16
        N_points, N_pillar = 50,16
    if case_num == 15:
        B, N, D, fH, fW = 10,12,12,14,16
        D_Z, D_Y, D_X, C = 12,12,14,18
        N_points, N_pillar = 60,18
    if case_num == 16:
        B, N, D, fH, fW = 2,2,2,4,6
        D_Z, D_Y, D_X, C = 2,2,4,8
        N_points, N_pillar = 12,6
    if case_num == 17:
        B, N, D, fH, fW = 4,4,4,6,8
        D_Z, D_Y, D_X, C = 4,4,6,10
        N_points, N_pillar = 24,10
    if case_num == 18:
        B, N, D, fH, fW = 6,6,6,8,10
        D_Z, D_Y, D_X, C = 6,6,8,12
        N_points, N_pillar = 36,12
    if case_num == 19:
        B, N, D, fH, fW = 8,8,8,10,12
        D_Z, D_Y, D_X, C = 8,8,10,14
        N_points, N_pillar = 48,14
    if case_num == 20:
        B, N, D, fH, fW = 10,10,10,12,14
        D_Z, D_Y, D_X, C = 10,10,12,16
        N_points, N_pillar = 60,16
    if case_num == 21:
        B, N, D, fH, fW = 2,6,10,14,18
        D_Z, D_Y, D_X, C = 6,10,14,22
        N_points, N_pillar = 42,14
    if case_num == 22:
        B, N, D, fH, fW = 4,8,12,16,20
        D_Z, D_Y, D_X, C = 8,12,16,24
        N_points, N_pillar = 48,16
    if case_num == 23:
        B, N, D, fH, fW = 6,10,14,18,22
        D_Z, D_Y, D_X, C = 10,14,18,26
        N_points, N_pillar = 54,18
    if case_num == 24:
        B, N, D, fH, fW = 8,12,16,20,24
        D_Z, D_Y, D_X, C = 12,16,20,28
        N_points, N_pillar = 60,20
    if case_num == 25:
        B, N, D, fH, fW = 10,14,18,22,26
        D_Z, D_Y, D_X, C = 14,18,22,30
        N_points, N_pillar = 66,22
    if case_num == 26:
        B, N, D, fH, fW = 2,4,8,12,16
        D_Z, D_Y, D_X, C = 4,8,12,20
        N_points, N_pillar = 32,12
    if case_num == 27:
        B, N, D, fH, fW = 4,6,10,14,18
        D_Z, D_Y, D_X, C = 6,10,14,22
        N_points, N_pillar = 42,14
    if case_num == 28:
        B, N, D, fH, fW = 6,8,12,16,20
        D_Z, D_Y, D_X, C = 8,12,16,24
        N_points, N_pillar = 48,16
    if case_num == 29:
        B, N, D, fH, fW = 8,10,14,18,22
        D_Z, D_Y, D_X, C = 10,14,18,26
        N_points, N_pillar = 54,18
    if case_num == 30:
        B, N, D, fH, fW = 10,12,16,20,24
        D_Z, D_Y, D_X, C = 12,16,20,28
        N_points, N_pillar = 60,20
    if case_num == 31:
        B, N, D, fH, fW = 2,2,6,10,14
        D_Z, D_Y, D_X, C = 2,6,10,18
        N_points, N_pillar = 30,10
    if case_num == 32:
        B, N, D, fH, fW = 4,4,8,12,16
        D_Z, D_Y, D_X, C = 4,8,12,20
        N_points, N_pillar = 32,12
    if case_num == 33:
        B, N, D, fH, fW = 6,6,10,14,18
        D_Z, D_Y, D_X, C = 6,10,14,22
        N_points, N_pillar = 42,14
    if case_num == 34:
        B, N, D, fH, fW = 8,8,12,16,20
        D_Z, D_Y, D_X, C = 8,12,16,24
        N_points, N_pillar = 48,16
    if case_num == 35:
        B, N, D, fH, fW = 10,10,14,18,22
        D_Z, D_Y, D_X, C = 10,14,18,26
        N_points, N_pillar = 54,18
    if case_num == 36:
        B, N, D, fH, fW = 2,4,10,16,22
        D_Z, D_Y, D_X, C = 4,10,16,28
        N_points, N_pillar = 32,12
    if case_num == 37:
        B, N, D, fH, fW = 4,6,12,18,24
        D_Z, D_Y, D_X, C = 6,12,18,30
        N_points, N_pillar = 42,14
    if case_num == 38:
        B, N, D, fH, fW = 6,8,14,20,26
        D_Z, D_Y, D_X, C = 8,14,20,32
        N_points, N_pillar = 48,16
    if case_num == 39:
        B, N, D, fH, fW = 8,10,16,22,28
        D_Z, D_Y, D_X, C = 10,16,22,34
        N_points, N_pillar = 54,18
    if case_num == 40:
        B, N, D, fH, fW = 10,12,18,24,30
        D_Z, D_Y, D_X, C = 12,18,24,36
        N_points, N_pillar = 60,20
    if case_num == 41:
        B, N, D, fH, fW = 2,6,12,18,24
        D_Z, D_Y, D_X, C = 6,12,18,30
        N_points, N_pillar = 42,14
    if case_num == 42:
        B, N, D, fH, fW = 4,8,14,20,26
        D_Z, D_Y, D_X, C = 8,14,20,32
        N_points, N_pillar = 48,16
    if case_num == 43:
        B, N, D, fH, fW = 6,10,16,22,28
        D_Z, D_Y, D_X, C = 10,16,22,34
        N_points, N_pillar = 54,18
    if case_num == 44:
        B, N, D, fH, fW = 8,12,18,24,30
        D_Z, D_Y, D_X, C = 12,18,24,36
        N_points, N_pillar = 60,20
    if case_num == 45:
        B, N, D, fH, fW = 10,14,20,26,32
        D_Z, D_Y, D_X, C = 14,20,26,38
        N_points, N_pillar = 66,22
    if case_num == 46:
        B, N, D, fH, fW = 2,4,12,20,28
        D_Z, D_Y, D_X, C = 4,12,20,36
        N_points, N_pillar = 40,14
    if case_num == 47:
        B, N, D, fH, fW = 4,6,14,22,30
        D_Z, D_Y, D_X, C = 6,14,22,38
        N_points, N_pillar = 42,14
    if case_num == 48:
        B, N, D, fH, fW = 6,8,16,24,32
        D_Z, D_Y, D_X, C = 8,16,24,40
        N_points, N_pillar = 48,16
    if case_num == 49:
        B, N, D, fH, fW = 8,10,18,26,34
        D_Z, D_Y, D_X, C = 10,18,26,42
        N_points, N_pillar = 54,18
    if case_num == 50:
        B, N, D, fH, fW = 10,12,20,28,36
        D_Z, D_Y, D_X, C = 12,20,28,44
        N_points, N_pillar = 60,20

    # 生成随机输入数据
    depth = np.random.randn(B, N, D, fH, fW).astype(np.float32)
    feat = np.random.randn(B, N, fH, fW, C).astype(np.float32)

    # 保证 ranks_depth 和 ranks_feat 共享相同的 b, n, fh, fw
    b_indices = np.random.randint(0, B, N_points).astype(np.int32)
    n_indices = np.random.randint(0, N, N_points).astype(np.int32)
    fh_indices = np.random.randint(0, fH, N_points).astype(np.int32)
    fw_indices = np.random.randint(0, fW, N_points).astype(np.int32)
    d_indices = np.random.randint(0, D, N_points).astype(np.int32)
    c_indices = np.random.randint(0, C, N_points).astype(np.int32)

    ranks_depth = np.ravel_multi_index((b_indices, n_indices, d_indices, fh_indices, fw_indices), depth.shape).astype(np.int32)
    ranks_feat = np.ravel_multi_index((b_indices, n_indices, fh_indices, fw_indices, c_indices), feat.shape).astype(np.int32)

    # 保证 ranks_bev 中的 b 和 c 与 ranks_depth 和 ranks_feat 一致
    dz_indices = np.random.randint(0, D_Z, N_points).astype(np.int32)
    dy_indices = np.random.randint(0, D_Y, N_points).astype(np.int32)
    dx_indices = np.random.randint(0, D_X, N_points).astype(np.int32)
    ranks_bev = np.ravel_multi_index((b_indices, dz_indices, dy_indices, dx_indices, c_indices), (B, D_Z, D_Y, D_X, C)).astype(np.int32)

    bev_feat_shape = (B, D_Z, D_Y, D_X, C)
    interval_starts = np.random.randint(0, N_points // 2, N_pillar).astype(np.int32)
    interval_lengths = np.random.randint(1, N_points - interval_starts.max(), N_pillar).astype(np.int32).astype(np.int32)

    os.system("mkdir -p input")
    depth.tofile("./input/input_depth.bin")
    feat.tofile("./input/input_feat.bin")
    ranks_depth.tofile("./input/input_ranks_depth.bin")
    ranks_feat.tofile("./input/input_ranks_feat.bin")
    ranks_bev.tofile("./input/input_ranks_bev.bin")
    interval_starts.tofile("./input/input_interval_starts.bin")
    interval_lengths.tofile("./input/input_interval_lengths.bin")

    # 调用 bev_pool 算子
    result = bev_pool(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                      bev_feat_shape, interval_starts, interval_lengths)
    
    os.system("mkdir -p output")
    result.tofile("./output/result.bin")

    print("BEV 特征形状:", result.shape)
    