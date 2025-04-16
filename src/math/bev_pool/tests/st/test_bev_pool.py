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

def bev_pool_test(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths):
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
    bev_feat = np.zeros(bev_feat_shape, dtype=np.float16)

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
            # assert b == b_ and n == n_ and fh == fh_ and fw == fw_

            # 计算 bev_feat 中的位置
            b__, dz, dy, dx, c_ = np.unravel_index(rank_bev, bev_feat_shape)
            # assert b == b__ and c == c_

            # 累加特征
            bev_feat[b__, dz, dy, dx, c_] += depth[b, n, d, fh, fw] * feat[b_, n_, fh_, fw_, c]

    # 调整维度为 (B, C, Dz, Dy, Dx)
    x = np.transpose(bev_feat, (0, 4, 1, 2, 3))
    return x


def calc_expect_func(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, out, bev_feat_shape):
    # B = 1
    # N = 2
    # D = 3
    # fH = 4
    # fW = 5
    # C = 6
    # D_Z = 3
    # D_Y = 4
    # D_X = 5
    # N_points = 10
    # N_pillar = 5
    # res = depth["value"]
    # bev_feat_shape1 = (B, D_Z, D_Y, D_X, C)
    res = bev_pool_test(depth["value"], feat["value"], ranks_depth["value"], ranks_feat["value"], ranks_bev["value"], bev_feat_shape, interval_starts["value"], interval_lengths["value"])
    return [res]