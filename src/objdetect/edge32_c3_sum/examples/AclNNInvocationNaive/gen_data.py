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
from numpy.lib.stride_tricks import as_strided


def Edge32C3Sum_CPU_Optimized(Img, Edge2temp1, width, height):
    """
    优化后的函数，使用NumPy向量化操作。
    """
    nR = 16
    half_win_start = 7
    half_win_end = 9
    window_size = half_win_start + half_win_end  # 7 + 9 = 16

    img_int = Img.astype(np.int32)
    h, w, c = img_int.shape
    
    s_h, s_w, s_c = img_int.strides

    windows = as_strided(
        img_int,
        shape=(h - window_size + 1, w, c, window_size),
        strides=(s_h, s_w, s_c, s_h)
    )

    sums = np.sum(windows, axis=3)

    output_slice = sums[0 : h - window_size, :, :]
    
    Edge2temp1[half_win_start : h - half_win_end, :, :] = (output_slice // nR).astype(Edge2temp1.dtype)

def gen_golden_data_simple():
    dtype = np.uint8
    input_shape = [3440, 4887, 3]
    output_shape = [3440, 4887, 3]

    x = np.random.randint(0, 255, input_shape).astype(dtype)
    golden = np.zeros(output_shape).astype(dtype)

    Edge32C3Sum_CPU_Optimized(x, golden, input_shape[1], input_shape[0])

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

