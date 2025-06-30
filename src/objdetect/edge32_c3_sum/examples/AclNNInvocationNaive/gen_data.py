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


def edge32_c3_sum(data, out):
    half_win_start = 7
    half_win_end = 9
    window_size = half_win_start + half_win_end  # 7 + 9 = 16

    data_int = data.astype(np.int32)
    h, w, c = data_int.shape
    
    s_h, s_w, s_c = data_int.strides

    windows = as_strided(
        data_int,
        shape=(h - window_size + 1, w, c, window_size),
        strides=(s_h, s_w, s_c, s_h)
    )

    sums = np.sum(windows, axis=3)

    output_slice = sums[0:h - window_size, :, :]
    
    out[half_win_start:h - half_win_end, :, :] = (output_slice // 16).astype(out.dtype)

    return out

def gen_golden_data_simple():
    dtype = np.uint8
    input_shape = [3440, 4887, 3]
    output_shape = [3440, 4887, 3]

    x = np.random.randint(0, 255, input_shape).astype(dtype)
    golden = np.zeros(output_shape).astype(dtype)

    golden = edge32_c3_sum(x, golden)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

