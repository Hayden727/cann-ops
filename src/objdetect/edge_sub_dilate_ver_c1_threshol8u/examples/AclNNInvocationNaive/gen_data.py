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

def edge_sub_dilate_ver_c1(Edge2tmp3, Edge2tmp4, width, height):
    Edge2tmp4[:, :2] = 0
    Edge2tmp4[:, -2:] = 0

    for y in range(height):
        row = Edge2tmp3[y, :]
        Edge2tmp4[y, 2:-2] = np.maximum.reduce([row[:-4], row[1:-3], row[2:-2], row[3:-1], row[4:]])
    return Edge2tmp4

def threshol8u(Edge2tmp4):
    return np.where(Edge2tmp4 != 0, 1, 0)

def gen_golden_data_simple():
    dtype = np.uint8
    input_shape = [3440, 4887]
    output_shape = [3440, 4887]

    x = np.random.randint(0, 255, input_shape).astype(dtype)
    golden = np.zeros(output_shape).astype(dtype)

    golden = edge_sub_dilate_ver_c1(x, golden, input_shape[1], input_shape[0])
    golden = threshol8u(golden).astype(dtype)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

