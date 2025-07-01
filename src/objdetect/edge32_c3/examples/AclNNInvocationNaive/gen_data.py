#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDINg BUT NOT LIMITED TO NON-INFRINgEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import torch
import numpy as np


def edge32_c3(data, out, width, height):
    out[:15, :] = 0
    out[-16:, :] = 0
    out[:, 0] = 0
    out[:, -1] = 0

    core_x = np.arange(width)[1:-1]
    core_y = np.arange(height)[15:-16]
    core_y, core_x = np.meshgrid(core_y, core_x, indexing='ij')

    for i in range(3):
        x0 = data[core_y, core_x - 1, i]
        x2 = data[core_y, core_x + 1, i]
        g = np.abs(x2.astype(np.int32) - x0.astype(np.int32))
        g = np.minimum(g, 255)
        if i == 0:
            max_g = g
        else:
            max_g = np.maximum(max_g, g)

    out[core_y, core_x] = max_g
    return out


def gen_golden_data_simple():
    dtype = np.uint8
    input_shape = [3440, 4887, 3]
    output_shape = [3440, 4887]

    x = np.random.randint(0, 255, input_shape).astype(dtype)
    golden = np.zeros(output_shape).astype(dtype)

    golden = edge32_c3(x, golden, input_shape[1], input_shape[0])

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()

