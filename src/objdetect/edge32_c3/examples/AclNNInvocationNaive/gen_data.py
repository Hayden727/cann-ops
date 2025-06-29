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

def Edge32C3_CPU_fast(Img, Edge2temp1, width, height):
    """
    对输入图像进行边缘计算。
    :param Img: 输入图像，三维数组 (height, width, 3)
    :param Edge2temp1: 输出图像，二维数组 (height, width)
    :param width: 图像宽度
    :param height: 图像高度
    """
    Edge2temp1[:15, :] = 0
    Edge2temp1[-16:, :] = 0
    Edge2temp1[:, 0] = 0
    Edge2temp1[:, -1] = 0

    core_mask = (Edge2temp1 == 0) == False  # 获取核心区域的掩码
    core_x = np.arange(width)[1:-1]  # 排除边界
    core_y = np.arange(height)[15:-16]  # 排除边界
    core_y, core_x = np.meshgrid(core_y, core_x, indexing='ij')

    for i in range(3):
        x0 = Img[core_y, core_x - 1, i]
        x2 = Img[core_y, core_x + 1, i]
        G = np.abs(x2.astype(np.int32) - x0.astype(np.int32))
        G = np.minimum(G, 255)  # 限制最大值为255
        if i == 0:
            max_G = G
        else:
            max_G = np.maximum(max_G, G)

    Edge2temp1[core_y, core_x] = max_G

def gen_golden_data_simple():
    dtype = np.uint8
    input_shape = [3440, 4887, 3]
    output_shape = [3440, 4887]

    x = np.random.randint(0, 255, input_shape).astype(dtype)
    golden = np.zeros(output_shape).astype(dtype)

    Edge32C3_CPU_fast(x, golden, input_shape[1], input_shape[0])

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

