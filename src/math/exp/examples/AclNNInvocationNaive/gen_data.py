#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================

import os
import numpy as np
import math



def exp(x, scale: float = 1.0, shift: float = 0.0, base: float = -1.0):
    x = x.astype(np.float32)
    # 计算线性组合
    exponent = x * scale + shift
    
    # 底数处理逻辑
    if base != -1.0:
        base = math.log(base)
        print(f"base,scale,shift: {base,scale,shift}")
        exponent *= base  # 转换为自然对数基底
    print(f"exponent: {exponent}")
    # 指数运算
    return np.exp(exponent)


def gen_golden_data_simple():
    dtype = np.float16
    output_shape = [10, 10]
    input_x = np.random.uniform(1, 4, output_shape).astype(dtype)
    golden = exp(input_x, 2.0, 1.0, 2.0).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
