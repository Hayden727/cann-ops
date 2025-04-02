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

import numpy as np
import os
np.random.seed(123)

def gen_golden_data_simple():
    input_x_real = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float16)
    input_x_image = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float16)
    # 通过 1j * input_x_image 将虚部与实部结合，形成复数矩阵 input_x
    input_x = input_x_real + 1j * input_x_image
    print(input_x)
    value = 1.2
    # muls算子并没有第二个输入
    # input_y_real = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    # input_y_image = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    # input_y = input_y_real + 1j * input_y_image
    golden = input_x * value
    # print("----------------------------------------")
    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")

    # input_y.tofile("./input/input_y.bin")

    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    gen_golden_data_simple()
