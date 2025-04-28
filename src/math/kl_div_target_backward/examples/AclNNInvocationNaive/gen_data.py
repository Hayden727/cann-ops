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

def gen_golden_data_simple():
    dtype = np.float16
    input_shape = [4, 400]
    input1_shape = [1, 400]
    output_shape = [4, 400]
    reduction = 0
    logTarget = True

    gradOutput = torch.from_numpy(np.random.uniform(-1, 1, input_shape).astype(dtype))
    selfX = torch.from_numpy(np.random.uniform(-1, 1, input1_shape).astype(dtype))
    target = torch.from_numpy(np.random.uniform(-1, 1, input_shape).astype(dtype))
    if logTarget:
        gradTarget = target + 1
        gradTarget = gradTarget - selfX
        tmp = torch.exp(target)
        gradTarget = gradTarget * tmp
        gradTarget = gradOutput * gradTarget
    else:
        tmp = torch.log(target)
        gradTarget = tmp + 1
        gradTarget = gradTarget - selfX
        gradTarget = gradOutput * gradTarget
        gradTarget = gradTarget.masked_fill(target==0, 0)

    if reduction == 1:
        gradTarget = gradTarget * (1 / target.numel())

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    gradOutput.numpy().astype(dtype).tofile("./input/input_x0.bin")
    selfX.numpy().astype(dtype).tofile("./input/input_x1.bin")
    target.numpy().astype(dtype).tofile("./input/input_x2.bin")
    gradTarget.numpy().astype(dtype).tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

