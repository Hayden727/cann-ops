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
import tensorflow as tf

import os


def gen_golden_data_simple():
    input_x = np.random.uniform(1, 10, [7, 2045]).astype(np.float16)
    clip_value_min = np.random.uniform(1, 3, [1]).astype(np.float16)
    clip_value_max = np.random.uniform(4, 10, [1]).astype(np.float16)
    y = tf.clip_by_value(input_x, clip_value_min, clip_value_max)

    golden =y.numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    clip_value_min.tofile("./input/input_clip_value_min.bin")
    clip_value_max.tofile("./input/input_clip_value_max.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()


