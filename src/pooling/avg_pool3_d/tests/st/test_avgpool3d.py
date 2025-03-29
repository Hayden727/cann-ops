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

import torch
import numpy as np
import tensorflow as tf


dtype_map = {
    'float': np.float32,
    'float16': np.float16,
    'bfloat16': tf.bfloat16.as_numpy_dtype
}


def calc_expect_func(x, 
                    y, 
                    ksize, 
                    strides, 
                    pads, 
                    ceil_mode=False, 
                    count_include_pad=True, 
                    divisor_override=0, 
                    data_format="NCDHW"
                ):
    x_np = x['value'].astype(np.float32)
    x_torch = torch.tensor(x_np)
    pads = pads[0::2]
    model = torch.nn.AvgPool3d(
                                ksize, 
                                strides,
                                pads,
                                ceil_mode,
                                count_include_pad,
                                divisor_override
                                )
    output = model(x_torch)
    res = output.numpy()
    res = res.astype(dtype_map[x['dtype']])
    return [res]
