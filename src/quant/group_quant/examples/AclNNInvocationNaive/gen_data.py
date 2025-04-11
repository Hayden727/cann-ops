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

import torch

def pack_int4_as_uint8(src: np.ndarray):
    """
    Pack int4 numpy array (each int4 number stored in one byte actually) to be continuous bytes stored in uint8
    """
    pack_size = 2
    shift = np.array([0, 4], dtype=np.uint8)
    array = src
    if src.size % pack_size != 0:
        array = np.pad(src.flatten(), (0, pack_size - src.size % pack_size), mode='constant')
    reshaped = array.reshape([-1, 2])

    # bitwise_and is for arm
    out = np.sum(np.bitwise_and(reshaped.view(np.uint8), 0b00001111) << shift, axis=1, dtype=np.uint8)
    return out

def group_quant_golden(context):
    x, scale, group_index, offset = context.input_arrays[0], context.input_arrays[1], context.input_arrays[2], None
    if len(context.input_arrays) > 3:
        offset = context.input_arrays[3]
    attr_dst_type = context.other_compilation_params.get("dst_type", 2)

    dim_s = x.shape[0]
    dim_h = x.shape[1]
    dim_e = scale.shape[0]
    if attr_dst_type == 29:
        assert dim_h % 2 == 0, "For output y, if datatype is int4, dim of last axis should be even number"

    x_fp32 = x.astype('float32')
    scale_fp32 = scale.astype('float32')
    offset_fp32 = offset.astype('float32') if offset is not None else None
    y_fp32 = np.empty(shape=(0, dim_h), dtype='float32')

    # check for group_index
    min_index = np.min(group_index)
    assert min_index >= 0, "group_index value should be greater than or equal 0"
    max_index = np.max(group_index)
    assert max_index <= dim_s, "group_index value should be less than or equal S"
    if group_index.size > 1:
        diff_index = group_index[1:] - group_index[:-1]
        min_diff = np.min(diff_index)
        assert min_diff >= 0, "group_index value must be monotonically non decreasing"
    assert group_index[-1] == dim_s, "group_index last value must be S"

    for row_scale in range(dim_e):
        x_start_row = 0 if row_scale == 0 else group_index[row_scale - 1]
        x_end_row = group_index[row_scale]
        if x_start_row < x_end_row:
            y_rows = x_fp32[x_start_row: x_end_row] * scale_fp32[row_scale]
            if offset is not None:
                y_rows = y_rows + offset_fp32
            y_fp32 = np.concatenate([y_fp32, y_rows], axis=0)
    y_round = np.round(y_fp32, 0)

    res = None
    if attr_dst_type == 2:
        res = np.clip(y_round, -128, 127).astype('int8')
    elif attr_dst_type == 29:
        y_int4 = np.clip(y_round, -8, 7).astype('int4')
        y_pack = pack_int4_as_uint8(y_int4).reshape(-1, (dim_h + 1) // 2)
        res = y_pack
    else:
        raise Exception("attr dst_type only support 2(int8) or 29(int4)")

    return [res, ]

def gen_golden_data_simple():
    # 输入张量
    input_shape = (1, 1, 3, 3)
    input_tensor = torch.randn(input_shape, dtype=torch.float32) * 255    # value range: (-255, 255), shape: (1, 1, 2, 5) dtype: float32

    # 插值参数
    output_size = [5, 5]    # 输出shape
    mode = 'bicubic'    # 插值模式
    align_corners = False    # 角对齐

    # 调用函数
    output_tensor = interpolate(input_tensor, size=output_size, mode=mode, align_corners=align_corners)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_tensor.numpy().tofile("./input/input_tensor.bin")
    output_tensor.numpy().tofile("./output/golden_out.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
