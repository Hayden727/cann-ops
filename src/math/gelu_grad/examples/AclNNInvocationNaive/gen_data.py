#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import os
import numpy as np
import torch
import torch.nn.functional as F


def gen_golden_data_simple():
    dtype = np.float16
    soc = '910b'
    input_shape = [1024, 1024]

    dy = np.random.uniform(-1, 1, input_shape).astype(dtype).astype(np.float32)
    y = np.random.uniform(-1, 1, input_shape).astype(dtype).astype(np.float32)
    input_x = np.random.uniform(-1, 1, input_shape).astype(dtype).astype(np.float32)

    if soc == '310b':
        x_square = np.multiply(input_x, input_x)
        px = np.multiply(x_square, -0.0713548162726002527220)
        px = np.add(px, -1.5957691216057308)
        px = np.multiply(px, input_x)
        px = np.exp(px)
        px = np.add(px, 1.0)
        const_one = np.ones(input_shape, dtype=dtype)
        px = np.divide(const_one, px)

        res = np.add(px, -1.0)
        res = np.multiply(res, input_x)

        g2 = np.multiply(x_square, -0.21406444881780074632901625683959062)
        g2 = np.add(g2, -1.5957691216057307117597842397375274738)

        res = np.multiply(res, g2)
        res = np.add(res, 1.0)
        res = np.multiply(res, px)
        golden = np.multiply(dy, res).astype(dtype)
    elif soc == '910b':
        x_square = np.multiply(input_x, input_x)
        px = np.multiply(x_square, -0.0713548162726002527220)
        px = np.add(px, -1.595769121605730711759)
        px = np.multiply(px, input_x)
        px = np.exp(px)

        res0 = np.multiply(x_square, 0.2140644488178007)
        res0 = np.add(res0, 1.595769121605730711759)
        res0 = np.multiply(res0, input_x)

        t = np.add(px, 1.0)
        const_one = np.ones(input_shape, dtype=dtype)
        t = np.divide(const_one, t)

        resp = np.multiply(px, t)
        resp = np.multiply(resp, res0)
        resp = np.multiply(resp, t)
        mask_select = np.equal(resp, resp)
        resp = np.where(mask_select, resp, 0.0)
        resp = np.add(resp, t)
        golden = np.multiply(dy, resp).astype(dtype)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.astype(dtype).tofile("./input/input_x.bin")
    y.astype(dtype).tofile("./input/input_y.bin")
    dy.astype(dtype).tofile("./input/input_dy.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
