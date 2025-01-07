#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import os
import torch
import numpy as np

def gen_golden_data_simple():
    dtype = np.float16
    input_shape = [8, 2048]
    output_shape = [8, 2048]

    x = np.random.uniform(-1, 1, input_shape).astype(dtype)
    y = np.random.uniform(-1, 1, input_shape).astype(dtype)

    golden = np.add(x, y)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.astype(dtype).tofile("./input/input_x.bin")
    y.astype(dtype).tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()

