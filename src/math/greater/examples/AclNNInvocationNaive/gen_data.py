# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import os
import numpy as np

def gen_golden_data_simple():
    dtype = np.float32
    np.random.seed(0)
    input_x1 = np.random.uniform(-100, 100, [8, 17, 7, 19, 16]).astype(dtype)
    input_x2 = np.random.uniform(-100, 100, [8, 17, 7, 19, 16]).astype(dtype)
    golden = np.greater(input_x1, input_x2).astype(np.bool_)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()