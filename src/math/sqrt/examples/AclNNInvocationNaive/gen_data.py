#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    dtype = np.float16
    output_shape = [1024, 1024]
    input_x = np.random.uniform(1, 4, output_shape).astype(dtype)

    golden = np.sqrt(input_x).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
