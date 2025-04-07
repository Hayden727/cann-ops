#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    input_x1 = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    input_x2 = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    golden = np.less_equal(input_x1, input_x2)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
