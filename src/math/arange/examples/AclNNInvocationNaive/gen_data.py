#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import os
import numpy as np


def gen_golden_data_simple():
    input = np.array([1, -1113, -5], dtype=np.int32)
    golden = np.arange(1, -1113, -5, dtype=np.int32)
    os.system("mkdir -p output")
    os.system("mkdir -p input")
    input.tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
