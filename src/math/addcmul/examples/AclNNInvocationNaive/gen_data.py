#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import numpy
import os


def gen_golden_data_simple():
    dtype = numpy.float16
    output_shape = [100, 100]
    input_data = numpy.random.uniform(1, 4, output_shape).astype(dtype)
    input_x1 = numpy.random.uniform(1, 4, output_shape).astype(dtype)
    input_x2 = numpy.random.uniform(1, 4, output_shape).astype(dtype)
    input_value = numpy.random.uniform(1, 4, [1]).astype(dtype)
    if dtype == numpy.int32:
        input_value[0] = 2
    else:
        input_value[0] = 1.2
    golden = (input_data + input_x1*input_x2*input_value).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_data.tofile("./input/input_data.bin")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    input_value.tofile("./input/input_value.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
