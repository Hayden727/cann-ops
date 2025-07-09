#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
np.random.seed(123)

def gen_golden_data_simple():
    input_x_real = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    input_x_image = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    # 通过 1j * input_x_image 将虚部与实部结合，形成复数矩阵 input_x
    input_x = input_x_real + 1j * input_x_image
    # print(input_x)
    value = 1.2
    # muls算子并没有第二个输入
    # input_y_real = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    # input_y_image = np.random.uniform(-100, 100, [1999, 1999]).astype(np.float32)
    # input_y = input_y_real + 1j * input_y_image
    golden = input_x * value
    # print("----------------------------------------")
    # print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")

    # input_y.tofile("./input/input_y.bin")

    golden.tofile("./output/golden.bin")



if __name__ == "__main__":
    gen_golden_data_simple()
