#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
import math

def logM(x, base: float = -1.0, scale: float = 1.0, shift: float = 0.0):
    # 转换为 PyTorch 张量

    exponent = x * scale + shift

    if base != -1.0:
        return np.log(exponent) / math.log(base)
        print(f"base,scale,shift: {base,scale,shift}")
        return
  

    print(f"exponent: {exponent}")
    return np.log(exponent)

def gen_golden_data_simple():
    dtype = np.float16
    output_shape = [100, 100]
    input_x = np.random.uniform(1, 4, output_shape).astype(dtype)
    
    # 计算 golden 数据
    golden = logM(input_x, -1.0, 5.0, 5.0).astype(dtype)


    
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()