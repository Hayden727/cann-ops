#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import os
import sys
import numpy as np


def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.bool_) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=np.bool_) # 从bin文件读取预期运算结果
    for i, (real, gold) in enumerate(zip(real_result, golden)):
        if real != gold:
            print("[ERROR] result error for output index [{}] , expect {} but {}.".format(i, gold, real))
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])
