# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==========================================================================================================
import os
import sys
import numpy as np

loss = 1e-5 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10
resType = np.float32
# resType = np.float16

def verify_result(cal_result, golden):
    dimSize = 5
    
    cal_result = np.fromfile(cal_result, dtype=resType).reshape(( dimSize, dimSize)) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=resType).reshape(( dimSize, dimSize)) # 从bin文件读取预期运算结果
    result = np.abs(cal_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(cal_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    
    # tol 为 1 说明符合误差范围
    
    # 打印结果
    np.set_printoptions(precision=2, suppress=True)
    print("cal_result:")
    print(cal_result)
    print("golden:")
    print(golden)
        
    # case 误差检查方法
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > cal_result.size * loss and np.sum(result_atol == False) > cal_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1],sys.argv[2])
