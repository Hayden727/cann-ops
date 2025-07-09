import os
import sys
import numpy as np

loss = 1e-4 # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 10e-10

def verify_result(real_result, golden):

    result = np.abs(real_result - golden) # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss) # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss) # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss: # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    return True


def verify_complex(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.complex64)
    print(real_result)
    print("----------------------------------------------")
    golden = np.fromfile(golden, dtype=np.complex64)
    print(golden)
    if verify_result(real_result.real.astype(np.float32), golden.real.astype(np.float32)) and verify_result(real_result.imag.astype(np.float32), golden.imag.astype(np.float32)):
        print("test pass")
        return True
    else:
        return False
if __name__ == '__main__':
    verify_complex(sys.argv[1],sys.argv[2])
