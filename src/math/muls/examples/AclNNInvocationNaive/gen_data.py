import os
import numpy as np


def gen_golden_data_simple():
    # float16
    dtype = np.float16
    output_shape = [7,1]
    input_x = np.array(
        [[-0.6401, -4.7422,  0.4966, -0.6470, -0.7964, -1.6963, -2.9531]],
        dtype=dtype
    )
    input_value = np.array(1.2, dtype=dtype)  # 标量也转为 NumPy 类型
    golden = input_x * input_value
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
