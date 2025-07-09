import os
import numpy as np


def gen_golden_data_simple():
    # float32
    dtype = np.float32
    output_shape = [15,55290794]
    input_x = np.random.uniform(-5.0, 5.0, size=output_shape).astype(dtype)
    print(input_x)
    input_value = np.array(1.2, dtype=dtype)  # 标量也转为 NumPy 类型
    golden = input_x * input_value
    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
