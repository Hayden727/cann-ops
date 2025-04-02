import torch
import os
def gen_golden_data_simple():
    input1 = torch.randn(8, 2048, dtype=torch.float16)
    input2 = torch.randn(8, 2048, dtype=torch.float16)
    
    
    golden = input1 + input2
    input1.numpy().tofile('./script/input/input0.bin')
    input2.numpy().tofile('./script/input/input1.bin')
    golden.numpy().tofile("./script/output/golden0.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
