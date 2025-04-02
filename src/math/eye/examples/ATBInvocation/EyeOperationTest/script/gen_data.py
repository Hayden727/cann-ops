import torch
import os
def gen_golden_data_simple():
    dtype = "float32"
    input1 = torch.zeros(3 , 4, 133, 4095, dtype=torch.float)
    golden = torch.eye(133,4095, dtype=torch.float)
    golden = golden.unsqueeze(0).unsqueeze(0)
    golden = golden.repeat(3,4,1,1)
    input1.numpy().tofile('./script/input/input0.bin')
    golden.numpy().tofile("./script/output/golden0.bin")
    
    with open("./script/output/meta", "w") as fp:
        print(dtype, file=fp)

if __name__ == "__main__":
    gen_golden_data_simple()
