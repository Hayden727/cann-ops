import torch
import torch_npu
from typing import Tuple
from ml_dtypes import bfloat16
import numpy as np

# =============================================================================
# 模块1: 简化的NPU单算子基线
# (精确匹配 AddRMSNormQuant 类的计算逻辑)
# =============================================================================
def npu_simplified_baseline(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    scales1: torch.Tensor,
    bias: torch.Tensor,
    zero_points1: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    使用NPU单算子串行执行，精确模拟 AddRMSNormQuant 的 forward 逻辑。
    """
    # 步骤 1: Add + RMSNorm
    norm_result, _, _ = torch_npu.npu_add_rms_norm(x1, x2, gamma, epsilon=epsilon)
    # print("================norm_result===============\n")
    print(norm_result)

    # 步骤 2: Add Bias
    # pre_quant_result = norm_result + bias
    pre_quant_result = norm_result
    # print("================pre_quant_result===============\n")
    # print(pre_quant_result)
    
    # 步骤 3: Quantize (使用官方支持的NPU量化算子)
    # NPU kernel要求 x, scale, offset 的浮点类型必须一致
    target_dtype = pre_quant_result.dtype
    
    # 准备scale和zero_point的类型
    scales1_casted = scales1.to(target_dtype)
    zp1_casted = zero_points1.to(target_dtype)
    
    # 使用npu_quantize并正确设置参数以进行per-channel量化:
    # 1. 直接传递 scale (而不是它的倒数) 来匹配 C++ kernel 的 Div 操作.
    # 2. 设置 axis=-1, 明确指出 scale 和 zero_point 是沿着输入的最后一个维度应用的.
    y_baseline = torch_npu.npu_quantize(
        pre_quant_result,
        scales1_casted,
        zp1_casted,
        torch.qint8,
        -1,  # <-- 关键: 指定 per-channel 的轴
        False
    )
    
    return y_baseline

# =============================================================================
# 模块2: 简化的结果对比函数
# =============================================================================
def compare_simplified_results(y_fused: torch.Tensor, y_baseline: torch.Tensor) -> bool:
    """
    对比单个量化输出结果。由于都是在NPU上计算，理论上结果应完全相同。
    """
    print("\n--- Comparing final quantized output ---")
    
    y_fused_cpu = y_fused.cpu()
    y_baseline_cpu = y_baseline.cpu()
    
    is_equal = torch.equal(y_fused_cpu, y_baseline_cpu)
    
    if is_equal:
        print("✅ PASSED: Fused operator output is bit-for-bit identical to the baseline.")
        return True
    else:
        print("❌ FAILED: Fused operator output has differences compared to the baseline.")
        diff = torch.abs(y_fused_cpu.to(torch.int32) - y_baseline_cpu.to(torch.int32))
        print(f"   - Mismatched elements count: {(diff > 0).sum().item()}")
        print(f"   - Max Absolute Error: {diff.max().item()}")
        return False

# =============================================================================
# 模块3: 主测试流程
# =============================================================================
if __name__ == '__main__':
    if not torch.npu.is_available():
        print("NPU device not available. Skipping the test.")
    else:
        # --- 1. 初始化设备和测试配置 ---
        device = 'npu:0'
        torch.npu.set_device(device)
        input_dtype = torch.bfloat16

        test_cases = {
            "NORMAL_MODE": {"shape": (4, 512, 4096)},
            # "SPLIT_D_MODE": {"shape": (1, 1, 65536)},
            # "SINGLE_N_MODE": {"shape": (1, 128, 1024)}
        }
        
        # --- 2. 循环执行所有测试场景 ---
        for name, params in test_cases.items():
            print(f"\n{'='*70}")
            print(f"RUNNING TEST CASE: [{name}] with shape {params['shape']}")
            print(f"{'='*70}")
            
            # --- a. 准备NPU输入数据 (只准备需要的参数) ---
            torch.manual_seed(123)
            shape, hidden_size = params['shape'], params['shape'][-1]
            print("shape:",shape)
            print("hidden_size:",hidden_size)
            x1_cpu = np.random.randn(4, 512, 4096).astype(bfloat16)
            x1_npu = torch.from_numpy(x1_cpu.astype(np.float32)).to(input_dtype).to(device)
            x2_cpu = np.random.randn(4, 512, 4096).astype(bfloat16)
            x2_npu = torch.from_numpy(x2_cpu.astype(np.float32)).to(input_dtype).to(device)
            gamma_cpu = np.random.randn(hidden_size).astype(bfloat16)
            gamma_npu = torch.from_numpy(gamma_cpu.astype(np.float32)).to(input_dtype).to(device)
            bias_cpu = np.random.randn(hidden_size).astype(bfloat16)
            bias_npu = torch.from_numpy(bias_cpu.astype(np.float32)).to(input_dtype).to(device)
            
            # 为Per-Channel量化创建长度为hidden_size的张量
            scales1_npu = torch.randn(hidden_size, dtype=torch.float32, device=device).abs() + 0.01
            zero_points1_npu = torch.randint(-5, 5, (hidden_size,), dtype=torch.int32, device=device)
            
            # --- b. 运行NPU单算子基线 ---
            print("Step 1: Running NPU Step-by-Step Baseline...")
            y_baseline = npu_simplified_baseline(
                x1_npu, x2_npu, gamma_npu, scales1_npu,
                bias_npu, zero_points1_npu
            )
            torch.npu.synchronize()



            x1_cpu.tofile("./input/input_x1.bin")
            x2_cpu.tofile("./input/input_x2.bin")
            gamma_cpu.tofile("./input/input_gamma.bin")
            bias_cpu.tofile("./input/input_bias.bin")
            scales1_npu.cpu().numpy().tofile("./input/input_scales1.bin")
            zero_points1_npu.cpu().numpy().tofile("./input/input_zero_points1.bin")
            y_baseline.cpu().numpy().tofile("./output/golden.bin")
            print("x1_npu shape:",x1_npu.shape)
            print("x2_npu shape:",x2_npu.shape)
            print("gamma_npu shape:",gamma_npu.shape)
            print("bias_npu shape:",bias_npu.shape)
            print("scales1_npu shape:",scales1_npu.shape)
            print("zero_points1_npu shape:",zero_points1_npu.shape)
            print("y_baseline shape:",y_baseline.shape)
          
            
           