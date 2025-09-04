import torch
import torch_npu
from typing import Tuple

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
    print("================norm_result===============\n")
    print(norm_result[0][0][:32])

    # 步骤 2: Add Bias
    pre_quant_result = norm_result + bias
    # pre_quant_result = norm_result
    print("================pre_quant_result===============\n")
    print(pre_quant_result[0][0][:32])
    
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
            
            x1_npu = torch.randn(shape, dtype=input_dtype, device=device)
            x2_npu = torch.randn(shape, dtype=input_dtype, device=device)
            gamma_npu = torch.randn(hidden_size, dtype=input_dtype, device=device)
            bias_npu = torch.randn(hidden_size, dtype=input_dtype, device=device)
            
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


            # --- c. 运行NPU融合算子 (只传入需要的参数) ---
            print("Step 2: Running Fused Operator on NPU...")
            # 我们只关心第一个输出 y1
            y1_fused, _, _ = torch_npu.add_rms_norm_quant_v2(
                x1_npu, x2_npu, gamma_npu, scales1_npu,
                zero_points1=zero_points1_npu, 
                bias=bias_npu
            )
        
            torch.npu.synchronize()
            # print("================x1_npu===============\n")
            # print(x1_npu)
            # print("================x2_npu===============\n")
            # print(x2_npu)
            # print("================gamma_npu===============\n")
            # print(gamma_npu)
            # print("================bias_npu===============\n")
            # print(bias_npu)
            # print("================scales1_npu===============\n")
            # print(scales1_npu)
            # print("================zero_points1_npu===============\n")
            # print(zero_points1_npu)
            print("================y_baseline===============\n")
            print(y_baseline)
            print("================y_fused===============\n")
            print(y1_fused)

            # --- d. 对比结果 ---
            print("Step 3: Comparing results...")
            passed = compare_simplified_results(y1_fused, y_baseline)
            
            # --- 最终结论 ---
            if passed:
                print(f"\nFINAL VERDICT for [{name}]: ✅✅✅ PASSED ✅✅✅")
            else:
                print(f"\nFINAL VERDICT for [{name}]: ❌❌❌ FAILED ❌❌❌")