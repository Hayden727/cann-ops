#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import os
import sys
import time
import numpy as np
import torch
import torch_npu


debug_switch = False


class Result:
    def __init__(self, result_name, total_big_num=0, total_big_ratio=0, diff_big_max=0, diff_big_avg=0, diff_big_sum=0,
                 total_small_num=0, total_small_ratio=0, err_small_num=0, err_small_ratio=0,
                 diff_rmse=0, rst_eb=0, diff_eb=0,
                 num_total_nan=0, err_total_nan=0, num_total_inf=0, err_total_inf=0, num_total_ninf=0,
                 err_total_ninf=0):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf

    # 打印精度结果细节
    def print_result(self):
        print(f"正在打印结果：{self.result_name}")
        print(f" 大值总数：{self.total_big_num}")
        print(f" 大值占比：{self.total_big_ratio:.2%}")
        print(f" 大值最大误差：{self.diff_big_max:.8f}")
        print(f" 大值平均误差：{self.diff_big_avg:.8f}")
        print(f" 大值误差总和：{self.diff_big_sum:.2f}")
        print(f" 小值总数：{self.total_small_num}")
        print(f" 小值占比：{self.total_small_ratio:.2%}")
        print(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print(f" 均衡性偏差计数：{self.rst_eb}")
        print(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0) or \
                (self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0) or True:
            print(f" golden nan总数：{self.num_total_nan}")
            print(f" nan误差数：{self.err_total_nan}")
            print(f" golden inf总数：{self.num_total_inf}")
            print(f" inf误差数：{self.err_total_inf}")
            print(f" golden -inf总数：{self.num_total_ninf}")
            print(f" -inf误差数：{self.err_total_ninf}")

    # 解析精度报错细节
    def check_result_debug(self, benchmark):
        reason_str = ''
        if self.diff_big_max > benchmark.diff_big_max * 10:
            reason_str += ' diff_big_max error,'
        elif self.diff_big_max > benchmark.diff_big_max:
            reason_str += ' diff_big_max warning,'
        if self.diff_big_avg > benchmark.diff_big_avg * 2:
            reason_str += ' diff_big_avg error,'
        elif self.diff_big_avg > benchmark.diff_big_avg:
            reason_str += ' diff_big_avg warning,'
        if self.diff_big_sum > benchmark.diff_big_sum * 2:
            reason_str += ' diff_big_sum error,'
        elif self.diff_big_sum > benchmark.diff_big_sum:
            reason_str += ' diff_big_sum warning,'

        if self.err_small_num > benchmark.err_small_num * 2:
            reason_str += ' err_small_num error,'
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += ' err_small_num warning,'

        if self.diff_rmse > benchmark.diff_rmse * 2:
            reason_str += ' diff_rmse error,'
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += ' diff_rmse warning,'

        # if self.rst_eb > benchmark.rst_eb*4:
        #     reason_str += ' rst_eb error,'
        # elif self.rst_eb > benchmark.rst_eb*2:
        #     reason_str += ' rst_eb warning,'

        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += ' err_total_nan error,'
        elif self.err_total_nan > 0:
            reason_str += ' err_total_nan warning,'
        if self.err_total_inf > benchmark.err_total_inf or self.err_total_ninf > benchmark.err_total_ninf:
            reason_str += ' err_total_inf error,'
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += ' err_total_inf warning,'

        return reason_str

    # 与竞品对比精度结果，benchmark传入gpu竞品数据或基线版本数据，返回检查结果与检查不通过原因
    def check_result(self, benchmark):
        # print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        if self.diff_big_max > benchmark.diff_big_max * 10 or \
                self.diff_big_avg > benchmark.diff_big_avg * 2 or \
                self.diff_big_sum > benchmark.diff_big_sum * 2 or \
                self.err_small_num > benchmark.err_small_num * 2 or \
                self.diff_rmse > benchmark.diff_rmse * 2:
            # print('diff_big_max(大于0即error)', self.diff_big_max - benchmark.diff_big_max * 10)
            # print('diff_big_sum(大于0即error)', self.diff_big_sum - benchmark.diff_big_sum * 2)
            # print('err_small_num(大于0即error)', self.err_small_num - benchmark.err_small_num * 2)
            # print('diff_rmse(大于0即error)', self.diff_rmse - benchmark.diff_rmse * 2)
            # print(self.result_name + 'compare result: error')
            reason_str = self.check_result_debug(benchmark)
            return 'error', reason_str

        # print(self.result_name + 'compare result: ok')
        return 'ok', ''


def checkResult(value, golden, name):
    # print(f"info：开始计算 {name} 精度。")
    if value.shape == golden.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(value, golden)):
            # print(f"info：{name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            if value.numel() == 0:
                return Result(name)
        # inf nan对比
        mask_golden_is_nan = torch.isnan(golden)
        mask_value_is_nan = torch.isnan(value)
        num_total_nan = torch.sum(mask_golden_is_nan)
        err_total_nan = torch.sum(mask_golden_is_nan.logical_xor(mask_value_is_nan))

        # 将所有inf处理为边界值（inf误差转换为数值误差）
        golden[golden == torch.inf] = torch.finfo(value.dtype).max
        golden[golden == -torch.inf] = torch.finfo(value.dtype).min
        value[value == torch.inf] = torch.finfo(value.dtype).max
        value[value == -torch.inf] = torch.finfo(value.dtype).min

        mask_golden_is_inf = torch.isinf(golden) & (golden > 0)
        mask_value_is_inf = torch.isinf(value) & (value > 0)
        num_total_inf = torch.sum(mask_golden_is_inf)
        err_total_inf = torch.sum(mask_golden_is_inf.logical_xor(mask_value_is_inf))

        mask_golden_is_ninf = torch.isinf(golden) & (golden < 0)
        mask_value_is_ninf = torch.isinf(value) & (value < 0)
        num_total_ninf = torch.sum(mask_golden_is_ninf)
        err_total_ninf = torch.sum(mask_golden_is_ninf.logical_xor(mask_value_is_ninf))

        # if debug_switch:
        #     print(f" inf/nan总数：{num_total_nan + num_total_inf + num_total_ninf}")
        #     print(f" inf/nan误差数：{err_total_nan + err_total_inf + err_total_ninf}")

        # 对inf/nan统一赋1，忽略影响
        golden[torch.isinf(golden)] = 1
        value[torch.isinf(value)] = 1
        golden[torch.isnan(golden)] = 1
        value[torch.isnan(value)] = 1

        if value.dtype == torch.float16:
            small_value = 0.001
            small_value_atol = 0.00001
        elif value.dtype == torch.bfloat16:
            small_value = 0.001
            small_value_atol = 0.00001
        elif value.dtype == torch.float32:
            small_value = 0.000001
            small_value_atol = 0.000000001
        else:
            small_value = 0.000025
            small_value_atol = 0.000000001

        # 大值对比
        total_big_num = torch.sum(golden >= small_value)
        total_big_ratio = total_big_num / golden.numel()

        # 对小值统一赋1，忽略影响
        value_big = value.clone()
        value_big[golden < small_value] = 1
        golden_big = golden.clone()
        golden_big[golden < small_value] = 1

        diff_big = torch.abs(value_big.sub(golden_big))
        diff_big_max = diff_big.max()
        diff_big_sum = diff_big.sum()
        diff_big_avg = diff_big_sum / total_big_num
        diff_big_ratio = diff_big / golden_big

        # if debug_switch:
        #     print(f" 大值总数：{total_big_num}")
        #     print(f" 大值占比：{total_big_ratio:.2%}")
        #     print(f" 大值最大误差：{diff_big_max:.8f}")
        #     print(f" 大值平均误差：{diff_big_avg:.8f}")
        #     print(f" 大值误差总和：{diff_big_sum:.2f}")

        # 小值对比
        total_small_num = torch.sum(golden < small_value)
        total_small_ratio = total_small_num / golden.numel()

        # 对大值统一赋1，忽略影响
        value_small = value.clone()
        value_small[golden > small_value] = 1
        golden_small = golden.clone()
        golden_small[golden > small_value] = 1

        diff_small = torch.abs(value_small.sub(golden_small))
        err_small_num = torch.sum(diff_small > small_value_atol)
        err_small_ratio = err_small_num / total_small_num

        # if debug_switch:
        #     print(f" 小值总数：{total_small_num}")
        #     print(f" 小值占比：{total_small_ratio:.2%}")
        #     print(f" 小值错误数：{err_small_num}，占比{err_small_ratio:.2%}")

        # 计算均方根误差（rmse）
        diff = torch.abs(value.sub(golden))
        diff_rmse = torch.sqrt(torch.mean(torch.square(diff)))
        # if debug_switch:
        #     print(f" 误差均方根（RMSE）：{diff_rmse:.8f}")

        # 计算误差均衡性（eb）
        eb_bigger = torch.sum(value > golden)
        eb_smaller = torch.sum(value < golden)
        rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
        diff_eb = torch.sum(value.sub(golden))
        # if debug_switch:
        #     print(f" 均衡性偏差计数：{rst_eb}")
        #     print(f" 均衡性diff总和：{diff_eb:.8f}")

        return Result(name, total_big_num, total_big_ratio, diff_big_max, diff_big_avg, diff_big_sum,
                      total_small_num, total_small_ratio, err_small_num, err_small_ratio, diff_rmse, rst_eb,
                      diff_eb,
                      num_total_nan, err_total_nan, num_total_inf, err_total_inf, num_total_ninf, err_total_ninf)
    else:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: 输入shape {value.shape}")
        print(f"debug: 真值shape  {golden.shape}")

    return

def data_compare(cpu_data, gpu_data, npu_data, rank):
    rst_npu = checkResult(npu_data, cpu_data, "{}_dq_npu".format(rank))
    # rst_npu.print_result()
    rst_gpu = checkResult(gpu_data, cpu_data, "{}_dq_gpu".format(rank))
    # rst_gpu.print_result()
    str1, str2 = rst_npu.check_result(rst_gpu)
    if 'error' in str1:
        res = False
    else:
        res = True
    # print('[INFO] device_{} 精度结果为：{}'.format(rank, res))
    # print('[INFO] device_{} 精度计算结束：{} '.format(rank, time.strftime('%H:%M:%S', time.localtime())))
    return res


def verify_result(out_dir, output_dtype='fp16'):
    for rank in range(1):
        cpu_out_path = f"{out_dir}/cpu_out_{rank}.bin"
        gpu_out_path = f"{out_dir}/gpu_out_{rank}.bin"
        npu_out_path = f"{out_dir}/out_{rank}.bin"
        if output_dtype == 'fp16':
            np_dtype = np.float16
        else:
            import tensorflow as tf
            np_dtype = tf.bfloat16.as_numpy_dtype
        cpu_out = torch.tensor(np.fromfile(cpu_out_path, np.float32))
        gpu_out = torch.tensor(np.fromfile(gpu_out_path, np_dtype).astype(np.float32))
        gpu_out = gpu_out.cpu().view(-1)
        npu_out = torch.tensor(np.fromfile(npu_out_path, np_dtype).astype(np.float32))
        npu_out = npu_out.cpu().view(-1)

        if not data_compare(cpu_out, gpu_out, npu_out, rank):
            print("============ out_{} precession check failed", rank)
            return False

    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1])

