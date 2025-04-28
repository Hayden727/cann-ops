#!/usr/bin/python3
# coding=utf-8
#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from typing import Any
import torch
from torch.library import impl
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.meta._meta_registrations import m


@impl(m, "npu_kl_div_target_backward")
def npu_kl_div_target_backward_meta(grad_output, self, target, reduction, log_target):
    return torch.empty_like(target)


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.npu.npu_kl_div_target_backward.default)
def convert_npu_kl_div_target_backward(grad_output: Tensor, self: Tensor, target: Tensor, reduction: Int, log_target: Bool, grad_target: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "KlDivTargetBackward",
        inputs={
            "grad_output": grad_output,
            "self": self,
            "target": target,
            "reduction": reduction,
            "log_target": log_target,
        },
        outputs=['grad_target']
    )


class TestTorchCompileCustomKlDivTargetBackward(TestCase):

    def test_kl_div_target_backward(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        length = [8, 2048]
        gradOutput = torch.rand(length, device='cpu', dtype=torch.float16)
        selfX = torch.rand(length, device='cpu', dtype=torch.float16)
        target = torch.rand(length, device='cpu', dtype=torch.float16)
        reduction = 0
        logTarget = False
        print(gradOutput, '\n', selfX, '\n', target)
        if logTarget:
            gradTarget = target + 1
            gradTarget = gradTarget - selfX
            tmp = torch.exp(target)
            gradTarget = gradTarget * tmp
            gradTarget = gradOutput * gradTarget
        else:
            tmp = torch.log(target)
            gradTarget = tmp + 1
            gradTarget = gradTarget - selfX
            gradTarget = gradOutput * gradTarget
            gradTarget = gradTarget.masked_fill(target==0, 0)

        if reduction == 1:
            gradTarget = gradTarget / target.numel()
        
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, gradOutput, selfX, target, reduction, logTarget):
                return torch_npu.npu_kl_div_target_backward(gradOutput, selfX, target, reduction, logTarget)
        mod = torch.compile(Module().npu(), backend=npu_backend)
        output = mod(gradOutput, selfX, target, reduction, logTarget)
        print(output)
        self.assertRtolEqual(output, gradTarget)


if __name__ == "__main__":
    run_tests()
