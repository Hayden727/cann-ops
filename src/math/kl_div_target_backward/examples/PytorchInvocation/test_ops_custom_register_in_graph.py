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
from torch.library import Library
m = Library("npu", "IMPL", "Meta")


@impl(m, "npu_kl_div_target_backward")
def npu_kl_div_target_backward_meta(grad_output, self_x, target, reduction, log_target):
    return torch.empty_like(target)


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.npu.npu_kl_div_target_backward.default)
def convert_npu_kl_div_target_backward(grad_output: Tensor, self_x: Tensor, target: Tensor,
    reduction: int, log_target: bool, grad_target: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "KlDivTargetBackward",
        inputs={
            "grad_output": grad_output,
            "self": self_x,
            "target": target,
        },
        attrs={
            "reduction": torchair.ge.attr.Int(reduction),
            "log_target": torchair.ge.attr.Bool(log_target),
        },
        outputs=['grad_target']
    )


class TestTorchCompileCustomKlDivTargetBackward(TestCase):
    def test_kl_div_target_backward(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        length = [8, 2048]
        grad_output = torch.rand(length, device='cpu', dtype=torch.float16)
        self_x = torch.rand(length, device='cpu', dtype=torch.float16)
        target = torch.rand(length, device='cpu', dtype=torch.float16)
        reduction = 1
        log_target = True
        if log_target:
            grad_target = target + 1
            grad_target = grad_target - self_x
            tmp = torch.exp(target)
            grad_target = grad_target * tmp
            grad_target = grad_output * grad_target
        else:
            tmp = torch.log(target)
            grad_target = tmp + 1
            grad_target = grad_target - self_x
            grad_target = grad_output * grad_target
            grad_target = grad_target.masked_fill(target == 0, 0)

        if reduction == 1:
            max_len = max(max(grad_output.numel(), self_x.numel()), target.numel())
            grad_target = grad_target / max_len
        
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, grad_output, self_x, target, reduction, log_target):
                return torch_npu.npu_kl_div_target_backward(grad_output, self_x, target, reduction, log_target)
        mod = torch.compile(Module().npu(), backend=npu_backend)
        output = mod(grad_output.npu(), self_x.npu(), target.npu(), reduction, log_target).cpu()
        self.assertRtolEqual(output, grad_target)


if __name__ == "__main__":
    run_tests()
