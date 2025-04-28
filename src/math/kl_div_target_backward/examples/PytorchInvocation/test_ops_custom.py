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

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False


class TestCustomKlDivTargetBackward(TestCase):

    def test_kl_div_target_backward(self):
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

        torch.npu.synchronize()
        output = torch_npu.npu_kl_div_target_backward(gradOutput.npu(), selfX.npu(), target.npu(), reduction, logTarget).cpu()
        torch.npu.synchronize()

        print(output)
        self.assertRtolEqual(output, gradTarget)


if __name__ == "__main__":
    run_tests()
