#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchair
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
import custom_ops


# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.myops.my_op.default)
def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "AddCustom",
        inputs={
            "x": x,
            "y": y,
        },
        outputs=['z']
    )


class TestCustomAdd(TestCase):

    def test_add_custom_graph(self):

        class PlugInAdd(torch.nn.Module):

            def __init__(self):
                super().__init__()

            def forward(self, input1, input2):
                return torch.ops.myops.my_op(input1, input2)

        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        model = PlugInAdd().npu()

        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True)

        with torch.no_grad():
            output = model(x.npu(), y.npu())

        cpuout = torch.add(x, y)

        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
