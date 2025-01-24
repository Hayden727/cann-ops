/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file register.cpp
 */
#include <torch/extension.h>
#include <torch/library.h>

#include "function.h"

// Register two schema: my_op and my_op_backward in the myops namespace
TORCH_LIBRARY(myops, m)
{
    m.def("my_op(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward(Tensor self) -> (Tensor, Tensor)");
    m.def("my_op1(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward1(Tensor self) -> (Tensor, Tensor)");
}

// bind c++ interface to python interface by pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_custom", &my_op_impl_autograd, "x + y");
    m.def("add_custom1", &my_op_impl_autograd1, "x + y");
}
