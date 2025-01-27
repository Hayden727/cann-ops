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
 * @file function.h
 */
#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <ATen/ATen.h>

at::Tensor my_op_impl_autograd(const at::Tensor &self, const at::Tensor &other);
at::Tensor my_op_impl_autograd1(const at::Tensor &self, const at::Tensor &other);

#endif //  FUNCTION_H_
