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
 * @file custom_assign_add_custom.cc
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// 注册TensorFlow自定义算子
REGISTER_OP("AddCustom")                                    // TensorFlow自定义算子名称
    .Input("x: T")                                          // 输入tensor x
    .Input("y: T")                                          // 输入tensor y
    .Output("z: T")                                         // 输出tensor z
    .Attr("T: {half}")                                      // 属性T，支持half数据类型
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn); // 设置shape函数，用于推断输出tensor的shape，BroadcastBinaryOpShapeFn函数用于处理输入、输出tensor的shape相同的情况


// TensorFlow自定义算子的CPU实现
class AddCustomOp : public OpKernel {
public:
    explicit AddCustomOp(OpKernelConstruction* context) : OpKernel(context) {}
    // 当前算子不支持CPU设备，实现该函数以抛出异常，提示该算子不支持CPU设备
    void Compute(OpKernelContext* context) override {
        OP_REQUIRES(context, false, errors::Unimplemented("AddCustomOp is not supported on CPU"));
    }
};

// 注册TensorFlow自定义算子的CPU实现
REGISTER_KERNEL_BUILDER(Name("AddCustom").Device(DEVICE_CPU), AddCustomOp);