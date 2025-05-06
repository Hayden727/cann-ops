/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *Conv2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv2d5HdFp1625HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                       const aclIntArray *stride, const aclIntArray *padding,
                                       const aclIntArray *dilation, int groups, aclOpExecutor *executor);

const aclTensor *Conv2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv26HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, bool useHf32, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3dv2NCDHWFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *Conv2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                               const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
                               int groups, aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor);

const aclTensor *ConvTranspose2d5HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose3d6HdFp16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

const aclTensor *ConvTranspose3d6HdFp32(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        bool useHf32, aclOpExecutor *executor);
                                        
const aclTensor *ConvTranspose3d6HdBf16(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                        const aclIntArray *stride, const aclIntArray *padding,
                                        const aclIntArray *dilation, int groups, const aclIntArray *outputPadding,
                                        aclOpExecutor *executor);

bool IsSupportConv3DToConv3DV2();
}  // namespace l0op

#endif  // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTION_OP_H_
