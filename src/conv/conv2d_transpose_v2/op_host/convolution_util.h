/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef OP_API_SRC_CONVOLUTION_UTIL_H_
#define OP_API_SRC_CONVOLUTION_UTIL_H_

#include <map>
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"

struct ConvolutionOpInfo {
    op::DataType inputDtype;
    op::Format inputFormat;
    op::DataType weightDtype;
    op::Format weightFormat;
    op::DataType biasDtype;
    op::Format biasFormat;
    op::DataType outputDtype;
    op::Format outputFormat;
};

class Conv2DSplitWInfo {
public:
    void InitConv2DSplitWInfo(const aclTensor* input, const aclTensor* weight, const aclIntArray* stride,
                            const aclIntArray* padding, const aclIntArray* dilation);
    bool CanSwitchSplitW(const aclTensor* bias, aclTensor* output, int64_t groups, const ConvolutionOpInfo& opInfo);

private:
    bool CheckConv2DTbeOptFlag(const ConvolutionOpInfo& opInfo);
    bool CheckConv2DPad();
    bool CheckConv2DInput();
    bool CheckBasicInfoInSplitW(int64_t groups, const ConvolutionOpInfo& opInfo);
    bool CheckLoad3dIns();
    bool CheckLoadL1InSplitW(const aclTensor* bias, aclTensor* output);

private:
    int64_t hi = 0;
    int64_t wi = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t padU = 0;
    int64_t padD = 0;
    int64_t padL = 0;
    int64_t padR = 0;
    int64_t biasTypeSize = 0;
    int64_t k0 = 0;
};

aclnnStatus ChangeConv2dAttrToConv3d(const aclIntArray* &stride, const aclIntArray* &padding,
                                    const aclIntArray* &dilation, aclOpExecutor* executor);
aclnnStatus ChangeConv2dInputToConv3d(const aclTensor* &input, const aclTensor* &weight, aclOpExecutor* executor);
const aclTensor* View5dAs4dForOutput(const aclTensor* input, aclOpExecutor* executor);

#endif  // OP_API_SRC_CONVOLUTION_UTIL_H_