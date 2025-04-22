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
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ConvolutionBackwardInputTensor {
  const aclTensor *gradOutput;
  const aclTensor *input;
  const aclTensor *weight;
};

struct ConvolutionBackwardParams {
  const aclIntArray *biasSizes;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const bool transposed;
  const aclIntArray *outputPadding;
  const int64_t groups;
  const aclBoolArray *outputMask;
  const int8_t cubeMathType;
};

const aclTensor *CalculateConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                              ConvolutionBackwardParams &params, aclOpExecutor *executor);

#ifdef __cplusplus
}
#endif