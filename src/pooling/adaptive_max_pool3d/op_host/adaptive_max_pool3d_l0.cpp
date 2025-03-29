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

#include "adaptive_max_pool3d_l0.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AdaptiveMaxPool3d);

static constexpr size_t NCDHW_DIM_NUM = 5;
static constexpr size_t DIM_D = 2;
static constexpr size_t DIM_H = 3;
static constexpr size_t DIM_W = 4;
static constexpr size_t DIM_ZERO = 0;
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;

static const std::tuple<aclTensor*, aclTensor*> AdaptiveMaxPool3dAiCore(const aclTensor *self, const aclIntArray* outputSize,
                                                           aclTensor *outputOut, aclTensor *indicesOut,
                                                           aclOpExecutor *executor) {
  L0_DFX(AdaptiveMaxPool3dAiCore, self, outputSize, outputOut, indicesOut);

  ADD_TO_LAUNCHER_LIST_AICORE(AdaptiveMaxPool3d, OP_INPUT(self), OP_OUTPUT(outputOut, indicesOut), OP_ATTR(outputSize));
  return std::tuple<aclTensor*, aclTensor*>(outputOut, indicesOut);
}

const std::tuple<const aclTensor*, const aclTensor*> AdaptiveMaxPool3d(const aclTensor *self, const aclIntArray* outputSize,
                                                     aclOpExecutor *executor) {
  L0_DFX(AdaptiveMaxPool3d, self, outputSize);
  
  op::Shape outShape = self->GetViewShape();
  size_t dimNum = outShape.GetDimNum();
  if (dimNum != NCDHW_DIM_NUM) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Input dim num is not supported.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  auto input_format = self->GetViewFormat();
  outShape.SetDim(DIM_D, (*outputSize)[DIM_ZERO]);
  outShape.SetDim(DIM_H, (*outputSize)[DIM_ONE]);
  outShape.SetDim(DIM_W, (*outputSize)[DIM_TWO]);

  auto outputOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
  auto indicesOut = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());
  if (outputOut == nullptr || indicesOut == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "outputOut or indicesOut is nullptr.");
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
  }
  return AdaptiveMaxPool3dAiCore(self, outputSize, outputOut, indicesOut, executor);
}
}  // namespace l0op

