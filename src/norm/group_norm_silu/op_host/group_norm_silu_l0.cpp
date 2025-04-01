/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_silu_l0.cpp
 * \brief
 */

#include "group_norm_silu_l0.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupNormSilu);

const std::tuple<aclTensor*, aclTensor*, aclTensor*> GroupNormSilu(const aclTensor *x, const aclTensor *gamma,
                                                                   const aclTensor *beta,
                                                                   int64_t numGroups,
                                                                   float eps, bool activateSilu,
                                                                   aclOpExecutor *executor) {
  L0_DFX(GroupNormSilu, x, gamma, beta, numGroups, eps, activateSilu);
  auto y = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
  auto nNum = (x->GetViewShape())[0];
  auto mean = executor->AllocTensor(op::Shape({nNum, numGroups}), x->GetDataType(), op::Format::FORMAT_ND);
  auto rstd = executor->AllocTensor(op::Shape({nNum, numGroups}), x->GetDataType(), op::Format::FORMAT_ND);
  if (y == nullptr || mean == nullptr || rstd == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc output tensor failed.");
    return std::tie(y, mean, rstd);
  }
  auto retAicore =
    ADD_TO_LAUNCHER_LIST_AICORE(GroupNormSilu, OP_INPUT(x, gamma, beta), OP_OUTPUT(y, mean, rstd),
                                OP_ATTR(numGroups, eps, activateSilu));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr, nullptr),
                                       "GroupNormSilu add to aicore launch list failed.");
  return std::tie(y, mean, rstd);
}

}  // namespace l0op
