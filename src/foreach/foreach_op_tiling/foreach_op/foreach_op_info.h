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
 * \file foreach_op_info.h
 * \brief
 */
#ifndef __FOREACH_OP_INFO_H__
#define __FOREACH_OP_INFO_H__
#include "../../tbe/impl/ascendc/foreach_common/foreach_op_def.h"
#include "foreach_op_resident_ub.h"

namespace optiling {
using namespace ForeachOpDef;

constexpr int32_t EXTRA_BUF = 32;

// extraBuf compute function
void AddOnTensorListMaxUBUseCnt(int32_t opCode, int32_t& maxLiveNodeCnt, int32_t& extraBuf) {
    if (opCode > 0) {
      maxLiveNodeCnt = 0;
      extraBuf = EXTRA_BUF;
    }
}

// registern OPID and callbck
BEGIN_REGISTER_USER_OP_UB_BUFFER_USE
    // add OP_ID and callbck
    REGISTER_OP_UB_BUFFER_USE(ADD_TENSOR_LIST, AddOnTensorListMaxUBUseCnt);
END_REGISTER_USER_OP_UB_BUFFER_USE

}
#endif // __FOREACH_OP_INFO_H__