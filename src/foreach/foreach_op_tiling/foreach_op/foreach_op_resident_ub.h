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
 * \file foreach_op_resident_ub.h
 * \brief
 */
#ifndef __FOREACH_OP_RESIDENT_UB_H__
#define __FOREACH_OP_RESIDENT_UB_H__

namespace optiling {

typedef void (*GetMaxResidentCntT)(int32_t opCode, int32_t& maxLiveNodeCnt, int32_t& extraBuf);

extern GetMaxResidentCntT maxResidentCntInUB[];

void GetMaxResidentCntInUB(int32_t opCode, int32_t& maxLiveNodeCnt, int32_t& extraBuf) {
  if (maxResidentCntInUB[opCode] != nullptr) {
    (*maxResidentCntInUB[opCode])(opCode, maxLiveNodeCnt, extraBuf);
  }
}

#define REGISTER_OP_UB_BUFFER_USE(KEYID, GET_UB_USE_FUN_ADDR) \
  { maxResidentCntInUB[KEYID] = GET_UB_USE_FUN_ADDR; }

#define BEGIN_REGISTER_USER_OP_UB_BUFFER_USE void INIT_MAX_RESIDENT_CNT_IN_UB() {
#define END_REGISTER_USER_OP_UB_BUFFER_USE }

}  // namespace optiling
#endif  // __FOREACH_OP_RESIDENT_UB_H__
