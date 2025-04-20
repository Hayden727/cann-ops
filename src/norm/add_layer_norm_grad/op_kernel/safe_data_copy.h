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
 * \file safe_data_copy.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_
#define OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "platform.h"

template <bool forAtomicAdd = false, typename T>
__aicore__ inline void SafeDataCopy(const AscendC::GlobalTensor<T> &dstGlobal, const AscendC::LocalTensor<T> &srcLocal,
    const int64_t &calCount, bool recoverUbTailFormat = false)
{
    constexpr int typeSize = sizeof(T);                                 // 元素字节数
    constexpr int numElemsPerBlock = AscendC::ONE_BLK_SIZE / typeSize;  // 32byte元素数
    if constexpr (PlatformSocInfo::IsDataCopyPadSupport() &&
                  sizeof(T) < 8) {  // 如果支持DataCopyPad则直接DataCopyPad拷贝
        AscendC::DataCopyParams copyParams{1, static_cast<uint16_t>(calCount * typeSize), 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    } else {
        if (likely(!(calCount % numElemsPerBlock))) {  // 对齐则直接DataCopy拷贝
            struct AscendC::DataCopyParams copyParams;
            copyParams.blockLen = calCount / AscendC::AscendCUtils::GetC0Count(typeSize);
            DataCopy(dstGlobal, srcLocal, copyParams);
        } else {  // 如果既不支持DataCopyPad也不对齐则地址回退拷贝
            const int numAlignedBlocks = calCount / numElemsPerBlock * numElemsPerBlock;  // 对齐部分
            if (calCount * typeSize < AscendC::ONE_BLK_SIZE) {
                DataCopy(dstGlobal, srcLocal, numElemsPerBlock);
                return;  // 此处依然有内存踩踏
            }
            DataCopy(dstGlobal, srcLocal, numAlignedBlocks);
            event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventID);
            const int rollbackEleCount = calCount - numAlignedBlocks;           // 计算需要回退处理byte数
            const size_t rollbackDstIdx = numAlignedBlocks - numElemsPerBlock;  // 操作回退的block元素索引
            const size_t rollbackSrcIdx = rollbackDstIdx + rollbackEleCount;    // 回退来源元素索引
            if constexpr (!forAtomicAdd) {
                for (int i = 0; i < numElemsPerBlock;
                     ++i) {  // 将整个非对齐的尾块以一个block的size复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i));  // 重造local buf
                }
            } else {
                const size_t setZeroEleCount = numElemsPerBlock - rollbackEleCount;  // 需要置0的元素量
                for (int i = 0; i < setZeroEleCount; ++i) {
                    srcLocal.SetValue((rollbackDstIdx + i), 0);  // Atomic模式下，回退部分置0，使得回退部分不会重复加
                }
                for (int i = setZeroEleCount; i < numElemsPerBlock; ++i) {  // 将整个非对齐的尾块复制填充到前一个block中
                    srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i));  // 重造local buf
                }
                DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
                return;  // AtomicAdd模式下 暂不支持recoverUbTailFormat，待后续扩展支持
            }
            DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
            if (recoverUbTailFormat) {  // 还原回滚现场
                event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
                DataCopy(
                    srcLocal[rollbackDstIdx], dstGlobal[rollbackDstIdx], numElemsPerBlock);  // 还原用于回退的block内容
            }
        }
    }
}

#endif  // OPS_BUILT_IN_OP_ASCENDC_SAFE_DATA_COPY_H_