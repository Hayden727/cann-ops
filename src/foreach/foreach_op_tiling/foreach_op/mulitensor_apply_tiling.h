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
 * \file mulitensor_apply_tiling.h
 * \brief
 */
#ifndef ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILING_H
#define ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILING_H

#include <vector>
#include "mulitensor_apply_tilingdata.h"
#include "foreach_op_info.h"

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint64_t SINGLE_CORE_LEATEST_DATACOUNT = 256;  // BYTE half memory throupht

GetMaxResidentCntT maxResidentCntInUB[END_OPCODE] = {nullptr};

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
class MultiTensorApplyTiling {
 public:
  explicit MultiTensorApplyTiling(int32_t opCode,
                                  const std::vector<std::vector<std::vector<int64_t>>>& tensorShapeLists,
                                  uint32_t coreNum, uint32_t ubSize)
      : allInputShapes(tensorShapeLists) {
    this->opCode = opCode;
    this->hwCoreNum = coreNum;
    this->hwUBSize = FloorAlign(ubSize, BYTE_BLOCK);
    this->dataTypeSize = dataTypeLen;
    this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
    this->totalTensorCount = allInputShapes[0].size();
    return;
  }

  int32_t Init();
  int32_t GetTiling(TilingData* tilingData);

 private:
  void GetBlockTiling();
  void GetUbtiling();
  void SplitDataToEachCore();
  void FillTilingData(TilingData* tilingData);
  template <typename T1, typename T2>
  inline T1 CeilDiv(T1 a, T2 b) {
    if (b != 0) {
      return (a + b - 1) / b;
    }
    return a;
  }

  template <typename T1, typename T2>
  inline T1 FloorDiv(T1 a, T2 b) {
    if (b != 0) {
      return (a) / (b);
    }
    return a;
  }

  template <typename T1, typename T2>
  inline T1 CeilAlign(T1 a, T2 b) {
    if (b != 0) {
      return (a + b - 1) / b * b;
    }
    return a;
  }

  template <typename T1, typename T2>
  inline T1 FloorAlign(T1 a, T2 b) {
    if (b != 0) {
      return (a) / b * b;
    }
    return a;
  }

 private:
  std::vector<std::vector<std::vector<int64_t>>> allInputShapes;
  uint32_t hwCoreNum = 0;
  uint32_t hwUBSize = 0;
  int64_t totalDataCount = 0;
  uint8_t dataTypeSize = 0;
  uint8_t elementsPerBlock = 0;
  uint32_t totalTensorCount = 0;

  // tilingData
  int64_t tensorDataCountList[DEPTH_TO_MAX_TENSORS[0]] = {0};
  uint16_t listStartIdx[MAX_CORE_COUNT] = {0};
  uint16_t listEndIdx[MAX_CORE_COUNT] = {0};
  int64_t tensorStartOffset[MAX_CORE_COUNT] = {0};
  int64_t tensorEndOffset[MAX_CORE_COUNT] = {0};
  uint32_t ubFactorElement = 0;
  uint32_t blockNum = 0;
  int32_t opCode = 0;
};

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
int32_t MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::Init() {
  INIT_MAX_RESIDENT_CNT_IN_UB();
  totalDataCount = 0;
  for (size_t i = 0; i < allInputShapes[0].size(); i++) {
    tensorDataCountList[i] = 1;
    for (size_t j = 0; j < allInputShapes[0][i].size(); j++) {
      tensorDataCountList[i] *= allInputShapes[0][i][j];
    }
    totalDataCount += tensorDataCountList[i];
  }
  return 0;
}

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
void MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::GetBlockTiling() {
  uint32_t leastCountElements = SINGLE_CORE_LEATEST_DATACOUNT / dataTypeSize;
  uint32_t tempCoreNum = static_cast<uint32_t>(CeilDiv(totalDataCount, leastCountElements));
  if (tempCoreNum < hwCoreNum) {
    blockNum = tempCoreNum;
  } else {
    blockNum = hwCoreNum;
  }
  SplitDataToEachCore();
}

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
int32_t MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::GetTiling(TilingData* tilingData) {
  GetBlockTiling();
  GetUbtiling();
  FillTilingData(tilingData);
  return 0;
}

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
void MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::GetUbtiling() {
  // enable double, and all compute need to align to 32B
  int32_t maxLiveNodeCount = 0;
  int32_t extraBuf = 0;
  int32_t reserveBaseBlock = BYTE_BLOCK;
  GetMaxResidentCntInUB(this->opCode, maxLiveNodeCount, extraBuf);
  uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
  uint32_t canUseUbSize = FloorAlign((hwUBSize - tilingDataSize - reserveBaseBlock) / 2, BYTE_BLOCK);
  this->ubFactorElement =
      FloorAlign((canUseUbSize - CeilAlign(extraBuf, BYTE_BLOCK)) / (maxLiveNodeCount + depth), BYTE_BLOCK) /
      dataTypeSize;
}

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
void MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::SplitDataToEachCore() {
  // split data to each core based on block-wise
  int64_t totalBlockCount = CeilDiv(totalDataCount, elementsPerBlock);
  int64_t mainCoreDataCount = FloorDiv(totalBlockCount, this->blockNum) * elementsPerBlock;
  int64_t tailCoreCount = totalBlockCount % this->blockNum;
  uint16_t coreIndex = 0;
  int64_t dataCount = 0;
  int64_t curCoreDataCount = 0;
  int64_t curTensorPos = 0;
  listStartIdx[coreIndex] = 0;
  tensorStartOffset[coreIndex] = 0;
  for (uint16_t i = 0; i < this->totalTensorCount; i++) {
    // add extra block-wise data to the first core
    if (tailCoreCount && coreIndex < tailCoreCount) {
      curCoreDataCount = mainCoreDataCount + elementsPerBlock;
    } else {
      curCoreDataCount = mainCoreDataCount;
    }
    int64_t tempCount = tensorDataCountList[i] - curTensorPos;
    if (dataCount + tempCount < curCoreDataCount) {
      dataCount += tempCount;
      curTensorPos = 0;
      continue;
    }

    listEndIdx[coreIndex] = i;
    curTensorPos = curTensorPos + curCoreDataCount - dataCount;
    tensorEndOffset[coreIndex] = curTensorPos - 1;
    dataCount = 0;
    coreIndex++;
    if (curTensorPos < tensorDataCountList[i]) {
      listStartIdx[coreIndex] = i;
      tensorStartOffset[coreIndex] = curTensorPos;
      --i;  // keep in the same tensor
    } else if (coreIndex != this->blockNum) {
      listStartIdx[coreIndex] = i + 1;
      tensorStartOffset[coreIndex] = 0;
      curTensorPos = 0;
    }
  }
  // last data trunk of last tensor
  if (dataCount > 0) {
    listEndIdx[coreIndex] = totalTensorCount - 1;
    tensorEndOffset[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
  }
}

template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
void MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen>::FillTilingData(TilingData* tilingData) {
  tilingData->set_opCode(opCode);
  tilingData->set_tensorDataCountList(tensorDataCountList);
  tilingData->set_listStartIdx(listStartIdx);
  tilingData->set_listEndIdx(listEndIdx);
  tilingData->set_tensorStartOffset(tensorStartOffset);
  tilingData->set_tensorEndOffset(tensorEndOffset);
  tilingData->set_blockNum(blockNum);
  tilingData->set_ubFactorElement(ubFactorElement);
}

// 1<=Depth<5
template <int32_t depth, typename TilingData, int32_t dataTypeLen, int32_t scalarTypeLen>
void MultiTensorApplyGetTiling(int32_t opCode, const std::vector<std::vector<std::vector<int64_t>>>& tensorShapeLists,
                               TilingData* mtaTilingData, uint32_t coreNum, uint32_t ubSize) {
  class MultiTensorApplyTiling<depth, TilingData, dataTypeLen, scalarTypeLen> mtaTiling(opCode, tensorShapeLists,
                                                                                        coreNum, ubSize);
  mtaTiling.Init();
  mtaTiling.GetTiling(mtaTilingData);
}

}  // namespace optiling

#endif // ASL_OPS_CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_FOREACH_OP_MULTITENSOR_APPLY_TILING_H