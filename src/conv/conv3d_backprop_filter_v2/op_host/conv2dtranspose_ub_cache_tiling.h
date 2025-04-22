/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

/*!
 * \file cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CONV2DTRANSPOSE_UB_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_CONV2DTRANSPOSE_UB_CACHE_TILING_H

#include <iostream>
#include <array>

#include "cube_tiling_runtime.h"
#include "cube/include/cube_tiling_param.h"
// #include "op_log.h"

namespace optiling {
struct ConvTTilingRunInfo {
  int32_t c1Aub = 1;
  int32_t hBub = 1;
  int32_t wBub = 1;
  int32_t batchUb = 1;
  int32_t c1Cub = 1;
  int32_t hCub = 1;
  int32_t wCub = 1;
};

struct TensorUbSize {
  int32_t fifoResUbSize = 0;
  int32_t fmUbSize = 0;
  int32_t weightUbSize = 0;
  int32_t biasUbSize = 0;
  int32_t deqUbSize = 0;
  int32_t resUbSize = 0;
  int32_t fmUbSizeSplitBatch = 0;
  int32_t weightUbSizeSplitCout = 0;
  int32_t resUbSizeSplitBatch = 0;
  int32_t resUbSizeSplitBatchCout = 0;
  int32_t biasUbSizeSplitCout = 0;
  int32_t deqUbSizeSplitCout = 0;
  int32_t resUbSizeSplitBatchHW = 0;
  int32_t fmUbSizeSplitBatchHW = 0;
  int32_t fmUbSizeSplitBatchCin = 0;
  int32_t weightUbSizeSplitCin = 0;
  int32_t weightUbSizeSplitCinCout = 0;
};

struct ConvTRunParasTemp {
  int32_t c1Aub = 1;
  int32_t c1Cub = 1;
  int32_t hCub = 1;
  int32_t wCub = 1;
};

struct ConvTRunParas {
  int32_t coreNum = 1;
  int32_t ubSize = 128 * 1024;

  int32_t availableUbSize = 0;

  int32_t oriCAub = 1;
  int32_t oriHAub = 1;
  int32_t oriWAub = 1;
  int32_t oriHBub = 1;
  int32_t oriWBub = 1;
  int32_t oriBatchUb = 1;
  int32_t oriCCub = 1;
  int32_t oriHCub = 1;
  int32_t oriWCub = 1;
  int32_t c1Aub = 1;
  int32_t hBub = 1;
  int32_t wBub = 1;
  int32_t batchUb = 1;
  int32_t c1Cub = 1;
  int32_t hCub = 1;
  int32_t wCub = 1;
  int32_t fmC0 = 16;
  int32_t wK0 = 16;
  int32_t wN0 = 16;
  int32_t resC0 = 16;
  int32_t dyDtypeBytes = 1;
  int32_t weightDtypeBytes = 1;
  int32_t resDtypeBytes = 4;
  int32_t biasDtypeBytes = 4;
  int32_t deqDtypeBytes = 2;
  int32_t aPb = 1;
  int32_t bPb = 1;
  int32_t cPb = 1;

  bool fifoFusionFlag = false;
  bool reorderMnFlag = false;
  bool splitCoutFlag = false;
  bool splitKFlag = false;
  bool biasFlag = false;
};

struct ConvTParas {
  const ConvTCompileParas *compileParams = nullptr;
  ConvTRunParas *runParams = nullptr;
};

struct ConvTCoreStatus {
  int32_t coreNumUse = 1;
  int32_t fmCCoreMain = 1;
  int32_t wHCoreMain = 1;
  int32_t wWCoreMain = 1;
  int32_t resCCoreMain = 1;
  int32_t resHCoreMain = 1;
  int32_t resWCoreMain = 1;
  int32_t batchCoreMain = 1;
};

struct ConvTSingleCoreStatus {
  int32_t fmCSingleCore = 1;
  int32_t wHSingleCore = 1;
  int32_t wWSingleCore = 1;
  int32_t resCSingleCore = 1;
  int32_t resHSingleCore = 1;
  int32_t resWSingleCore = 1;
  int32_t batchSingleCore = 1;

  int32_t c1Aub = 1;
  int32_t hBub = 1;
  int32_t wBub = 1;
  int32_t batchUb = 1;
  int32_t c1Cub = 1;
  int32_t hCub = 1;
  int32_t wCub = 1;

  bool reorderMnFlag = false;
  bool splitCoutFlag = false;
  bool splitKFlag = false;
};

class ConvTTiling {
public:
  uint64_t tilingId;
  int32_t blockM = 1;
  int32_t blockN = 1;
  int32_t c1Aub = 1;
  int32_t hBub = 1;
  int32_t wBub = 1;
  int32_t batchUb = 1;
  int32_t c1Cub = 1;
  int32_t hCub = 1;
  int32_t wCub = 1;
  ConvTTiling() = default;
  void SetParams(const std::string &opType, const ConvTSingleCoreStatus &singleCoreStatus);
  void GetTilingId(ConvTParas params, ConvTSingleCoreStatus singleCoreStatus);
  ~ConvTTiling() = default;
};

bool ConvTGenTiling(const std::string &opType, const ConvTCompileParas &compileParams,
                    ConvTRunParas &runParams, ConvTTiling &tiling, uint64_t &tilingId);
ge::graphStatus TilingForConv2DTransposeUb(gert::TilingContext *context);
}; // namespace optiling

#endif