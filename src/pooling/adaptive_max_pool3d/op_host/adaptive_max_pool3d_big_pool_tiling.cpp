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

/*!
 * \file adaptive_max_pool3d_big_pool_tiling.cc
 * \brief
 */

#include "adaptive_max_pool3d_tiling.h"

constexpr uint64_t TILING_KEY_BIG_KERNEL_FLOAT = 310000;
constexpr uint64_t TILING_KEY_BIG_KERNEL_HALF = 311000;
constexpr uint64_t TILING_KEY_BIG_KERNEL_BF16 = 312000;

using namespace AscendC;

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
bool AdaptiveMaxPool3dBigPoolTiling::IsCapable() {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigPoolTiling IsCapable check.");
  return true;
}

void AdaptiveMaxPool3dBigPoolTiling::SetTilingData() {
  tilingdata_.set_N(input_.N);
  tilingdata_.set_C(input_.C);
  tilingdata_.set_Di(input_.Di);
  tilingdata_.set_Hi(input_.Hi);
  tilingdata_.set_Wi(input_.Wi);
  tilingdata_.set_Do(input_.Do);
  tilingdata_.set_Ho(input_.Ho);
  tilingdata_.set_Wo(input_.Wo);
  tilingdata_.set_coreNums(input_.coreNum);
  tilingdata_.set_useCoreNum(calInfo_.useCoreNum);
  tilingdata_.set_totalIdx(calInfo_.totalIdx);
  tilingdata_.set_blockFactor(calInfo_.blockFactor);
  tilingdata_.set_blockTail(calInfo_.blockTail);
}

ge::graphStatus AdaptiveMaxPool3dBigPoolTiling::DoOpTiling() {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigPoolTiling DoOpTiling start.");
  calInfo_.totalIdx = input_.N * input_.C * input_.Do * input_.Ho * input_.Wo;  // 总共的输出点
  if(input_.coreNum != 0) {
    calInfo_.blockFactor = calInfo_.totalIdx / input_.coreNum;
    calInfo_.blockTail = calInfo_.totalIdx % input_.coreNum;
  } else {
    return ge::GRAPH_FAILED;
  }
  if(calInfo_.blockFactor == 0) {
    calInfo_.useCoreNum = calInfo_.totalIdx;
  } else {
    calInfo_.useCoreNum = input_.coreNum;
  }

  SetTilingData();
  return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveMaxPool3dBigPoolTiling::GetTilingKey() const {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigPoolTiling GetTilingKey start.");
  uint64_t tilingKey = TILING_KEY_BIG_KERNEL_FLOAT;
  if (input_.xDtype == ge::DT_FLOAT) {
    tilingKey = TILING_KEY_BIG_KERNEL_FLOAT;
  } else if (input_.xDtype == ge::DT_FLOAT16) {
    tilingKey = TILING_KEY_BIG_KERNEL_HALF;
  } else {
    tilingKey = TILING_KEY_BIG_KERNEL_BF16;
  }
  return tilingKey;
}

ge::graphStatus AdaptiveMaxPool3dBigPoolTiling::PostTiling() {
  OP_LOGD(context_->GetNodeName(), "AdaptiveMaxPool3dBigPoolTiling PostTiling start.");
  context_->SetBlockDim(tilingdata_.get_useCoreNum());
  tilingdata_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingdata_.GetDataSize());

  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("AdaptiveMaxPool3d", AdaptiveMaxPool3dBigPoolTiling, 3);
}  // namespace optiling
