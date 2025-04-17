/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file mse_loss_tiling.h
 * \brief
 */

#ifndef MSE_LOSS_TILING_H
#define MSE_LOSS_TILING_H

#include <cstring>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, mode); 
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLoss, MseLossTilingData)

class MseLossTiling{
  public:
    explicit MseLossTiling(gert::TilingContext* context) : context_(context) {}
    ~MseLossTiling() = default;

    ge::graphStatus DoTiling() {
        auto ret = TilingFunc();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = PostTiling();
        return ret;
    }

  private:
    gert::TilingContext* context_;
    MseLossTilingData tiling;
    uint32_t BLOCK_SIZE = 32;
    uint32_t size_of_data_type;
    uint32_t totalLengthAligned;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;
    uint32_t lastTileLength = 0;
    uint32_t ub_block_num = 1024;
  private:
    ge::graphStatus TilingFunc()
    {
      uint32_t totalLength = context_->GetInputShape(0)->GetStorageShape().GetShapeSize();
      auto dt = context_->GetInputDesc(0)->GetDataType();
      if (dt == 1) {
        size_of_data_type = 2;
      }
  
      uint32_t ALIGN_NUM = BLOCK_SIZE / size_of_data_type;
      
      uint32_t tile_num;
  
      if (ub_block_num % 2 != 0) {
          ub_block_num = ub_block_num - 1;
      }
  
      // 获取reduction的值，并设置传入kernel的mode值
      const char* reduction = context_->GetAttrs()->GetStr(0);
      const char* mode1 = "mean";
      const char* mode2 = "sum";
      const char* mode3 = "none";
      size_t str_len = strlen(reduction);
      uint32_t mode = static_cast<int32_t>(0);
      
      if (str_len == strlen(mode1)) {
          for (size_t i = 0; i < str_len; i++) {
              if (reduction[i] != mode1[i]) {
                  break;
              }
              if (i == str_len-1) {
                  mode = 1;
              }
          }
      }
      if (str_len == strlen(mode2)) {
          for (size_t i = 0; i < str_len; i++) {
              if (reduction[i] != mode2[i]) {
                  break;
              }
              if (i == str_len-1) {
                  mode = 2;
              }
          }
      }
      if (str_len == strlen(mode3)) {
          for (size_t i = 0; i < str_len; i++) {
              if (reduction[i] != mode3[i]) {
                  break;
              }
              if (i == str_len-1) {
                  mode = 3;
              }
          }
      }
  
      tiling.set_mode(mode);
  
      if (totalLength % ALIGN_NUM != 0) {  
          totalLengthAligned =
              ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
      } else {
          totalLengthAligned = totalLength;
      }
  
      tiling.set_totalLength(totalLength);
  
      // 环境为单核环境，故直接设置为1个核
      context_->SetBlockDim(1);
  
      auto block_dim = context_->GetBlockDim();
      if (block_dim == 0) {
          throw std::runtime_error("Block dimension cannot be zero.");
      }

  
      blockLength = totalLengthAligned / block_dim;
      tile_num = blockLength / ALIGN_NUM / ub_block_num;
  
      // 数据切分策略： 由于为单核环境，则将tileLength设置得尽可能大，最后单独处理剩余数据
      if (ub_block_num != 0 && ((totalLengthAligned / block_dim / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0)) {  
          if (tile_num == 0) {
              tile_num = 1;
          } 
          if (blockLength < ub_block_num * ALIGN_NUM) {
              tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM;
              lastTileLength = tileLength;
          } 
          else {
              tileLength = ub_block_num * ALIGN_NUM;
              lastTileLength = tileLength;
          }
      } 
      else {  
          tile_num = tile_num + 1;
          tileLength = ub_block_num * ALIGN_NUM;
          lastTileLength = blockLength - (tile_num - 1) * tileLength;
      }
      return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling()
    {
      tiling.set_blockLength(blockLength);
      tiling.set_tileNum(tile_num);
      tiling.set_tileLength(tileLength);
      tiling.set_lastTileLength(lastTileLength);
  
      size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
      currentWorkspace[0] = static_cast<size_t>(0);
      tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                          context_->GetRawTilingData()->GetCapacity());
      context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
      context_->SetTilingKey(1);

      return ge::GRAPH_SUCCESS;
    }
  };
}

#endif // MSE_LOSS_TILING_H