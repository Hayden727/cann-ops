/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "trunc_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_kl.h"
using kunlun::tiling::SimpleTilingStrategy;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_SIZE = 256;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TruncTilingData tiling; // TilingData

    constexpr bool DOUBLE_BUFFER_ENABLE = true; // 是否启用 DoubleBuffer
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()); // Ascend 平台信息
    uint64_t ubSize; // UB容量
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t CORE_NUM = 1; // 获取设备的核心数
    
     // 简易核间&核内切分工具
    SimpleTilingStrategy tilingStrategy(context, platform_ascendc::CoreMemType::UB);
    {
        tilingStrategy.SetBlockDim(CORE_NUM); // 设置核心数
         // 配置IO比重
        auto& inputList = tilingStrategy.inputs;
        auto& outputList = tilingStrategy.outputs;
        inputList[0].lengthWeight = 1; // input_x
        switch(context->GetInputDesc(0)->GetDataType()){  // 跟踪数据类型
            case ge::DataType::DT_BF16:{
                tilingStrategy.addCalc(2, 2);
                outputList[0].lengthWeight = 1; // output_y
                break;
            }
            case ge::DataType::DT_FLOAT:{
                outputList.erase(outputList.begin()); // output_y
                break;
            }
            default:{
                outputList[0].lengthWeight = 1; // output_y
            }
        }
         // 核间&核内一键切分
        tilingStrategy.reTiling<DOUBLE_BUFFER_ENABLE>();
         // 根据切分结果推测IO详情
        tilingStrategy.autoInferQueueDetail();
    }
    const auto& coreDetail = tilingStrategy.formerCore;
    {
         // 获取&设置 切分信息
        const auto& coreDetail = tilingStrategy.formerCore;
         // 核内切分数据
        tiling.set_Len(coreDetail.batchPartitionLength); // length
        tiling.set_fLen(coreDetail.formerTilePartitionLength); // former_tile-length <=> ub-partion-length
        tiling.set_fNum(coreDetail.formerTileNum); // former_tile-num
        tiling.set_tLen(coreDetail.tailTilePartitionLength); // tail_tile-length
    }
     // 保存Tiling数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

     // 设置任务核心数
    context->SetBlockDim(1);
    constexpr size_t userWorkspaceSize = 0;
    context->GetWorkspaceSizes(1)[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

}

namespace ops {
class Trunc : public OpDef {
public:
    explicit Trunc(const char* name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_INT8,ge::DT_INT32,ge::DT_UINT8 })
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });
        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_INT8,ge::DT_INT32,ge::DT_UINT8 })
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND })
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND });

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b")
            .AddConfig("ascend310b");
    }
};

OP_ADD(Trunc);
}

