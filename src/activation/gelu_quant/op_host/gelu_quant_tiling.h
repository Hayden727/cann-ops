/* *
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

/* !
 * \file gelu_quant_tiling.h
 * \brief
 */

#ifndef GELU_QUANT_TILING_H
#define GELU_QUANT_TILING_H

#include "gelu_quant_tiling_def.h"

namespace optiling {
struct GeluQuantBaseInfoParams {
    // platformInfo
    int64_t vectorCoreNum{ 0 };
    uint64_t ubSize{ 0 };

    // shapeInfo
    int64_t xDimNum{ 0 };
    int64_t endAxisLen{ 0 };
    int64_t endAxisLenAligned{ 0 };
    int64_t fusedFrontAxis{ 1 };
    int64_t fusedAllAxis{ 1 };
    int64_t elementNumAlign{ 0 };

    // dtype
    ge::DataType xInputDtype{ ge::DT_FLOAT };
    ge::DataType scaleInputDtype{ ge::DT_FLOAT };
    ge::DataType offsetInputDtype{ ge::DT_FLOAT };

    // optional
    int64_t inputScaleType{ 0 };
    int64_t inputOffsetType{ 0 };

    // attr
    int64_t quantMode{ 0 };
    int64_t approximate{ 0 };
};

struct GeluQuantSplitCoreParams {
    int64_t normalCoreProcessNum{ 0 };
    int64_t usedCoreNum{ 0 };
    int64_t tailCoreProcessNum{ 0 };

    int64_t coexistentNodeNum{ 0 };
    int64_t coexistentNodeElementNum{ 0 };

    int64_t templateMode{ 0 };

    int64_t rowInner{ 1 };
    int64_t rowOuter{ 1 };
    int64_t rowTail{ 1 };
    int64_t colInner{ 1 };
    int64_t colOuter{ 1 };
    int64_t colTail{ 1 };
    int64_t tilingKey{ 0 };
};

class GeluQuantTiling {
public:
    explicit GeluQuantTiling(gert::TilingContext *context) : context_(context), nodeName_(context->GetNodeName()) {}
    ~GeluQuantTiling() {}
    GeluQuantTilingData tilingData;
    ge::graphStatus RunGeluQuantTiling();
    ge::graphStatus GetInputInfo();
    ge::graphStatus ProcessAttrsInfo();
    ge::graphStatus ProcessRequiredInfo();
    ge::graphStatus ProcessOptionalScaleInfo();
    ge::graphStatus ProcessOptionalOffsetInfo();
    ge::graphStatus DoTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoStaticQuantTiling();
    ge::graphStatus DoStaticQuantPerTensorTiling();
    ge::graphStatus DoStaticQuantFullKernelSmallEndAxis();
    ge::graphStatus DoStaticQuantNotFullKernelSplitEndAxis();
    ge::graphStatus DoDynamicQuantTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo() const;
    uint64_t GetTilingKey() const;
    void SaveToTilingData();

protected:
    gert::TilingContext *context_ = nullptr;
    const ge::char_t *nodeName_;

private:
    GeluQuantBaseInfoParams baseInfoOp;
    GeluQuantSplitCoreParams splitCoreOp;
};

struct GeluQuantCompileInfo {
};


} // namespace optiling
#endif // GELU_QUANT_TILING_H
