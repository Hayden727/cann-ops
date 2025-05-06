/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file conv2d_transpose_v2_tiling.cc
 * \brief
 */
#include "conv2d_transpose_v2_tiling.h"
#include "op_log.h"
#include "tiling/tiling_templates_registry.h"

namespace {
using Conv2DTransposeV2CompileInfo = optiling::Conv3DBackPropInputCompileInfo;
}

namespace optiling {
// TODO : 3D TILING
REGISTER_TILING_TEMPLATE("Conv2DTransposeV2", Conv2DTransposeV2Tiling, 0);

static ge::graphStatus Conv2DTransposeV2TilingFunc(gert::TilingContext *context)
{
   return AdaptTilingToConv3DBp(context, "Conv2DTransposeV2");
}

static ge::graphStatus TilingParseForConv2DBackpropInputV2(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto compileInfoPtr = context->GetCompiledInfo<Conv2DTransposeV2CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfo is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfoPtr);
    compileInfoPtr->core_num = ascendcPlatform.GetCoreNumAic();
    optiling::PlatformInfo &plaformInstance = optiling::PlatformInfo::GetInstance();
    plaformInstance.SetInstance(*compileInfoPtr);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv2DTransposeV2)
    .Tiling(Conv2DTransposeV2TilingFunc)
    .TilingParse<Conv2DTransposeV2CompileInfo>(TilingParseForConv2DBackpropInputV2);
}