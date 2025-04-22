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
 * \file conv3d_backprop_filter_v2.cpp
 * \brief Conv3DBackpropFilter ascendc impl
 */

#include "register/op_def_registry.h"
#include "cube_util.h" 
// #include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
// #include "op_log.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kConv3dDimSizeLimit = 5;

// Conv3DTranspose
constexpr size_t kConv3DTransposeFilterIdx = 2;
constexpr size_t kConv3DTransposePadsIdx = 1;
constexpr size_t kConv3DTransposeGroupsIdx = 3;
constexpr size_t kConv3DTransposeOutputPaddingIdx = 5;
constexpr size_t kConv3DTransposePaddingIdx = 7;

// Conv2DTranspose
constexpr size_t kConv2DTransposeFmapIdx = 1;
constexpr size_t kConv2DTransposeFilterIdx = 2;
constexpr size_t kConv2DTransposePadsIdx = 1;
constexpr size_t kConv2DTransposeGroupsIdx = 3;
constexpr size_t kConv2DTransposeOutputPaddingIdx = 5;
constexpr size_t kConv2DTransposePaddingIdx = 7;
constexpr size_t kConv2DTransposeAutoPadIdx = 8;

// Deconvolution
constexpr size_t kDeconvolutionFmapIdx = 0;
constexpr size_t kDeconvolutionFilterIdx = 1;
constexpr size_t kDeconvolutionPadsIdx = 1;
constexpr size_t kDeconvolutionGroupsIdx = 3;
constexpr size_t kDeconvolutionPaddingIdx = 6;
constexpr size_t kDeconvolutionAutoPadIdx = 7;

using gert::InferShapeContext;
using ge::Format;
using ge::FORMAT_NCDHW;
using ge::FORMAT_NDHWC;
using ge::FORMAT_NCHW;
using ge::FORMAT_NHWC;
using ge::FORMAT_HWCN;
using ge::GRAPH_FAILED;
using ge::graphStatus;
}  // namespace
namespace ge {

  
static graphStatus InferShapeForConvBackprop(InferShapeContext *context, size_t const_tensor_idx,
                                             const char *const_tensor_name, size_t dim_num) {
//   OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"),
    // return GRAPH_FAILED);
  const auto op_name = context->GetNodeName();
  auto y_shape = context->GetOutputShape(0);
//   OP_CHECK(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "y shape"),
    // return GRAPH_FAILED);

  auto const_tensor = context->GetInputTensor(const_tensor_idx);
//   OP_CHECK(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor", const_tensor_name),
        // return GRAPH_FAILED);
  size_t const_tensor_dim_num = static_cast<size_t>(const_tensor->GetOriginShape().GetShapeSize());
//   OP_CHECK(const_tensor_dim_num != dim_num,
        // CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid", const_tensor_name, const_tensor_dim_num),
        // return GRAPH_FAILED);
  y_shape->SetDimNum(dim_num);

  auto dtype = const_tensor->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = const_tensor->GetData<int32_t>();
    // OP_CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
    //       return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = const_tensor->GetData<int64_t>();
    // OP_CHECK(tensor_data == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor data", const_tensor_name),
    //       return GRAPH_FAILED);
    for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
      y_shape->SetDim(idx, tensor_data[idx]);
    }
  } else {
    // CUBE_INNER_ERR_REPORT(op_name, "tensor %s not support dtype %s", const_tensor_name,
                        //   ge::TypeUtils::DataTypeToAscendString(dtype).GetString());
    return GRAPH_FAILED;
  }

//   OP_LOGD(context->GetNodeName(), "y_shape: %s", ge::Shape2String(*y_shape).c_str());
  return ge::GRAPH_SUCCESS;
}

static graphStatus InferShapeForConv3DBackpropFilter(InferShapeContext *context) {
  return InferShapeForConvBackprop(context, 1, "filter_size", kConv3dDimSizeLimit);
}

static ge::graphStatus InferDataTypeForConv3DBackpropFilterV2(gert::InferDataTypeContext *context) {
//   OP_LOGD(context->GetNodeName(), "InferDataTypeForConv3DBackpropFilterV2 enter");
  ge::graphStatus ret = context->SetOutputDataType(0, ge::DT_FLOAT);
//   OP_CHECK(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "[InferDataType] Failed."), 
                                                                    // return ge::GRAPH_FAILED);
//   OP_LOGD(context->GetNodeName(), "InferDataTypeForConv3DBackpropFilterV2 enter");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Conv3DBackpropFilterV2)
    .InferShape(InferShapeForConv3DBackpropFilter)
    .InferDataType(InferDataTypeForConv3DBackpropFilterV2)
    .InputsDataDependency({1})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", 0L);
}  // namespace ge

namespace ops {
class Conv3DBackpropFilterV2 : public OpDef {
public:
    explicit Conv3DBackpropFilterV2(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});
        this->Input("filter_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("out_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "conv3d_backprop_filter_v2")
            .ExtendCfgInfo("opInterface.value", "conv3d_backprop_filter_v2")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(Conv3DBackpropFilterV2);
}  // namespace ops