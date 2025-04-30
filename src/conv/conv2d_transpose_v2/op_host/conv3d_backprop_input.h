/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file conv3d_backprop_input.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CONV3D_BACKPROP_INPUT_H_
#define OPS_BUILT_IN_OP_TILING_CONV3D_BACKPROP_INPUT_H_
#include <cstdint>

#include "cube/algorithm/hash/tiling_cache.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/util/cube_util.h"
#include "cube/util/timer.h"
#include "cube_tiling_runtime.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace optiling {

const size_t kConv2dDimNumLimit = 4;
const int32_t kConv2dNDim = 0;
const int32_t kConv2dHDim = 2;
const int32_t kConv2dWDim = 3;

ge::graphStatus TilingForConv3DDx(gert::TilingContext *context, cachetiling::OpType op_type);
class Conv3DDxParas {
public:
    explicit Conv3DDxParas(cachetiling::OpType type) : tiling_param(type) {}

    int32_t multiple_extend = 0;
    int64_t filter_gdkci1ghw = 0;
    int64_t fmap_d_padding = 0;
    int64_t fmap_h_padding = 0;
    int64_t fmap_w_padding = 0;
    int32_t shape_up_modify = 0;
    int32_t shape_left_modify = 0;
    int32_t shape_down_modify = 0;
    int32_t shape_right_modify = 0;
    int32_t pad_head_before = 0;
    int32_t pad_up_before = 0;
    int32_t pad_left_before = 0;
    int32_t pad_tail_after = 0;
    int32_t pad_down_after = 0;
    int32_t pad_right_after = 0;
    int32_t output_padding_d = 0;
    int32_t output_padding_h = 0;
    int32_t output_padding_w = 0;
    bool repo_binary_flag = false;
    cachetiling::Conv3DBpInputTilingParam tiling_param;
};

struct RunInfoPara {
    // shape vars
    int32_t batch_n;
    int32_t dedy_cout;
    int32_t dedy_d;
    int32_t dedy_h;
    int32_t dedy_w;
    int32_t dedx_cin;
    int32_t dedx_d;
    int32_t dedx_h;
    int32_t dedx_w;
    int32_t kernel_d;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t dedy_cout1;
    int32_t dedx_cin1;
    int32_t real_g;
    int32_t dedy_cout1_g;
    int32_t dedx_cin1_g;
    int32_t kernel_g_dk_cin1g_hk_wk;
    // attr vars
    int32_t stride_d;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_t;
    int32_t pad_u;
    int32_t pad_d;
    int32_t pad_l;
    int32_t pad_r;
    int32_t dilation_d;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t shape_up_modify;
    int32_t shape_left_modify;
    int32_t shape_down_modify;
    int32_t shape_right_modify;
    int32_t backprop_pad_h;
    int32_t backprop_pad_t;
    int32_t backprop_pad_u;
    int32_t backprop_pad_d;
    int32_t backprop_pad_l;
    int32_t backprop_pad_r;
    // tiling vars
    int32_t batch_dim;
    int32_t n_dim;
    int32_t m_dim;
    int32_t group_dim;
    int32_t d_dim;
    int32_t m_al1;
    int32_t n_bl1;
    int32_t m_l0;
    int32_t n_l0_div_ub;
    int32_t n_cub;
    int32_t k_l0;
    int32_t min_kl1_div_kl0;
    int32_t max_kl1_div_min_kl1;
    int32_t k_div_max_kl1;
    int32_t d_al1;
    int32_t d_bl1;
    int32_t d_al0;
    int32_t d_bl0;
    int32_t d_cl0;
    int32_t k_aub;
    int32_t m_aub;
    int32_t wo_aub;
    int32_t al1_bound;
    int32_t bl1_bound;
    int32_t aub_bound;

    int32_t load3d_special;
    int32_t hf32_flag;
};
bool Conv3DBackpropInputParseFunc(gert::TilingContext *context, cachetiling::OpType op_type,
                                  Conv3DDxParas &conv3ddx_paras, bool isV2Impl = false);

bool SetRunInfoConv3DDx(const Conv3DDxParas &conv3ddx_paras, const cachetiling::Conv3DBpInputTiling &tiling,
                        RunInfoPara &run, gert::TilingContext *context);
bool CheckParams(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context);
bool CheckCalPads(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CONV3D_BACKPROP_INPUT_H_
