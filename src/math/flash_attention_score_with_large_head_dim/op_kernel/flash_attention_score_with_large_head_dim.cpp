/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "flash_attention_score_with_large_head_dim_s1s2_bn2gs1.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void flash_attention_score_with_large_head_dim(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR attention_out, GM_ADDR workspace, GM_ADDR tiling) {
    
    TPipe tPipe;
    set_mask_norm();    
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreWithLargeHeadDimTilingData, tilingDataIn, tiling);
    const FlashAttentionScoreWithLargeHeadDimTilingData *__restrict tilingData = &tilingDataIn;
    const TCubeTiling *__restrict bmm1tiling = &(tilingData->bmm1TilingData);
    const TCubeTiling *__restrict bmm2tiling = &(tilingData->bmm2TilingData);
    FlashAttentionScoreWithLargeHeadDimS1s2Bn2gs1 op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);
    op.Init(query, key, value, softmax_max, softmax_sum, attention_out, user, tilingData, &tPipe);
    op.Process();
}