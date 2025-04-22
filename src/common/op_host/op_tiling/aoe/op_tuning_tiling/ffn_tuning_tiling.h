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

#ifndef FFN_TUNING_TILING_H
#define FFN_TUNING_TILING_H

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
constexpr uint32_t MAX_EXPERT_NUM_FFN_TILING = 256;
#pragma pack(push, 1)  // the compiler uses 1-byte alignment for structure byte alignment

// configure FFN query input structure in knowledge base repo
struct FFNInputArgs {
    uint32_t totalTokensRepoFFN;  // the total number of tokens
    uint32_t nRepoFFN;  // the number of columns N that mm1 outputs
    uint32_t hRepoFFN;  // the number of columns H of input x
    uint32_t expertNumRepoFFN;  // the number of experts
    uint32_t maxTokensRepoFFN;  // the number of Tokens inside the largest expert
    int32_t tokensArrRepoFFN[MAX_EXPERT_NUM_FFN_TILING];  // the distributed array of Tokens number for each expert
};

#pragma pack(pop)  // The compiler restores the previously saved byte alignment and pops the alignment from the stack

// define the MatMul1TuningTiling class, which inherits from TilingDef
BEGIN_TUNING_TILING_DEF(MatmulTunnerTiling)
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, batchDim);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseM);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseN);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepM);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepN);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKa);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKb);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbAL1Flag);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, dbBL1Flag);
    TUNING_TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(MatmulTunnerTiling, FIELD(MatmulTunnerTiling, batchDim), FIELD(MatmulTunnerTiling, mDim),
               FIELD(MatmulTunnerTiling, nDim), FIELD(MatmulTunnerTiling, kDim), FIELD(MatmulTunnerTiling, baseM),
               FIELD(MatmulTunnerTiling, baseN), FIELD(MatmulTunnerTiling, baseK), FIELD(MatmulTunnerTiling, stepM),
               FIELD(MatmulTunnerTiling, stepN), FIELD(MatmulTunnerTiling, stepKa), FIELD(MatmulTunnerTiling, stepKb),
               FIELD(MatmulTunnerTiling, dbAL1Flag), FIELD(MatmulTunnerTiling, dbBL1Flag),
               FIELD(MatmulTunnerTiling, iterateOrder));

// define the FfnTunnerTiling class, which inherits from TilingDef
BEGIN_TUNING_TILING_DEF(FfnTunnerTiling)
    TUNING_TILING_DATA_FIELD_DEF(MatmulTunnerTiling, mm1TilingSpace);
    TUNING_TILING_DATA_FIELD_DEF(MatmulTunnerTiling, mm2TilingSpace);
END_TUNING_TILING_DEF

// declares FfnTunnerTiling struct schema information and struct field information
DECLARE_SCHEMA(FfnTunnerTiling,
    FIELD(FfnTunnerTiling, mm1TilingSpace),
    FIELD(FfnTunnerTiling, mm2TilingSpace));

// define a shared pointer to an FfnTunnerTiling object
using FfnTunnerTilingPtr = std::shared_ptr<FfnTunnerTiling>;

}  // namespace tuningtiling
#endif
