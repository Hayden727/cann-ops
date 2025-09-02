/**
 * @file add_rms_norm_quant_v2.h
 * @brief Definition of the Ascend C NPU kernel for AddRmsNormQuantV2.
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @section DESCRIPTION
 * This operator implements a fused computation of "Add + RMS Norm + Bias Addition + Quantization".
 * The core computation formula is:
 * x_add = x1 + x2
 * x_norm = RMSNorm(x_add, gamma)
 * y1 = QuantizeToInt8(x_norm + bias)
 *
 * @tparam TX          The data type of inputs x1, x2, gamma, and bias (e.g., half, bfloat16_t).
 * @tparam TScale      The data type of the quantization scale (e.g., float, bfloat16_t).
 * @tparam TOffset     The data type of the quantization zero point (e.g., int32_t, bfloat16_t).
 */

#ifndef ADD_RMS_NORM_QUANT_V2_H_
#define ADD_RMS_NORM_QUANT_V2_H_

#include "rms_norm_base.h"
#include "kernel_operator.h" // Include for DumpTensor

using namespace AscendC;

template <typename TX, typename TScale, typename TOffset>
class KernelAddRmsNormQuantV2 {
public:
    __aicore__ inline KernelAddRmsNormQuantV2(TPipe *pipe) : Ppipe(pipe) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2,
        GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR bias, GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
        const AddRmsNormQuantV2TilingData *tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "Block dimension cannot be zero!");

        this->numRow = tilingData->numRow;
        this->numCol = tilingData->numCol;
        this->blockFactor = tilingData->blockFactor;
        this->rowFactor = tilingData->rowFactor;
        this->ubFactor = tilingData->ubFactor;
        this->epsilon = tilingData->epsilon;
        this->avgFactor = tilingData->avgFactor;
        this->hasZeroPoints1 = tilingData->hasZeroPoints1;
        this->hasBias = tilingData->hasBias;

        blockIdx_ = GetBlockIdx();
        uint32_t totalRows = numRow;
        uint32_t cores = GetBlockNum();
        this->rowWork = (totalRows + cores - 1) / cores;
        uint32_t startRow = blockIdx_ * this->rowWork;
        if (startRow >= totalRows) {
            this->rowWork = 0;
            return;
        }
        if (startRow + this->rowWork > totalRows) {
            this->rowWork = totalRows - startRow;
        }
        uint32_t gmOffset = startRow * numCol;

        x1Gm.SetGlobalBuffer((__gm__ TX *)x1 + gmOffset, rowWork * numCol);
        x2Gm.SetGlobalBuffer((__gm__ TX *)x2 + gmOffset, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ TX *)gamma, numCol);
        scales1Gm.SetGlobalBuffer((__gm__ TScale *)scales1, numCol);
        if (this->hasBias) {
            biasGm.SetGlobalBuffer((__gm__ TX *)bias, numCol);
        }
        if (this->hasZeroPoints1) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ TOffset *)zero_points1, numCol);
        }
        y1Gm.SetGlobalBuffer((__gm__ int8_t *)y1 + gmOffset, rowWork * numCol);
        xGm.SetGlobalBuffer((__gm__ TX *)x + gmOffset, rowWork * numCol);

        Ppipe->InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(TX));
        Ppipe->InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(TX));
        if (this->hasBias) {
             Ppipe->InitBuffer(inQueueBias, BUFFER_NUM, ubFactor * sizeof(TX));
        }
        Ppipe->InitBuffer(outQueueY1, BUFFER_NUM, ubFactor * sizeof(TX));

        Ppipe->InitBuffer(scales1Buf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(zeroPoints1Buf, ubFactor * sizeof(int32_t));
        if constexpr (IsSame<TX, half>::value || IsSame<TX, bfloat16_t>::value) {
            Ppipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(sqxBuf, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->rowWork == 0) return;

        CopyInSharedData();
        LocalTensor<TX> gammaLocal = inQueueGamma.DeQue<TX>();
        LocalTensor<TX> biasLocal;
        if (this->hasBias) {
            biasLocal = inQueueBias.DeQue<TX>();
        }

        uint32_t iOMax = CeilDiv(rowWork, rowFactor);
        uint32_t rowTail = rowWork % rowFactor == 0 ? rowFactor : rowWork % rowFactor;

        for (uint32_t iO = 0; iO < iOMax; iO++) {
            uint32_t calcRowNum = (iO == iOMax - 1) ? rowTail : rowFactor;
            SubProcess(iO, calcRowNum, gammaLocal, biasLocal);
        }

        inQueueGamma.FreeTensor(gammaLocal);
        if (this->hasBias) {
            inQueueBias.FreeTensor(biasLocal);
        }
    }

private:
    // [MODIFICATION] Pass loop counters iO and iI into SubProcess
    __aicore__ inline void SubProcess(uint32_t iO, uint32_t calcRowNum, LocalTensor<TX> &gammaLocal, LocalTensor<TX> &biasLocal)
    {
        for (uint32_t iI = 0; iI < calcRowNum; iI++) {
            uint32_t gmBias = (iO * rowFactor + iI) * numCol;
            CopyIn(gmBias);
            // [MODIFICATION] Pass loop counters to Compute
            Compute(gammaLocal, biasLocal, iO, iI);
            CopyOutY(gmBias);
        }
    }

    __aicore__ inline void CopyIn(uint32_t gmBias)
    {
        LocalTensor<TX> x1LocalIn = inQueueX.AllocTensor<TX>();
        LocalTensor<TX> x2Local = sqxBuf.Get<TX>();
        LocalTensor<TX> xLocal = outQueueY1.AllocTensor<TX>();

        DataCopyCustom<TX>(x1LocalIn, x1Gm[gmBias], numCol);
        DataCopyCustom<TX>(x2Local, x2Gm[gmBias], numCol);
        inQueueX.EnQue(x1LocalIn);
        auto x1Local = inQueueX.DeQue<TX>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
            Add(xLocal, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else if constexpr (IsSame<TX, bfloat16_t>::value) {
            LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
            LocalTensor<float> x2Fp32Local = sqxBuf.Get<float>();
            Cast(xFp32Local, x1Local, RoundMode::CAST_NONE, numCol);
            Cast(x2Fp32Local, x2Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, x2Fp32Local, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xLocal, xFp32Local, RoundMode::CAST_RINT, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            Add(x1Local, x1Local, x2Local, numCol);
            PipeBarrier<PIPE_V>();
            Adds(xLocal, x1Local, 0.0f, numCol);
        }
        inQueueX.FreeTensor(x1Local);

        outQueueY1.EnQue(xLocal);
        auto xOut = outQueueY1.DeQue<TX>();
        DataCopyCustom<TX>(xGm[gmBias], xOut, numCol);
        outQueueY1.FreeTensor(xOut);
    }

    __aicore__ inline void CopyInSharedData()
    {
        // ... (No changes in this function)
        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        if constexpr (IsSame<TScale, float>::value) {
            DataCopyCustom<float>(scales1Local, scales1Gm, numCol);
        } else { 
            LocalTensor<bfloat16_t> scales1Bf16 = scales1Buf.Get<bfloat16_t>()[ubFactor];
            DataCopyCustom<bfloat16_t>(scales1Bf16, scales1Gm, numCol);
            Cast(scales1Local, scales1Bf16, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        }

        if (hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            if constexpr (IsSame<TOffset, int32_t>::value) {
                LocalTensor<int32_t> zeroPoints1Int32 = zeroPoints1Buf.Get<int32_t>();
                DataCopyCustom<int32_t>(zeroPoints1Int32, zeroPoints1Gm, numCol);
                Cast(zeroPoints1Fp32, zeroPoints1Int32, RoundMode::CAST_NONE, numCol);
                PipeBarrier<PIPE_V>();
            } else { 
                LocalTensor<bfloat16_t> zeroPoints1Bf16 = zeroPoints1Buf.Get<bfloat16_t>()[ubFactor];
                DataCopyCustom<bfloat16_t>(zeroPoints1Bf16, zeroPoints1Gm, numCol);
                Cast(zeroPoints1Fp32, zeroPoints1Bf16, RoundMode::CAST_NONE, numCol);
                PipeBarrier<PIPE_V>();
            }
        }
        
        LocalTensor<TX> gammaLocal = inQueueGamma.AllocTensor<TX>();
        DataCopyCustom<TX>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);

        if (this->hasBias) {
            LocalTensor<TX> biasLocal = inQueueBias.AllocTensor<TX>();
            DataCopyCustom<TX>(biasLocal, biasGm, numCol);
            inQueueBias.EnQue(biasLocal);
        }
    }

    // [MODIFICATION] Add iO and iI parameters to the function definition
    __aicore__ inline void Compute(LocalTensor<TX> &gammaLocal, LocalTensor<TX> &biasLocal, uint32_t iO, uint32_t iI)
    {
        LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
        LocalTensor<float> sqx = sqxBuf.Get<float>();
        LocalTensor<float> reduceLocal = reduceFp32Buf.Get<float>();

        Mul(sqx, xFp32Local, xFp32Local, numCol);
        PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avgFactor, numCol);
        PipeBarrier<PIPE_V>();
        ReduceSumCustom(sqx, sqx, reduceLocal, numCol);
        PipeBarrier<PIPE_V>();

        Adds(sqx, sqx, epsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(sqx, sqx, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(reduceLocal, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(sqx, reduceLocal, sqx, 1);
        PipeBarrier<PIPE_V>();
        
        float rstdValue = sqx.GetValue(0);

        Muls(xFp32Local, xFp32Local, rstdValue, numCol);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSame<TX, half>::value) {
            LocalTensor<half> xFp16Cast = sqxBuf.Get<half>();
            Cast(xFp16Cast, xFp32Local, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp16Cast, gammaLocal, xFp16Cast, numCol);
            PipeBarrier<PIPE_V>();
            Cast(xFp32Local, xFp16Cast, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<float> gammaFp32 = sqxBuf.Get<float>();
            Cast(gammaFp32, gammaLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Mul(xFp32Local, xFp32Local, gammaFp32, numCol);
            PipeBarrier<PIPE_V>();
        }
        
        if (this->hasBias) {
            LocalTensor<float> biasFp32 = sqxBuf.Get<float>();
            Cast(biasFp32, biasLocal, RoundMode::CAST_NONE, numCol);
            PipeBarrier<PIPE_V>();
            Add(xFp32Local, xFp32Local, biasFp32, numCol);
            PipeBarrier<PIPE_V>();
        }

        // [DUMP TENSOR with CONDITION and LENGTH CONTROL]
        // Only dump data for the first core (blockIdx_ == 0) and its first-ever row (iO == 0 and iI == 0)
        if (blockIdx_ == 0 && iO == 0 && iI == 0) {
            // Define a smaller, controlled length for dumping.
            constexpr uint32_t DUMP_LENGTH = 32;
            const uint32_t actualDumpLen = numCol > DUMP_LENGTH ? DUMP_LENGTH : numCol;

            // Dump the final RMS Norm result (after multiplication with gamma).
            AscendC::DumpTensor(xFp32Local, 1, actualDumpLen);

            // Dump the result after bias has been added, right before quantization starts.
            AscendC::DumpTensor(xFp32Local, 2, actualDumpLen);
        }

        LocalTensor<float> scales1Local = scales1Buf.Get<float>();
        Div(xFp32Local, xFp32Local, scales1Local, numCol);
        PipeBarrier<PIPE_V>();

        if (this->hasZeroPoints1) {
            LocalTensor<float> zeroPoints1Fp32 = zeroPoints1Buf.Get<float>();
            Add(xFp32Local, xFp32Local, zeroPoints1Fp32, numCol);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Local = outQueueY1.AllocTensor<int8_t>();
        RoundFloat2Int8(y1Local, xFp32Local, numCol);
        outQueueY1.EnQue<int8_t>(y1Local);
    }

    __aicore__ inline void CopyOutY(uint32_t progress)
    {
        LocalTensor<int8_t> yLocal = outQueueY1.DeQue<int8_t>();
        DataCopyCustom<int8_t>(y1Gm[progress], yLocal, numCol);
        outQueueY1.FreeTensor(yLocal);
    }

private:
    TPipe *Ppipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma, inQueueBias;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY1;

    TBuf<TPosition::VECCALC> scales1Buf, zeroPoints1Buf, xFp32Buf, sqxBuf, reduceFp32Buf;

    GlobalTensor<TX> x1Gm, x2Gm, gammaGm, biasGm, xGm;
    GlobalTensor<TScale> scales1Gm;
    GlobalTensor<TOffset> zeroPoints1Gm;
    GlobalTensor<int8_t> y1Gm;

    uint32_t numRow, numCol, blockFactor, rowFactor, ubFactor;
    float epsilon, avgFactor;
    uint32_t hasZeroPoints1;
    bool hasBias;
    int32_t blockIdx_;
    uint32_t rowWork;
};

#endif // ADD_RMS_NORM_QUANT_V2_H_