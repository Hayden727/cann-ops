/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rms_norm_grad_split_n_high_precision.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_
#define RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_
#include "rms_norm_grad_common.h"
template <typename T_DY, typename T_GAMMA>
class RmsNormGradSplitNHighPrecision {
public:
    __aicore__ inline RmsNormGradSplitNHighPrecision()
    {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData *tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
        InitInputGmBuffer(dy, x, rstd, gamma, blockDim_, coreCalcTail_);
        InitOutputGmBuffer(dx, dgamma);
        InitInputQue();
        InitOutputQue();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        InitTmpBuffer();
        if (isDeterministic_ == 1) {
            InitWorkspace(usrWorkspace);
        } else {
            workspaceMiddleDgammaGm_.SetGlobalBuffer(
                (__gm__ float *)usrWorkspace + GetBlockIdx() * saveLineNum * colValAlign_, saveLine * colValAlign_);
            SyncAll();
        }
#else
        syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, ALIGN_32 * GetBlockNum());
        uint32_t syncLen = ALIGN_32 * GetBlockNum();
        pipe.InitBuffer(outZeroTmpBuf_, colValAlign_ * sizeof(float));
        pipe.InitBuffer(syncTmpBuf_, syncLen * sizeof(int32_t));

        InitGmZero<int32_t>(syncTmpSpaceGm_, outZeroTmpBuf_, syncLen, (uint32_t)0);
        if (isDeterministic_ != 1) {
            if (GetBlockIdx() == 0) {
                InitGmZero<float>(dgammaGm_, outZeroTmpBuf_, colValAlign_, (uint32_t)0);
            }
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
        } else {
            workspaceGm_.SetGlobalBuffer(
                (__gm__ float *)usrWorkspace + ALIGN_32 * GetBlockNum() + GetBlockIdx() * colVal_);
            if (GetBlockIdx() == 0) {
                InitGmZero<float>(workspaceGm_, outZeroTmpBuf_, colValAlign_, (uint32_t)0);
            }
        }
#endif
    }

    __aicore__ inline void InitWorkspace(GM_ADDR usrWorkspace)
    {
        // 中间数据的workspace大小
        workspaceMiddleDgammaGm_.SetGlobalBuffer(
            (__gm__ float *)usrWorkspace + GetBlockIdx() * saveLineNum * colValAlign_, saveLine * colValAlign_);
    }

    __aicore__ inline void InitVar(const RmsNormGradTilingData *tiling)
    {
        blockDim_ = tiling->block_dim;
        rowVal_ = tiling->row;
        colVal_ = tiling->col;
        avgFactor_ = tiling->avg_factor;
        dataType_ = tiling->data_type;
        coreCalcNum_ = tiling->core_calc_num;
        coreCalcTail_ = tiling->core_calc_tail;
        blockFactor_ = tiling->block_factor;
        ubFactor_ = tiling->ub_factor;
        ubCalcNum = tiling->ub_calc_num;
        ubCalcTail_ = tiling->ub_calc_tail;
        ubCalcLoop_ = tiling->ub_calc_loop;
        ubCalcTailNum_ = tiling->ub_calc_tail_num;
        ubCalcTailTail_ = tiling->ub_calc_tail_tail;
        ubCalcTailLoop_ = tiling->ub_calc_tail_loop;
        alignLen_ = dataType_ == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;
        colValAlign_ = (colVal_ + alignLen_ - 1) / alignLen_ * alignLen_;
        isDeterministic_ = tiling->fixed_output;
        if (coreCalcTail_ != 0 && GetBlockIdx() == blockDim_ - 1) {
            totalLine = coreCalcTail_;
        } else {
            totalLine = coreCalcNum_;
        }

        if (colValAlign_ > SMALLD_THRESHOLD) {
            saveLineNum = (coreCalcNum_ % 2 == 0) ? coreCalcNum_ / 2 : ((coreCalcNum_ / 2) + 1);
            saveLineTailNum = (coreCalcTail_ % 2 == 0) ? coreCalcTail_ / 2 : ((coreCalcTail_ / 2) + 1);
            if (coreCalcTail_ != 0 && GetBlockIdx() == blockDim_ - 1) {
                saveLine = saveLineTailNum;
            } else {
                saveLine = saveLineNum;
            }
        } else {
            saveLineNum = coreCalcNum_;
            saveLine = totalLine;
        }
        saveLine = saveLine > 0 ? saveLine : 1;
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, uint32_t blockDim, uint32_t coreCalcTail)
    {
        if (GetBlockIdx() < blockDim - 1) {
            coreOffset_ = blockFactor_;
        } else {
            coreOffset_ = coreCalcTail > 0 ? coreCalcTail : blockFactor_;
        }
        dyGm_.SetGlobalBuffer((__gm__ T_DY *)dy + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
        xGm_.SetGlobalBuffer((__gm__ T_DY *)x + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
        rstdGm_.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * blockFactor_, coreOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA *)gamma, colVal_);
    }

    __aicore__ inline void InitOutputGmBuffer(GM_ADDR dx, GM_ADDR dgamma)
    {
        dxGm_.SetGlobalBuffer((__gm__ T_DY *)dx + GetBlockIdx() * blockFactor_ * colVal_, coreOffset_ * colVal_);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        dgammaGm_.SetGlobalBuffer((__gm__ float *)dgamma, colValAlign_);
#else
        dgammaGm_.SetGlobalBuffer((__gm__ float *)dgamma, colVal_);
        if (isDeterministic_ == 1) {
            return;
        } else {
            if (GetBlockIdx() == 0) {
                InitOutput<float>(dgammaGm_, colVal_, 0);
            }
        }
#endif
    }

    __aicore__ inline void InitInputQue()
    {
        ubFactorAlign_ = ubFactor_ * colValAlign_;
        rstdLen_ = (ubFactor_ + alignLen_ - 1) / alignLen_ * alignLen_;
        bufferLenSize_ = ubFactorAlign_ * sizeof(float);
        bufferNum_ = BUFFER_NUM_DB;
        pipe.InitBuffer(inQueDY_, bufferNum_, bufferLenSize_);
        pipe.InitBuffer(inQueX_, bufferNum_, bufferLenSize_);
        pipe.InitBuffer(inQueRstd_, bufferNum_, rstdLen_ * sizeof(float));
        pipe.InitBuffer(inQueGamma_, 1, colValAlign_ * sizeof(float));
    }

    __aicore__ inline void InitOutputQue()
    {
        pipe.InitBuffer(outQueDX_, bufferNum_, bufferLenSize_);
    }

    __aicore__ inline void InitTmpBuffer()
    {
        if (colValAlign_ <= SMALLD_THRESHOLD) {
            pipe.InitBuffer(tmpBuf_, ubFactor_ * ELEM_PER_REP_FP32 * sizeof(float));
            pipe.InitBuffer(tmpMeanBuf_, ubFactor_ * sizeof(float));
            pipe.InitBuffer(outQueDgamma_, 1, bufferLenSize_);
        } else {
            pipe.InitBuffer(outQueDgamma_, 1, colValAlign_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        CopyGammaIn();
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, colValAlign_, colValAlign_);
        PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        if (colValAlign_ > SMALLD_THRESHOLD) {
#endif
            for (uint32_t i = 0; i < totalLine; i++) {
                LocalTensor<float> dgammaLocal = outQueDgamma_.AllocTensor<float>();
                Duplicate(dgammaLocal, 0.0f, colValAlign_);
                PipeBarrier<PIPE_V>();
                CopyIn(i, 1);
                Compute(gammaLocal, dgammaLocal);
                CopyOut(i, 1);
                PipeBarrier<PIPE_V>();
                CopyDgammaMiddleOutWorkspace(i, 1, 1, dgammaLocal, totalLine);
                outQueDgamma_.FreeTensor(dgammaLocal);
            }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        } else {
            uint32_t calcLen = ubCalcNum;
            uint32_t calcTailLen = ubCalcTail_;
            uint32_t calcLoop = (ubCalcTail_ == 0 ? ubCalcLoop_ : ubCalcLoop_ - 1);

            if (GetBlockIdx() == blockDim_ - 1 && coreCalcTail_ > 0) {
                calcLen = ubCalcTailNum_;
                calcTailLen = ubCalcTailTail_;
                calcLoop = (ubCalcTailTail_ == 0 ? ubCalcTailLoop_ : ubCalcTailLoop_ - 1);
            }

            for (uint32_t i = 0; i < calcLoop; i++) {
                LocalTensor<float> dgammaLocal = outQueDgamma_.AllocTensor<float>();
                Duplicate(dgammaLocal, 0.0f, calcLen * colValAlign_);
                PipeBarrier<PIPE_V>();
                CopyIn(i * ubFactor_, calcLen);
                ComputeSmallD(i, calcLen, gammaLocal, dgammaLocal);
                CopyOut(i * ubFactor_, calcLen);
                CopyDgammaMiddleOutWorkspace(i * ubFactor_, 2, calcLen, dgammaLocal, totalLine);
                outQueDgamma_.FreeTensor(dgammaLocal);
            }

            if (calcTailLen > 0) {
                LocalTensor<float> dgammaLocal = outQueDgamma_.AllocTensor<float>();
                Duplicate(dgammaLocal, 0.0f, calcTailLen * colValAlign_);
                PipeBarrier<PIPE_V>();
                CopyIn(calcLoop * ubFactor_, calcTailLen);
                ComputeSmallD(calcLoop, calcTailLen, gammaLocal, dgammaLocal);
                CopyOut(calcLoop * ubFactor_, calcTailLen);
                CopyDgammaMiddleOutWorkspace(calcLoop * ubFactor_, 2, calcTailLen, dgammaLocal, totalLine);
                outQueDgamma_.FreeTensor(dgammaLocal);
            }
        }
#endif
        inQueGamma_.FreeTensor(gammaLocal);
        // 销毁已申请的所有UB，重新申请分配
        pipe.Destroy();
        InitMiddleQue();
        // 二分进行核内累加
        computeMiddleDgamma();
    }

    __aicore__ inline void DoDGamma()
    {
        if (isDeterministic_ == 1) {
            CopyDgammaOutWorkspace();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
#else
            SyncAll();
#endif
            AddDgamma();
        } else {
            CopyDgammaOut();
        }
    }

    __aicore__ inline void CopyGammaIn()
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueGamma_.AllocTensor<T_GAMMA>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(colVal_ * sizeof(T_GAMMA)), 0, 0, 0};
        DataCopyPadExtParams<T_GAMMA> padParams{true, 0, 0, 0};
        if constexpr (!IsSame<T_GAMMA, float>::value) {
            DataCopyPad(gammaLocal[colValAlign_], gammaGm_, dataCopyParams, padParams);
        } else {
            DataCopyPad(gammaLocal, gammaGm_, dataCopyParams, padParams);
        }
#else
        if constexpr (!IsSame<T_GAMMA, float>::value) {
            DataCopy(gammaLocal[colValAlign_], gammaGm_, colValAlign_);
        } else {
            DataCopy(gammaLocal, gammaGm_, colValAlign_);
        }
#endif
        inQueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyIn(uint32_t rowIdx, uint32_t calcLen)
    {
        LocalTensor<float> rstd = inQueRstd_.AllocTensor<float>();
        LocalTensor<T_DY> xLocal = inQueX_.AllocTensor<T_DY>();
        LocalTensor<T_DY> dy = inQueDY_.AllocTensor<T_DY>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsRstd{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParamsRstd{true, 0, 0, 0};

        DataCopyExtParams dataCopyParams{(uint16_t)calcLen, (uint32_t)(colVal_ * sizeof(T_DY)), 0, 0, 0};
        DataCopyPadExtParams<T_DY> padParams{true, 0, 0, 0};
        DataCopyPad(rstd, rstdGm_[rowIdx], dataCopyParamsRstd, padParamsRstd);
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopyPad(xLocal[calcLen * colValAlign_], xGm_[rowIdx * colVal_], dataCopyParams, padParams);
        } else {
            DataCopyPad(xLocal, xGm_[rowIdx * colVal_], dataCopyParams, padParams);
        }
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopyPad(dy[calcLen * colValAlign_], dyGm_[rowIdx * colVal_], dataCopyParams, padParams);
        } else {
            DataCopyPad(dy, dyGm_[rowIdx * colVal_], dataCopyParams, padParams);
        }
#else
        uint32_t calcLenAlign = ROUND_UP(calcLen, alignLen_);
        DataCopy(rstd, rstdGm_[rowIdx], calcLenAlign);
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopy(xLocal[calcLen * colValAlign_], xGm_[rowIdx * colVal_], colValAlign_);
        } else {
            DataCopy(xLocal, xGm_[rowIdx * colVal_], colValAlign_);
        }
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopy(dy[calcLen * colValAlign_], dyGm_[rowIdx * colVal_], colValAlign_);
        } else {
            DataCopy(dy, dyGm_[rowIdx * colVal_], colValAlign_);
        }
#endif

        inQueRstd_.EnQue(rstd);
        inQueX_.EnQue(xLocal);
        inQueDY_.EnQue(dy);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx, uint32_t calcLen)
    {
        LocalTensor<T_DY> dx = outQueDX_.DeQue<T_DY>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{(uint16_t)calcLen, (uint32_t)(colVal_ * sizeof(T_DY)), 0, 0, 0};
        DataCopyPad(dxGm_[rowIdx * colVal_], dx, dataCopyParams);
#else
        DataCopyCustom<T_DY>(dxGm_, dx, rowIdx * colVal_, 0, colVal_);
#endif
        outQueDX_.FreeTensor(dx);
    }

    __aicore__ inline void CopyDgammaOut()
    {
        LocalTensor<float> dgammaOut = outQueDgamma2_.DeQue<float>();
        SetAtomicAdd<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_, dgammaOut, dataCopyParams);
#else
        DataCopy(dgammaGm_, dgammaOut, ROUND_UP(colVal_, ALIGN_32));
#endif
        SetAtomicNone();
        outQueDgamma2_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void CopyDgammaOutWorkspace()
    {
        LocalTensor<float> dgammaOut = outQueDgamma2_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{(uint16_t)1, (uint32_t)(colValAlign_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceMiddleDgammaGm_, dgammaOut, dataCopyParams);
#else
        uint32_t colValAlign = (colVal_ / ALIGN_32) * ALIGN_32;
        uint32_t colValTail = colVal_ % ALIGN_32;
        DataCopy(workspaceGm_, dgammaOut, colValAlign);
        if (colValTail != 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            for (uint32_t i = 0; i < ALIGN_32; i++) {
                float tensorValue = dgammaOut.GetValue(colVal_ - ALIGN_32 + i);
                dgammaOut.SetValue(i, tensorValue);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(workspaceGm_[colVal_ - ALIGN_32], dgammaOut, ALIGN_32);
        }
#endif
        outQueDgamma2_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void AddDgamma()
    {
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> dgammaLocal = outQueDgamma2_.AllocTensor<float>();
        Duplicate(dgammaLocal, 0.0f, colValAlign_);
        PipeBarrier<PIPE_V>();
        DataCopyParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint16_t>((colValAlign_ * sizeof(float)) / 32), 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        for (uint32_t blockidx = 0; blockidx < blockDim_; blockidx++) {
            LocalTensor<float> dgammaPart = inQueGamma2_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
            DataCopy(dgammaPart, workspaceMiddleDgammaGm_[blockidx * saveLineNum * colValAlign_], dataCopyParams);
#else
            DataCopy(dgammaPart, workspaceGm_[blockidx * colVal_], colValAlign_);
#endif
            inQueGamma2_.EnQue(dgammaPart);
            LocalTensor<float> dgammaPartLocal = inQueGamma2_.DeQue<float>();
            Add(dgammaLocal, dgammaLocal, dgammaPartLocal, colValAlign_);
            PipeBarrier<PIPE_V>();
            inQueGamma2_.FreeTensor(dgammaPartLocal);
        }
        outQueDgamma2_.EnQue(dgammaLocal);
        LocalTensor<float> dgammaOut = outQueDgamma2_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsOut{1, (uint32_t)(colVal_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_, dgammaOut, dataCopyParamsOut);
#else
        DataCopy(dgammaGm_, dgammaOut, colValAlign_);
#endif
        outQueDgamma2_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void Compute(LocalTensor<float> &gammaLocal, LocalTensor<float> &dgammaLocal)
    {
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float rstd_value = rstdLocal.GetValue(0);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        inQueRstd_.FreeTensor(rstdLocal);
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, colValAlign_, colValAlign_);
        WaitFlag<HardEvent::S_V>(eventSV);
        // x*rstd
        Muls(xLocal, xLocal, rstd_value, colValAlign_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, colValAlign_, colValAlign_);
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        // y * x * rstd
        Mul(dxLocal, dyLocal, xLocal, colValAlign_);
        PipeBarrier<PIPE_V>();

        Add(dgammaLocal, dgammaLocal, dxLocal, colValAlign_);
        PipeBarrier<PIPE_V>();

        // y * gamma
        Mul(dyLocal, dyLocal, gammaLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        //  y * gamma * x * rstd
        Mul(dxLocal, dyLocal, xLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        float sumValue = ReduceSumHalfInterval(dxLocal, colVal_);
        float meanValue = sumValue * avgFactor_;
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV2);
        WaitFlag<HardEvent::S_V>(eventSV2);
        // meanValue * x * rstd
        Muls(dxLocal, xLocal, meanValue, colValAlign_);
        PipeBarrier<PIPE_V>();
        // y * gamma - meanValue * x * rstd
        Sub(dxLocal, dyLocal, dxLocal, colValAlign_);
        PipeBarrier<PIPE_V>();
        Muls(dxLocal, dxLocal, rstd_value, colValAlign_);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSame<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, colValAlign_);
            PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (IsSame<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, colValAlign_);
            PipeBarrier<PIPE_V>();
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

    __aicore__ inline void ComputeSmallD(
        uint32_t loopIdx, uint32_t calcLen, LocalTensor<float> &gammaLocal, LocalTensor<float> &dgammaLocal)
    {
        uint32_t elementNum = colValAlign_ * calcLen;

        LocalTensor<float> tmp_reduce_buf = tmpBuf_.Get<float>();
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        // y = x * rstd
        const uint32_t srcN1Shape[2] = {calcLen, 1};
        const uint32_t dstNDShape[2] = {calcLen, colValAlign_};
        auto sharedTmp = tmp_reduce_buf.ReinterpretCast<uint8_t>();
        BroadCast<float, DIM_NUM, DIM_D>(dxLocal, rstdLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, elementNum, elementNum);

        Mul(xLocal, xLocal, dxLocal, elementNum);  // x save x*rstd
        PipeBarrier<PIPE_V>();

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, elementNum, elementNum);
        // dg=sum(dy * (x * rstd), dim=0)
        Mul(dxLocal, dyLocal, xLocal, elementNum);
        PipeBarrier<PIPE_V>();

        Add(dgammaLocal, dxLocal, dgammaLocal, calcLen * colValAlign_);
        PipeBarrier<PIPE_V>();

        // broadcast gamma
        const uint32_t src1DShape[2] = {1, colValAlign_};
        BroadCast<float, DIM_NUM, DIM_N>(dxLocal, gammaLocal, dstNDShape, src1DShape, sharedTmp);  // x reuse gamma_nd
        PipeBarrier<PIPE_V>();
        // dy * gamma
        Mul(dyLocal, dyLocal, dxLocal, elementNum);  // dy save dy*gamma
        PipeBarrier<PIPE_V>();
        Mul(dxLocal, dyLocal, xLocal, elementNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpMeanLocal = tmpMeanBuf_.Get<float>();
        ReduceSumMultiN(tmpMeanLocal, dxLocal, tmp_reduce_buf, calcLen, colVal_, colValAlign_);
        PipeBarrier<PIPE_V>();
        Muls(tmpMeanLocal, tmpMeanLocal, avgFactor_, calcLen);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(dxLocal, tmpMeanLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();
        Mul(dxLocal, xLocal, dxLocal, elementNum);
        PipeBarrier<PIPE_V>();
        Sub(dxLocal, dyLocal, dxLocal, elementNum);
        PipeBarrier<PIPE_V>();
        BroadCast<float, DIM_NUM, DIM_D>(dyLocal, rstdLocal, dstNDShape, srcN1Shape, sharedTmp);
        PipeBarrier<PIPE_V>();
        inQueRstd_.FreeTensor(rstdLocal);
        Mul(dxLocal, dxLocal, dyLocal, elementNum);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSame<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, elementNum);
            PipeBarrier<PIPE_V>();
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (IsSame<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, elementNum);
            PipeBarrier<PIPE_V>();
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

    __aicore__ inline void InitMiddleQue()
    {
        bufferNum_ = BUFFER_NUM_DB;
        uint32_t inQueDgammaSize = 0;
        uint32_t outQueMiddleDgammaSize = colValAlign_ * sizeof(float) * bufferNum_;
        if (isDeterministic_ == 1) {
            // 确定性计算搬入
            inQueDgammaSize = colValAlign_ * sizeof(float);
            pipeMiddle.InitBuffer(inQueGamma2_, 1, inQueDgammaSize);
        }
        // 可用UB大小
        totalUbSize =
            (191 * 1024 - outQueMiddleDgammaSize - inQueDgammaSize - colValAlign_ * sizeof(float)) / bufferNum_;

        // 每次搬入多少行
        cutInRowNum = 128;
        if (cutInRowNum >= saveLine) {
            cutInRowNum = saveLine;
            cutInRowTail = 0;
            cutInRowLoop = 1;
        } else {
            cutInRowTail = saveLine % cutInRowNum;
            // 多少次可以搬完所有行 = 下一次计算的行数
            cutInRowLoop = cutInRowTail > 0 ? (saveLine / cutInRowNum + 1) : (saveLine / cutInRowNum);
        }
        // 每次可以搬入的列数
        cutInColNum = totalUbSize / (cutInRowNum * sizeof(float));
        // 32字节对齐
        cutInColNum = cutInColNum / 8 * 8;

        if (cutInColNum >= colValAlign_) {
            cutInColNum = colValAlign_;
            cutInColTails = 0;
            cutInColLoop = 1;
        } else {
            cutInColTails = colValAlign_ % cutInColNum;
            // 多少次可以搬完所有列
            cutInColLoop = cutInColTails > 0 ? (colValAlign_ / cutInColNum + 1) : (colValAlign_ / cutInColNum);
        }

        // 搬入中间结果
        pipeMiddle.InitBuffer(inQueMiddleDagamma_, bufferNum_, totalUbSize);
        // 搬出dgamma, 大小为colValAlign_
        pipeMiddle.InitBuffer(outQueMiddleDgamma_, bufferNum_, colValAlign_ * sizeof(float));
        pipeMiddle.InitBuffer(outQueDgamma2_, 1, colValAlign_ * sizeof(float));
    }

    __aicore__ inline void CopyDgammaMiddleOutWorkspace(
        uint32_t rowIdx, uint32_t opType, uint32_t calcLen, LocalTensor<float> &dgammaLocal, uint32_t ttL)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        if (opType == 2) {
            DataCopyParams dataCopyParams{
                static_cast<uint16_t>(1), static_cast<uint16_t>((calcLen * colValAlign_ * sizeof(float)) / 32), 0, 0};
            DataCopy(workspaceMiddleDgammaGm_[rowIdx * colValAlign_], dgammaLocal, dataCopyParams);
        } else {
            // 2行中的第一行或者所有行的最后一个单行
            if (rowIdx % 2 == 0 || (rowIdx == ttL - 1 && ttL % 2 > 0)) {
                DataCopyParams dataCopyParams{
                    static_cast<uint16_t>(1), static_cast<uint16_t>((colValAlign_ * sizeof(float)) / 32), 0, 0};
                DataCopy(workspaceMiddleDgammaGm_[(rowIdx / 2) * colValAlign_], dgammaLocal, dataCopyParams);
            } else {
                SetAtomicAdd<float>();
                DataCopyParams dataCopyParams{
                    static_cast<uint16_t>(1), static_cast<uint16_t>((colValAlign_ * sizeof(float)) / 32), 0, 0};
                DataCopy(workspaceMiddleDgammaGm_[(((rowIdx - 1) / 2)) * colValAlign_], dgammaLocal, dataCopyParams);
                SetAtomicNone();
            }
        }
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
    }

    // 计算拷入
    __aicore__ inline void CopyDgammaMiddleIn(uint32_t calcRowIdx, uint32_t calcColIdx, uint32_t blockCounts,
        uint32_t blockSize, LocalTensor<float> &dstTensor)
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        uint32_t offset = cutInRowNum * calcRowIdx * colValAlign_ + calcColIdx * cutInColNum;

        DataCopyParams dataCopyParams{static_cast<uint16_t>(blockCounts),
            static_cast<uint16_t>((blockSize * sizeof(float)) / 32),
            static_cast<uint16_t>(((colValAlign_ - blockSize) * sizeof(float)) / 32),
            0};
        DataCopy(dstTensor, workspaceMiddleDgammaGm_[offset], dataCopyParams);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void DoAParallelReduce(
        LocalTensor<float> &dstTensor, int64_t lineLength, uint32_t rowNum, uint32_t idx)
    {
        /*
        函数实现rowNum行的二分累加，累加结果为一行
        srcTensor为reduce之前的Tensor: rowNum * lineLength
        dstTensor为存放reduce结果的tensor: 1 * lineLength
        */
        int64_t nowRows = rowNum;
        LocalTensor<float> srcTensor = inQueMiddleDagamma_.DeQue<float>();
        if (nowRows == 1) {
            Adds<float>(dstTensor[idx * cutInColNum], srcTensor, 0, lineLength);
            PipeBarrier<PIPE_V>();
            inQueMiddleDagamma_.FreeTensor(srcTensor);
            return;
        }
        uint32_t dichotomizeAddDiffSize = 0;
        dichotomizeAddDiffSize = FindDichotomizeAddDiffSize(rowNum);

        // row为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            uint32_t newNum = (nowRows - dichotomizeAddDiffSize) * lineLength;
            Add(srcTensor, srcTensor, srcTensor[newNum], dichotomizeAddDiffSize * lineLength);
            PipeBarrier<PIPE_V>();
            nowRows = nowRows - dichotomizeAddDiffSize;
        }

        while (nowRows > 1) {
            nowRows = nowRows / 2;
            if (nowRows == 1) {
                Add(dstTensor[idx * cutInColNum], srcTensor, srcTensor[lineLength], lineLength);
                PipeBarrier<PIPE_V>();
            } else {
                Add(srcTensor, srcTensor, srcTensor[nowRows * lineLength], nowRows * lineLength);
                PipeBarrier<PIPE_V>();
            }
        }

        inQueMiddleDagamma_.FreeTensor(srcTensor);
    }

    __aicore__ inline void computeMiddleDgamma()
    {
        while (cutInRowLoop >= 1) {
            for (uint32_t calcRowIdx = 0; calcRowIdx < cutInRowLoop; calcRowIdx++) {
                LocalTensor<float> middleGammaLocal = outQueMiddleDgamma_.AllocTensor<float>();
                Duplicate(middleGammaLocal, 0.0f, colValAlign_);
                PipeBarrier<PIPE_V>();
                // 每循环一次会得到一整行的结果
                uint32_t blockCounts = cutInRowNum;
                if (calcRowIdx == cutInRowLoop - 1 && cutInRowTail > 0) {
                    blockCounts = cutInRowTail;
                }
                for (uint32_t calcColIdx = 0; calcColIdx < cutInColLoop; calcColIdx++) {
                    uint32_t blockSize = cutInColNum;
                    if (calcColIdx == cutInColLoop - 1 && cutInColTails > 0) {
                        blockSize = cutInColTails;
                    }
                    LocalTensor<float> dgammaMiddleIn = inQueMiddleDagamma_.AllocTensor<float>();
                    CopyDgammaMiddleIn(calcRowIdx, calcColIdx, blockCounts, blockSize, dgammaMiddleIn);
                    inQueMiddleDagamma_.EnQue(dgammaMiddleIn);
                    DoAParallelReduce(middleGammaLocal, blockSize, blockCounts, calcColIdx);
                }
                // 存这一行的结果，直接覆盖，存完释放middleGammaLocal
                if (cutInRowLoop == 1) {
                    outQueDgamma2_.EnQue(middleGammaLocal);
                    DoDGamma();
                } else {
                    CopyDgammaMiddleOutWorkspace(calcRowIdx, 1, 1, middleGammaLocal, saveLineNew);
                    outQueMiddleDgamma_.FreeTensor(middleGammaLocal);
                }
            }
            // 更新totalLine，cutInRowLoop，cutInRowNum，cutInRowTail，cutInColNum，cutInColLoop，cutInColTails
            if (cutInRowLoop > 1) {
                saveLineNew = (cutInRowLoop % 2 > 0) ? (cutInRowLoop / 2 + 1) : cutInRowLoop / 2;
                // 每次搬入多少行
                cutInRowNum = 128;
                if (cutInRowNum >= saveLineNew) {
                    cutInRowNum = saveLineNew;
                    cutInRowTail = 0;
                    cutInRowLoop = 1;
                } else {
                    cutInRowTail = saveLineNew % cutInRowNum;
                    // 多少次可以搬完所有行 = 下一次计算的行数
                    cutInRowLoop = cutInRowTail > 0 ? (saveLineNew / cutInRowNum + 1) : (saveLineNew / cutInRowNum);
                }
                // 每次可以搬入的列数
                cutInColNum = totalUbSize / (cutInRowNum * sizeof(float));
                // 32字节对齐
                cutInColNum = cutInColNum / 8 * 8;

                if (cutInColNum >= colValAlign_) {
                    cutInColNum = colValAlign_;
                    cutInColTails = 0;
                    cutInColLoop = 1;
                } else {
                    cutInColTails = colValAlign_ % cutInColNum;
                    // 多少次可以搬完所有列
                    cutInColLoop = cutInColTails > 0 ? (colValAlign_ / cutInColNum + 1) : (colValAlign_ / cutInColNum);
                }
            } else {
                cutInRowLoop = 0;
            }
        }
    }

public:
    uint32_t rowVal_;
    uint32_t colVal_;
    uint32_t colValAlign_;
    float avgFactor_{1.0f};
    uint32_t coreCalcNum_;
    uint32_t coreCalcTail_;
    uint32_t blockFactor_;
    uint32_t blockDim_;
    uint32_t ubFactor_;
    uint32_t ubCalcNum;
    uint32_t ubCalcTail_;
    uint32_t ubCalcLoop_;
    uint32_t ubCalcTailNum_;
    uint32_t ubCalcTailTail_;
    uint32_t ubCalcTailLoop_;
    uint32_t dataType_;
    uint32_t alignLen_;
    uint32_t coreOffset_;
    uint32_t ubFactorAlign_;
    uint32_t rstdLen_;
    uint32_t bufferLenSize_;
    int32_t bufferNum_;
    uint32_t isDeterministic_{0};

    uint32_t cutInRowNum;
    uint32_t cutInRowLoop;
    uint32_t cutInRowTail;
    uint32_t cutInColNum;
    uint32_t cutInColTails;
    uint32_t cutInColLoop;
    uint32_t totalLine;
    uint32_t saveLineNum;
    uint32_t saveLineTailNum;
    uint32_t saveLine;
    uint32_t saveLineNew;
    uint32_t totalUbSize;

    TPipe pipe;
    TPipe pipeMiddle;
    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_DY> dxGm_;
    GlobalTensor<T_DY> xGm_;
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<float> dgammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> workspaceGm_;
    GlobalTensor<float> workspaceMiddleDgammaGm_;
    GlobalTensor<int32_t> syncTmpSpaceGm_;
    TQue<QuePosition::VECIN, 1> inQueDY_;
    TQue<QuePosition::VECIN, 1> inQueX_;
    TQue<QuePosition::VECIN, 1> inQueRstd_;
    TQue<QuePosition::VECIN, 1> inQueGamma_;
    TQue<QuePosition::VECOUT, 1> outQueDgamma_;
    TQue<QuePosition::VECOUT, 1> outQueDX_;

    // 二分用到的ub
    TQue<QuePosition::VECIN, 1> inQueMiddleDagamma_;
    TQue<QuePosition::VECIN, 1> inQueGamma2_;
    TQue<QuePosition::VECOUT, 1> outQueDgamma2_;
    TQue<QuePosition::VECOUT, 1> outQueMiddleDgamma_;

    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> tmpMeanBuf_;
    TBuf<TPosition::VECCALC> outZeroTmpBuf_;
    TBuf<TPosition::VECCALC> syncTmpBuf_;
};
#endif  // RMS_NORM_GRAD_SPLIT_N_HIGH_PRECISION_H_