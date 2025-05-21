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
 * \file rms_norm_grad_split_d_high_precision.h
 * \brief
 */
#ifndef RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_
#define RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_
#include "rms_norm_grad_common.h"
template <typename T_DY, typename T_GAMMA>
class RmsNormGradSplitDHighPrecision {
public:
    __aicore__ inline RmsNormGradSplitDHighPrecision()
    {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma,
        const RmsNormGradTilingData *tiling, GM_ADDR usrWorkspace)
    {
        InitVar(tiling);
        InitGmBuffer(dy, x, rstd, gamma, dx, dgamma, usrWorkspace);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
        InitGammaFor310(gamma, dgamma, usrWorkspace);
#endif
        InitUB();
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

        loopMainCol_ = colVal_ / ubFactor_;
        tailCol_ = colVal_ % ubFactor_;

        rowFactor_ = ROW_FACTOR_SPLIT_D;
        alignLen_ = dataType_ == FLOAT_DTYPE ? ALIGN_32 : ALIGN_16;

        colValAlign_ = (colVal_ + alignLen_ - 1) / alignLen_ * alignLen_;
        tailColAlign_ = colValAlign_ % ubFactor_;
        isDeterministic_ = tiling->fixed_output;

        saveLineNum = (coreCalcNum_ % 2 == 0) ? coreCalcNum_ / 2 : ((coreCalcNum_ / 2) + 1);
        saveLineTailNum = (coreCalcTail_ % 2 == 0) ? coreCalcTail_ / 2 : ((coreCalcTail_ / 2) + 1);
        if (coreCalcTail_ != 0 && GetBlockIdx() == blockDim_ - 1) {
            totalLine = coreCalcTail_;
            saveLine = saveLineTailNum > 0 ? saveLineTailNum : 1;
        } else {
            totalLine = coreCalcNum_;
            saveLine = saveLineNum > 0 ? saveLineNum : 1;
        }
    }

    __aicore__ inline void InitGmBuffer(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR usrWorkspace)
    {
        if (GetBlockIdx() < blockDim_ - 1) {
            coreOffset_ = blockFactor_;
        } else {
            coreOffset_ = coreCalcTail_ > 0 ? coreCalcTail_ : blockFactor_;
        }
        coreOffsetStart_ = blockFactor_ * colVal_;
        coreOffsetLen_ = coreOffset_ * colVal_;
        dyGm_.SetGlobalBuffer((__gm__ T_DY *)dy + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        xGm_.SetGlobalBuffer((__gm__ T_DY *)x + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        rstdGm_.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * blockFactor_, coreOffset_);
        dxGm_.SetGlobalBuffer((__gm__ T_DY *)dx + GetBlockIdx() * coreOffsetStart_, coreOffsetLen_);
        gammaGm_.SetGlobalBuffer((__gm__ T_GAMMA *)gamma, colVal_);
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ != 200)
        dgammaGm_.SetGlobalBuffer((__gm__ float *)dgamma, colVal_);
        if (isDeterministic_ == 0 && GetBlockIdx() == 0) {
            InitOutput<float>(dgammaGm_, colVal_, 0);
        }
        if (isDeterministic_) {
            workspaceMiddleDgammaGm_.SetGlobalBuffer(
                (__gm__ float *)usrWorkspace + GetBlockIdx() * saveLineNum * colValAlign_, saveLine * colValAlign_);
        } else {
            workspaceMiddleDgammaGm_.SetGlobalBuffer(
                (__gm__ float *)usrWorkspace + GetBlockIdx() * saveLineNum * colValAlign_, saveLine * colValAlign_);
        }
        SyncAll();
#endif
    }

    __aicore__ inline void InitGammaFor310(GM_ADDR gamma, GM_ADDR dgamma, GM_ADDR usrWorkspace)
    {
        uint32_t syncLen = ALIGN_32 * GetBlockNum();
        dgammaGm_.SetGlobalBuffer((__gm__ float *)dgamma, colValAlign_);
        syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, syncLen);

        pipe.InitBuffer(outZeroTmpBuf_, ubFactor_ * sizeof(float));
        pipe.InitBuffer(syncTmpBuf_, syncLen * sizeof(int32_t));

        InitGmZero<int32_t>(syncTmpSpaceGm_, outZeroTmpBuf_, syncLen, (uint32_t)0);
        if (isDeterministic_ == 0) {
            if (GetBlockIdx() == 0) {
                InitDgammaOut(dgammaGm_);
            }
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
        } else if (isDeterministic_) {
            workspaceGm_.SetGlobalBuffer((__gm__ float *)usrWorkspace + syncLen + GetBlockIdx() * colVal_);
            if (GetBlockIdx() == 0) {
                InitDgammaOut(workspaceGm_);
            }
        }
    }

    __aicore__ inline void InitDgammaOut(GlobalTensor<float> &outGm)
    {
        for (uint32_t iOuter = 0; iOuter < loopMainCol_; iOuter++) {
            InitGmZero<float>(outGm, outZeroTmpBuf_, ubFactor_, iOuter * ubFactor_);
        }

        if (tailCol_ > 0) {
            InitGmZero<float>(outGm, outZeroTmpBuf_, tailCol_, loopMainCol_ * ubFactor_);
        }
    }

    __aicore__ inline void InitUB()
    {
        bufferLenSize_ = ubFactor_ * sizeof(float);
        pipe.InitBuffer(inQueDY_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(inQueX_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(inQueRstd_, 1, rowFactor_ * sizeof(float));
        pipe.InitBuffer(inQueGamma_, 1, bufferLenSize_);
        pipe.InitBuffer(outQueDX_, BUFFER_NUM_DB, bufferLenSize_);
        pipe.InitBuffer(outQueDgamma_, 1, bufferLenSize_);
        pipe.InitBuffer(meanBuf_, rowFactor_ * sizeof(float));
        pipe.InitBuffer(meanTmpBuf_, rowFactor_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreCalcTail_ == 0) {
            ProcessMain(blockFactor_);
        } else {
            if (GetBlockIdx() < blockDim_ - 1) {
                ProcessMain(blockFactor_);
            } else {
                ProcessMain(coreCalcTail_);
            }
        }
    }

    __aicore__ inline void ProcessMain(uint32_t loop_len)
    {
        uint32_t loopMainOuter = loop_len / rowFactor_;
        uint32_t tailOuter = loop_len % rowFactor_;
        for (uint32_t iOuter = 0; iOuter < loopMainOuter; iOuter++) {
            SubProcess(iOuter, rowFactor_);
        }
        if (tailOuter > 0) {
            SubProcess(loopMainOuter, tailOuter);
        }

        // 销毁已申请的所有UB，重新申请分配
        pipe.Destroy();
        InitMiddleQue();
        for (uint32_t j = 0; j < loopMainCol_; j++) {
            computeMiddleDgamma(j, ubFactor_);
            if (isDeterministic_) {
                CopyDgammaOut(j, ubFactor_, workspaceMiddleDgammaGm_, 2);
            } else {
                CopyDgammaOut(j, ubFactor_, dgammaGm_, 1);
            }
        }
        if (tailCol_ > 0) {
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

            if (cutInColNum >= tailColAlign_) {
                cutInColNum = tailColAlign_;
                cutInColTails = 0;
                cutInColLoop = 1;
            } else {
                cutInColTails = tailColAlign_ % cutInColNum;
                // 多少次可以搬完所有列
                cutInColLoop = cutInColTails > 0 ? (tailColAlign_ / cutInColNum + 1) : (tailColAlign_ / cutInColNum);
            }
            computeMiddleDgamma(loopMainCol_, tailColAlign_);
            if (isDeterministic_) {
                CopyDgammaOut(loopMainCol_, tailCol_, workspaceMiddleDgammaGm_, 2);
            } else {
                CopyDgammaOut(loopMainCol_, tailCol_, dgammaGm_, 1);
            }
        }

        if (isDeterministic_) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
            LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
            SyncAll(syncTmpSpaceGm_, workLocal);
#else
            SyncAll();
#endif
            if (GetBlockIdx() != 0) {
                return;
            }
            for (uint32_t j = 0; j < loopMainCol_; j++) {
                AddDgamma(j, ubFactor_);
            }
            if (tailCol_ > 0) {
                AddDgamma(loopMainCol_, tailCol_);
            }
        }
    }

    __aicore__ inline void SubProcess(uint32_t iOuter, uint32_t calcRow)
    {
        // CopyRstd
        CopyRstdIn(iOuter, calcRow);
        LocalTensor<float> rstdLocal = inQueRstd_.DeQue<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        Duplicate(meanLocal, 0.0f, calcRow);
        for (uint32_t j = 0; j < loopMainCol_; j++) {
            loopColProcessBeforeReduce(iOuter, j, calcRow, ubFactor_, rstdLocal);
        }
        if (tailCol_ > 0) {
            loopColProcessBeforeReduce(iOuter, loopMainCol_, calcRow, tailCol_, rstdLocal);
        }
        Muls(meanLocal, meanLocal, avgFactor_, calcRow);
        pipe_barrier(PIPE_V);
        for (uint32_t j = 0; j < loopMainCol_; j++) {
            loopColProcessAfterReduce(iOuter, j, calcRow, ubFactor_, rstdLocal);
        }
        if (tailCol_ > 0) {
            loopColProcessAfterReduce(iOuter, loopMainCol_, calcRow, tailCol_, rstdLocal);
        }
        inQueRstd_.FreeTensor(rstdLocal);
    }

    // calcCol是单次搬入计算的列数 iOuter是行循环的索引， j是列循环的索引
    __aicore__ inline void loopColProcessBeforeReduce(
        uint32_t iOuter, uint32_t j, uint32_t calcRow, uint32_t calcCol, LocalTensor<float> &rstdLocal)
    {
        CopyGammaIn(j, calcCol);
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, ubFactor_, calcCol);

        for (uint32_t iInner = 0; iInner < calcRow; iInner++) {
            LocalTensor<float> dgamma = outQueDgamma_.AllocTensor<float>();
            Duplicate(dgamma, 0.0f, calcCol);
            CopyIn(iOuter * rowFactor_ + iInner, j, calcCol);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            float rstdValue = rstdLocal.GetValue(iInner);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            set_flag(PIPE_S, PIPE_V, eventSV);
            wait_flag(PIPE_S, PIPE_V, eventSV);
            ComputeFormer(iInner, rstdValue, calcCol, gammaLocal, dgamma);
            CopyDgammaMiddleOutWorkspace(iOuter * rowFactor_ + iInner, calcCol, j, dgamma, totalLine);
            outQueDgamma_.FreeTensor(dgamma);
        }
        inQueGamma_.FreeTensor(gammaLocal);
        LocalTensor<float> meanTmpLocal = meanTmpBuf_.Get<float>();
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        Add(meanLocal, meanLocal, meanTmpLocal, calcRow);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void loopColProcessAfterReduce(
        uint32_t iOuter, uint32_t j, uint32_t calcRow, uint32_t calcCol, LocalTensor<float> &rstdLocal)
    {
        CopyGammaIn(j, calcCol);
        LocalTensor<float> gammaLocal = inQueGamma_.DeQue<float>();
        Cast2FloatIf<T_GAMMA>(gammaLocal, ubFactor_, calcCol);
        LocalTensor<float> meanLocal = meanBuf_.Get<float>();
        for (uint32_t iInner = 0; iInner < calcRow; iInner++) {
            CopyIn(iOuter * rowFactor_ + iInner, j, calcCol);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            set_flag(PIPE_V, PIPE_S, eventVS);
            wait_flag(PIPE_V, PIPE_S, eventVS);
            float rstdValue = rstdLocal.GetValue(iInner);
            float meanValue = meanLocal.GetValue(iInner);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            set_flag(PIPE_S, PIPE_V, eventSV);
            wait_flag(PIPE_S, PIPE_V, eventSV);
            ComputeLatter(rstdValue, meanValue, calcCol, gammaLocal);
            CopyDxOut(iOuter * rowFactor_ + iInner, j, calcCol);
        }
        inQueGamma_.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyRstdIn(uint32_t iOuter, uint32_t calcRow)
    {
        LocalTensor<float> rstdLocal = inQueRstd_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsRstd{1, (uint32_t)(calcRow * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(rstdLocal, rstdGm_[iOuter * rowFactor_], dataCopyParamsRstd, padParams);
#else
        uint32_t calcRow_align = ROUND_UP(calcRow, alignLen_);
        DataCopy(rstdLocal, rstdGm_[iOuter * rowFactor_], calcRow_align);
#endif
        inQueRstd_.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyGammaIn(uint32_t dIdx, uint32_t calcLen)
    {
        LocalTensor<T_GAMMA> gammaLocal = inQueGamma_.AllocTensor<T_GAMMA>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_GAMMA)), 0, 0, 0};
        DataCopyPadExtParams<T_GAMMA> padParams{false, 0, 0, 0};
        if constexpr (!IsSame<T_GAMMA, float>::value) {
            DataCopyPad(gammaLocal[ubFactor_], gammaGm_[dIdx * ubFactor_], dataCopyParams, padParams);
        } else {
            DataCopyPad(gammaLocal, gammaGm_[dIdx * ubFactor_], dataCopyParams, padParams);
        }
#else
        uint32_t calcLen_align = ROUND_UP(calcLen, alignLen_);
        if constexpr (!IsSame<T_GAMMA, float>::value) {
            DataCopy(gammaLocal[ubFactor_], gammaGm_[dIdx * ubFactor_], calcLen_align);
        } else {
            DataCopy(gammaLocal, gammaGm_[dIdx * ubFactor_], calcLen_align);
        }
#endif

        inQueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyDgammaOut(uint32_t dIdx, uint32_t calcLen, GlobalTensor<float> &outGm, uint32_t opType)
    {
        LocalTensor<float> dgamma_out = outQueDgamma2_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        if (opType == 2) {
            DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm[dIdx * ubFactor_], dgamma_out, dataCopyParams);
        } else {
            SetAtomicAdd<float>();
            DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(outGm[dIdx * ubFactor_], dgamma_out, dataCopyParams);
            SetAtomicNone();
        }
#else
        SetAtomicAdd<float>();
        if (calcLen < ALIGN_32) {
            for (uint32_t i = 0; i < ALIGN_32; i++) {
                if (i >= calcLen) {
                    dgamma_out.SetValue(i, 0.0f);
                }
            }
            DataCopy(outGm[dIdx * ubFactor_], dgamma_out, ALIGN_32);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        } else {
            uint32_t calcLenAlign = (calcLen / ALIGN_32) * ALIGN_32;
            uint32_t calcLenTail = calcLen - calcLenAlign;
            DataCopy(outGm[dIdx * ubFactor_], dgamma_out, calcLenAlign);
            if (calcLenTail > 0) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                for (uint32_t i = 0; i < ALIGN_32; i++) {
                    if (i < (ALIGN_32 - calcLenTail)) {
                        dgamma_out.SetValue(i, 0.0f);
                    } else {
                        uint32_t tailOffset = calcLenAlign + i - (ALIGN_32 - calcLenTail);
                        dgamma_out.SetValue(i, dgamma_out.GetValue(tailOffset));
                    }
                }
                DataCopy(outGm[dIdx * ubFactor_ + calcLen - ALIGN_32], dgamma_out, ALIGN_32);
                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            }
        }
        SetAtomicNone();
#endif
        outQueDgamma2_.FreeTensor(dgamma_out);
    }

    __aicore__ inline void CopyIn(uint32_t nIdx, uint32_t dIdx, uint32_t calcLen)
    {
        // x dy, rstd
        LocalTensor<T_DY> xLocal = inQueX_.AllocTensor<T_DY>();
        LocalTensor<T_DY> dyLocal = inQueDY_.AllocTensor<T_DY>();
        uint32_t gmOffset = nIdx * colVal_ + dIdx * ubFactor_;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_DY)), 0, 0, 0};
        DataCopyPadExtParams<T_DY> padParams{true, 0, 0, 0};
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopyPad(xLocal[ubFactor_], xGm_[gmOffset], dataCopyParams, padParams);
        } else {
            DataCopyPad(xLocal, xGm_[gmOffset], dataCopyParams, padParams);
        }
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopyPad(dyLocal[ubFactor_], dyGm_[gmOffset], dataCopyParams, padParams);
        } else {
            DataCopyPad(dyLocal, dyGm_[gmOffset], dataCopyParams, padParams);
        }
#else
        uint32_t calcLen_align = ROUND_UP(calcLen, alignLen_);
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopy(xLocal[ubFactor_], xGm_[gmOffset], calcLen_align);
        } else {
            DataCopy(xLocal, xGm_[gmOffset], calcLen_align);
        }
        if constexpr (!IsSame<T_DY, float>::value) {
            DataCopy(dyLocal[ubFactor_], dyGm_[gmOffset], calcLen_align);
        } else {
            DataCopy(dyLocal, dyGm_[gmOffset], calcLen_align);
        }
#endif
        inQueX_.EnQue(xLocal);
        inQueDY_.EnQue(dyLocal);
    }

    __aicore__ inline void ComputeFormer(
        uint32_t iInner, float rstdValue, uint32_t calcLen, LocalTensor<float> &gammaLocal, LocalTensor<float> &dgamma)
    {
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, ubFactor_, calcLen);
        Muls(xLocal, xLocal, rstdValue, calcLen);  // x*rstd
        pipe_barrier(PIPE_V);

        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, ubFactor_, calcLen);
        Mul(xLocal, dyLocal, xLocal, calcLen);  // dy*x*rstd
        pipe_barrier(PIPE_V);
        Add(dgamma, dgamma, xLocal, calcLen);
        pipe_barrier(PIPE_V);
        Mul(xLocal, xLocal, gammaLocal, calcLen);  // dy*gamma*x*rstd
        pipe_barrier(PIPE_V);
        float sumValue = ReduceSumHalfInterval(xLocal, calcLen);
        LocalTensor<float> meanTmpLocal = meanTmpBuf_.Get<float>();
        meanTmpLocal.SetValue(iInner, sumValue);
        event_t eventSV2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        set_flag(PIPE_S, PIPE_V, eventSV2);
        wait_flag(PIPE_S, PIPE_V, eventSV2);
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ComputeLatter(
        float rstdValue, float meanValue, uint32_t calcLen, LocalTensor<float> &gammaLocal)
    {
        LocalTensor<float> xLocal = inQueX_.DeQue<float>();
        Cast2FloatIf<T_DY>(xLocal, ubFactor_, calcLen);
        Muls(xLocal, xLocal, rstdValue, calcLen);  // x*rstd
        pipe_barrier(PIPE_V);
        Muls(xLocal, xLocal, meanValue, calcLen);  // x*rstd*mean
        LocalTensor<float> dyLocal = inQueDY_.DeQue<float>();
        Cast2FloatIf<T_DY>(dyLocal, ubFactor_, calcLen);
        Mul(dyLocal, dyLocal, gammaLocal, calcLen);  // dy*gamma
        pipe_barrier(PIPE_V);
        LocalTensor<float> dxLocal = outQueDX_.AllocTensor<float>();
        Sub(dxLocal, dyLocal, xLocal, calcLen);
        pipe_barrier(PIPE_V);
        Muls(dxLocal, dxLocal, rstdValue, calcLen);
        pipe_barrier(PIPE_V);
        if constexpr (IsSame<T_DY, half>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, calcLen);
            pipe_barrier(PIPE_V);
        }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        else if constexpr (IsSame<T_DY, bfloat16_t>::value) {
            LocalTensor<T_DY> dxLocalB16 = dxLocal.ReinterpretCast<T_DY>();
            Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, calcLen);
            pipe_barrier(PIPE_V);
        }
#endif
        inQueX_.FreeTensor(xLocal);
        inQueDY_.FreeTensor(dyLocal);
        outQueDX_.EnQue(dxLocal);
    }

    __aicore__ inline void CopyDxOut(uint32_t nIdx, uint32_t dIdx, uint32_t calcLen)
    {
        LocalTensor<T_DY> dx = outQueDX_.DeQue<T_DY>();
        uint32_t gmOffset = nIdx * colVal_ + dIdx * ubFactor_;
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T_DY)), 0, 0, 0};
        DataCopyPad(dxGm_[gmOffset], dx, dataCopyParams);
#else
        uint32_t calcLenAlign32 = (calcLen / alignLen_) * alignLen_;
        if (calcLenAlign32 > 0) {
            DataCopy(dxGm_[gmOffset], dx, calcLenAlign32);
        }
        uint32_t calcLenTail32 = calcLen % alignLen_;

        if (calcLenTail32 > 0) {
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
            for (uint32_t i = 0; i < calcLenTail32; i++) {
                dxGm_.SetValue(gmOffset + calcLenAlign32 + i, dx.GetValue(calcLenAlign32 + i));
            }
            DataCacheCleanAndInvalid<T_DY, CacheLine::ENTIRE_DATA_CACHE>(dxGm_);
            pipe_barrier(PIPE_ALL);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        }
#endif
        outQueDX_.FreeTensor(dx);
    }

    __aicore__ inline void AddDgamma(uint32_t j, uint32_t calcCol)
    {
        uint32_t calcCol_align = ROUND_UP(calcCol, ALIGN_32);
        LocalTensor<float> dgamma = outQueDgamma2_.AllocTensor<float>();
        Duplicate(dgamma, 0.0f, calcCol_align);
        DataCopyExtParams dataCopyParams{1, (uint32_t)(calcCol * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        for (uint32_t blockidx = 0; blockidx < blockDim_; blockidx++) {
            LocalTensor<float> dgammaPart = inQueGamma2_.AllocTensor<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
            DataCopyPad(dgammaPart,
                workspaceMiddleDgammaGm_[blockidx * saveLineNum * colValAlign_ + j * ubFactor_],
                dataCopyParams,
                padParams);
#else
            DataCopy(dgammaPart, workspaceGm_[blockidx * colVal_ + j * ubFactor_], calcCol_align);
#endif
            inQueGamma2_.EnQue(dgammaPart);
            LocalTensor<float> dgammaPartLocal = inQueGamma2_.DeQue<float>();
            Add(dgamma, dgamma, dgammaPartLocal, calcCol);
            pipe_barrier(PIPE_V);
            inQueGamma2_.FreeTensor(dgammaPartLocal);
        }
        outQueDgamma2_.EnQue(dgamma);
        LocalTensor<float> dgammaOut = outQueDgamma2_.DeQue<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
        DataCopyExtParams dataCopyParamsOut{1, (uint32_t)(calcCol * sizeof(float)), 0, 0, 0};
        DataCopyPad(dgammaGm_[j * ubFactor_], dgammaOut, dataCopyParamsOut);
#else
        DataCopy(dgammaGm_[j * ubFactor_], dgammaOut, calcCol_align);
#endif
        outQueDgamma2_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void InitMiddleQue()
    {
        uint32_t inQueDgammaSize = 0;
        uint32_t outQueMiddleDgammaSize = ubFactor_ * sizeof(float) * BUFFER_NUM_DB;
        if (isDeterministic_ == 1) {
            // 确定性计算搬入
            inQueDgammaSize = ubFactor_ * sizeof(float);
            pipeMiddle.InitBuffer(inQueGamma2_, 1, inQueDgammaSize);
        }
        // 可用UB大小
        totalUbSize =
            (191 * 1024 - outQueMiddleDgammaSize - inQueDgammaSize - ubFactor_ * sizeof(float)) / BUFFER_NUM_DB;

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

        if (cutInColNum >= ubFactor_) {
            cutInColNum = ubFactor_;
            cutInColTails = 0;
            cutInColLoop = 1;
        } else {
            cutInColTails = ubFactor_ % cutInColNum;
            // 多少次可以搬完所有列
            cutInColLoop = cutInColTails > 0 ? (ubFactor_ / cutInColNum + 1) : (ubFactor_ / cutInColNum);
        }
        // 搬入中间结果
        pipeMiddle.InitBuffer(inQueMiddleDagamma_, BUFFER_NUM_DB, totalUbSize);
        // 搬出dgamma, 大小为ubFactor_
        pipeMiddle.InitBuffer(outQueMiddleDgamma_, BUFFER_NUM_DB, ubFactor_ * sizeof(float));
        pipeMiddle.InitBuffer(outQueDgamma2_, 1, ubFactor_ * sizeof(float));
    }

    __aicore__ inline void CopyDgammaMiddleOutWorkspace(
        uint32_t rowIdx, uint32_t calcCol, uint32_t ubFactorColIdx, LocalTensor<float> &dgammaLocal, uint32_t ttL)
    {
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        set_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        wait_flag(PIPE_V, PIPE_MTE3, eventVMTE3);
        // 2行中的第一行或者所有行的最后一个单行
        uint32_t calcColAlign_ = (calcCol + alignLen_ - 1) / alignLen_ * alignLen_;
        if (rowIdx % 2 == 0 || (rowIdx == ttL - 1 && ttL % 2 > 0)) {
            uint32_t startOffset = (rowIdx / 2) * colValAlign_ + ubFactorColIdx * ubFactor_;
            DataCopyParams dataCopyParams{
                static_cast<uint16_t>(1), static_cast<uint16_t>((calcColAlign_ * sizeof(float)) / 32), 0, 0};
            DataCopy(workspaceMiddleDgammaGm_[startOffset], dgammaLocal, dataCopyParams);
        } else {
            uint32_t startOffset = (((rowIdx - 1) / 2)) * colValAlign_ + ubFactorColIdx * ubFactor_;
            SetAtomicAdd<float>();
            DataCopyParams dataCopyParams{
                static_cast<uint16_t>(1), static_cast<uint16_t>((calcColAlign_ * sizeof(float)) / 32), 0, 0};
            DataCopy(workspaceMiddleDgammaGm_[startOffset], dgammaLocal, dataCopyParams);
            SetAtomicNone();
        }
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        set_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
        wait_flag(PIPE_MTE3, PIPE_V, eventMTE3V);
    }

    // 计算拷入
    __aicore__ inline void CopyDgammaMiddleIn(uint32_t calcRowIdx, uint32_t calcColIdx, uint32_t blockCounts,
        uint32_t blockSize, LocalTensor<float> &dstTensor, uint32_t calcCol, uint32_t ubFactorColIdx)
    {
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        set_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        wait_flag(PIPE_V, PIPE_MTE2, eventVMTE2);
        DataCopyParams dataCopyParams{static_cast<uint16_t>(blockCounts),
            static_cast<uint16_t>((blockSize * sizeof(float)) / 32),
            static_cast<uint16_t>(((colValAlign_ - blockSize) * sizeof(float)) / 32),
            0};
        DataCopy(dstTensor,
            workspaceMiddleDgammaGm_[cutInRowNum * calcRowIdx * colValAlign_ + ubFactorColIdx * ubFactor_ +
                                     calcColIdx * cutInColNum],
            dataCopyParams);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        set_flag(PIPE_MTE2, PIPE_V, eventIDMTE2ToV);
        wait_flag(PIPE_MTE2, PIPE_V, eventIDMTE2ToV);
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
            pipe_barrier(PIPE_V);
            inQueMiddleDagamma_.FreeTensor(srcTensor);
            return;
        }
        uint32_t dichotomizeAddDiffSize = 0;
        dichotomizeAddDiffSize = FindDichotomizeAddDiffSize(rowNum);

        // row为非二次幂，先将二次幂差值行加到前面
        if (dichotomizeAddDiffSize != 0) {
            uint32_t newNum = (nowRows - dichotomizeAddDiffSize) * lineLength;
            Add(srcTensor, srcTensor, srcTensor[newNum], dichotomizeAddDiffSize * lineLength);
            pipe_barrier(PIPE_V);
            nowRows = nowRows - dichotomizeAddDiffSize;
        }

        while (nowRows > 1) {
            nowRows = nowRows / 2;
            if (nowRows == 1) {
                Add(dstTensor[idx * cutInColNum], srcTensor, srcTensor[lineLength], lineLength);
                pipe_barrier(PIPE_V);
            } else {
                Add(srcTensor, srcTensor, srcTensor[nowRows * lineLength], nowRows * lineLength);
                pipe_barrier(PIPE_V);
            }
        }

        inQueMiddleDagamma_.FreeTensor(srcTensor);
    }

    __aicore__ inline void computeMiddleDgamma(uint32_t ubFactorColIdx, uint32_t calcCol)
    {
        while (cutInRowLoop >= 1) {
            for (uint32_t calcRowIdx = 0; calcRowIdx < cutInRowLoop; calcRowIdx++) {
                LocalTensor<float> middleGammaLocal = outQueMiddleDgamma_.AllocTensor<float>();
                Duplicate(middleGammaLocal, 0.0f, ubFactor_);
                pipe_barrier(PIPE_V);
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
                    CopyDgammaMiddleIn(
                        calcRowIdx, calcColIdx, blockCounts, blockSize, dgammaMiddleIn, calcCol, ubFactorColIdx);
                    inQueMiddleDagamma_.EnQue(dgammaMiddleIn);

                    DoAParallelReduce(middleGammaLocal, blockSize, blockCounts, calcColIdx);
                }
                // 存这一行的结果，直接覆盖，存完释放middleGammaLocal
                if (cutInRowLoop == 1) {
                    outQueDgamma2_.EnQue(middleGammaLocal);
                } else {
                    CopyDgammaMiddleOutWorkspace(calcRowIdx, calcCol, ubFactorColIdx, middleGammaLocal, saveLineNew);
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

                if (cutInColNum >= ubFactor_) {
                    cutInColNum = ubFactor_;
                    cutInColTails = 0;
                    cutInColLoop = 1;
                } else {
                    cutInColTails = ubFactor_ % cutInColNum;
                    // 多少次可以搬完所有列
                    cutInColLoop = cutInColTails > 0 ? (ubFactor_ / cutInColNum + 1) : (ubFactor_ / cutInColNum);
                }
            } else {
                cutInRowLoop = 0;
            }
        }
    }

public:
    uint32_t rowVal_{0};
    uint32_t colVal_{0};
    uint32_t colValAlign_{0};
    float avgFactor_{1.0f};
    uint32_t coreCalcNum_{0};
    uint32_t coreCalcTail_{0};
    uint32_t blockFactor_{0};
    uint32_t blockDim_{0};
    uint32_t ubFactor_{0};

    uint32_t loopMainCol_{0};
    uint32_t tailCol_{0};
    uint32_t tailColAlign_{0};

    uint32_t dataType_{0};
    uint32_t alignLen_{0};
    uint32_t coreOffset_{0};

    uint32_t rowFactor_{0};
    uint32_t bufferLenSize_{0};
    uint32_t coreOffsetStart_{0};
    uint32_t coreOffsetLen_{0};
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
    GlobalTensor<T_GAMMA> gammaGm_;
    GlobalTensor<T_DY> dxGm_;
    GlobalTensor<T_DY> xGm_;
    GlobalTensor<float> workspaceGm_;
    GlobalTensor<float> workspaceMiddleDgammaGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dgammaGm_;
    GlobalTensor<int32_t> syncTmpSpaceGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueDY_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueRstd_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGamma_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueDgamma_;

    // 二分用到的ub
    TQue<QuePosition::VECIN, 1> inQueMiddleDagamma_;
    TQue<QuePosition::VECIN, 1> inQueGamma2_;
    TQue<QuePosition::VECOUT, 1> outQueDgamma2_;
    TQue<QuePosition::VECOUT, 1> outQueMiddleDgamma_;

    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> meanTmpBuf_;
    TBuf<TPosition::VECCALC> outZeroTmpBuf_;
    TBuf<TPosition::VECCALC> syncTmpBuf_;
};
#endif  // RMS_NORM_GRAD_SPLIT_D_HIGH_PRECISION_H_