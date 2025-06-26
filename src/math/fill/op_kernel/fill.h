/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file fill.h
 */

#ifndef __FILL_H_
#define __FILL_H_

constexpr uint32_t BUFFER_NUM = 1; 

template <typename T, bool IsExistBigCore>
class KernelFill {
 public:
  __aicore__ inline KernelFill() {}
  __aicore__ inline void Init(GM_ADDR dims, GM_ADDR value, GM_ADDR y, uint32_t smallCoreDataNum,
    uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, 
    uint32_t smallCoreLoopNum, uint32_t ubPartDataNum, 
    uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum, 
    uint32_t tailBlockNum)  {
    
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = AscendC::GetBlockIdx();

    uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
    this->ubPartDataNum = ubPartDataNum;

    if constexpr (IsExistBigCore) 
    {
      if (coreNum < tailBlockNum) 
      { 
        this->coreDataNum = bigCoreDataNum;
        this->tileNum = bigCoreLoopNum;
        this->tailDataNum = bigCoreTailDataNum;
      }
      else 
      { 
        this->coreDataNum = smallCoreDataNum;
        this->tileNum = smallCoreLoopNum;
        this->tailDataNum = smallCoreTailDataNum;
        globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
      }
    }
    else
    {
      this->coreDataNum = smallCoreDataNum;
      this->tileNum = smallCoreLoopNum;
      this->tailDataNum = smallCoreTailDataNum;
      globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
    }

    this->value = *reinterpret_cast<__gm__ DTYPE_VALUE *>(value); 
    yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool> ) {
        pipe.InitBuffer(tmpBuf, this->ubPartDataNum * sizeof(half));
    }
    
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->ubPartDataNum * sizeof(DTYPE_Y));
  }
  __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount-1; i++) 
        {
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        Compute(loopCount-1);
        CopyOut(loopCount-1);
    }

 private:
  __aicore__ inline void Compute(uint32_t progress) {

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bool> ) {

      AscendC::LocalTensor<int8_t> outLocal = outQueueOUT.AllocTensor<int8_t>();
      AscendC::LocalTensor<half> tmpLocal = tmpBuf.Get<half>();
      AscendC::Duplicate<half>(tmpLocal, (half)(this->value), this->processDataNum);

      Cast(outLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      
      outQueueOUT.EnQue<int8_t>(outLocal);

    } else {

      AscendC::LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();
      AscendC::Duplicate<T>(outLocal, this->value, this->processDataNum);
      outQueueOUT.EnQue<T>(outLocal);
    
    }   
  }

  __aicore__ inline void CopyOut(uint32_t progress) {

    AscendC::LocalTensor<DTYPE_VALUE> outLocal = outQueueOUT.DeQue<DTYPE_VALUE>();
    AscendC::DataCopy(yGm[progress * this->ubPartDataNum], outLocal, this->processDataNum);
    outQueueOUT.FreeTensor(outLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  AscendC::GlobalTensor<DTYPE_Y> yGm;

  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
  T value;

  uint32_t coreDataNum;
  uint32_t tileNum;
  uint32_t ubPartDataNum;
  uint32_t tailDataNum;
  uint32_t processDataNum;
};
#endif