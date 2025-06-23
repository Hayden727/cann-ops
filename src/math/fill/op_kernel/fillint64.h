#ifndef __FILLINT64_H_
#define __FILLINT64_H_

template <bool IsExistBigCore>
class KernelFill1_INT64 {
 public:
  __aicore__ inline KernelFill1_INT64() {}
  __aicore__ inline void Init(GM_ADDR dims, GM_ADDR values, GM_ADDR y, uint32_t smallCoreDataNum,
    uint32_t bigCoreDataNum, uint32_t bigCoreLoopNum, 
    uint32_t smallCoreLoopNum, uint32_t ubPartDataNum, 
    uint32_t smallCoreTailDataNum, uint32_t bigCoreTailDataNum, 
    uint32_t tailBlockNum) {
      
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

    xGm.SetGlobalBuffer((__gm__ int32_t*)values, 2); 
    yGm.SetGlobalBuffer((__gm__ int32_t*)y + globalBufferIndex, this->coreDataNum);

    this->high = xGm.GetValue(1);
    this->low = xGm.GetValue(0);
    
    
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->ubPartDataNum * sizeof(int32_t));
  }
  __aicore__ inline void Process() {
    //  uint32_t loopCount = this->tileNum * BUFFER_NUM;
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->ubPartDataNum;
    this->repeatTimes = (this->processDataNum + 63)/  64;
    for (int32_t i = 0; i < loopCount-1; i++) 
    {
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    this->repeatTimes = (this->processDataNum + 63)/  64;
    Compute(loopCount-1);
    CopyOut(loopCount-1);
  }

 private:
  __aicore__ inline void Compute(uint32_t progress) {   
        AscendC::LocalTensor<int32_t> outLocal = outQueueOUT.AllocTensor<int32_t>();

        uint64_t mask2[2] = {0xAAAAAAAAAAAAAAAA, 0};
        uint64_t mask1[2] = {0x5555555555555555, 0};

        AscendC::Duplicate(outLocal, low, mask1,  this->repeatTimes, 1, 8);
        AscendC::Duplicate(outLocal, high, mask2,  this->repeatTimes, 1, 8);

        outQueueOUT.EnQue<int32_t>(outLocal);
  }
  __aicore__ inline void CopyOut(uint32_t progress) {
    AscendC::LocalTensor<int32_t> outLocal = outQueueOUT.DeQue<int32_t>();

    AscendC::DataCopy(yGm[progress * this->ubPartDataNum], outLocal, this->processDataNum);
    
    outQueueOUT.FreeTensor(outLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  AscendC::GlobalTensor<int32_t> yGm;
  AscendC::GlobalTensor<int32_t> xGm;

  int64_t value;

  uint32_t coreDataNum;
  uint32_t tileNum;
  uint32_t ubPartDataNum;
  uint32_t tailDataNum;
  uint32_t processDataNum;

  int32_t high;
  int32_t low;
  uint32_t repeatTimes;
};

#endif