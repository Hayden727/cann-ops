/* 
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef COPY_KUNLUN_H
#define COPY_KUNLUN_H
#include "kernel_operator.h"


#ifdef TRACE_DCS_ENABLE
#define DataCopySafe(dstTensor, srcTensor, calCount) kunlun::DataCopySafeImpl(dstTensor, srcTensor, calCount, __LINE__);
#else 
#define DataCopySafe(dstTensor, srcTensor, calCount) kunlun::DataCopySafeImpl(dstTensor, srcTensor, calCount);
#endif

namespace kunlun{
    using AscendC::DataCopy;
    using AscendC::DataCopyPad;
    using AscendC::DataCopyExtParams;
    using AscendC::DataCopyPadExtParams;
    using AscendC::LocalTensor;
    using AscendC::GlobalTensor;
    using AscendC::printf;

    #ifdef TRACE_DCS_ENABLE
    template<typename T>
    __aicore__ inline void DataCopySafeImpl(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const uint32_t& calCount, const uint32_t& LINE){
        if(calCount*sizeof(T)%32 == 0){
            DataCopy(dstTensor, srcTensor, calCount);
            printf("GM->UB: [len: %d, dtsize: %d, GmPos:%d, UbPos: %d, copySize: %d, Aligned: true, Line:%d ]\n",calCount,sizeof(T),srcTensor.GetPhyAddr(),dstTensor.GetPhyAddr(),calCount*sizeof(T),LINE);
        }else{
            DataCopyExtParams copyParams{1, calCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(dstTensor, srcTensor, copyParams, padParams); 
            printf("GM->UB: [len: %d, dtsize: %d, GmPos:%d, UbPos: %d, copySize: %d, Aligned: false, Line:%d]\n",calCount,sizeof(T),srcTensor.GetPhyAddr(),dstTensor.GetPhyAddr(),calCount*sizeof(T),LINE);
        }
    }
    template<typename T>
    __aicore__ inline void DataCopySafeImpl(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t& calCount, const uint32_t& LINE){
        if(calCount*sizeof(T)%32 == 0){
            DataCopy(dstTensor, srcTensor, calCount);
            printf("UB->GM: [len: %d, dtsize: %d, UbPos: %d, GmPos: %d, copySize: %d, Aligned: true, Line:%d]\n",calCount,sizeof(T),srcTensor.GetPhyAddr(),dstTensor.GetPhyAddr(),calCount*sizeof(T),LINE);
        }else{
            DataCopyExtParams copyParams{1, calCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
            DataCopyPad(dstTensor, srcTensor, copyParams); 
            printf("UB->GM: [len: %d, dtsize: %d, UbPos: %d, GmPos: %d, copySize: %d, Aligned: false, Line:%d]\n",calCount,sizeof(T),srcTensor.GetPhyAddr(),dstTensor.GetPhyAddr(),calCount*sizeof(T),LINE);
        }
    }
    #else
    template<typename T>
    __aicore__ inline void DataCopySafeImpl(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, const uint32_t& calCount){
        if(calCount*sizeof(T)%32 == 0){
            DataCopy(dstTensor, srcTensor, calCount);
        }else{
            DataCopyExtParams copyParams{1, calCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(dstTensor, srcTensor, copyParams, padParams); 
        }
    }
    template<typename T>
    __aicore__ inline void DataCopySafeImpl(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t& calCount){
        if(calCount*sizeof(T)%32 == 0){
            DataCopy(dstTensor, srcTensor, calCount);
        }else{
            DataCopyExtParams copyParams{1, calCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
            DataCopyPad(dstTensor, srcTensor, copyParams); 
        }
    }
    #endif

}

#endif// COPY_KUNLUN_H
