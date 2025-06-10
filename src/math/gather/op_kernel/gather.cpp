/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <type_traits>

// Marcos
// Tensor Allocation stuff
#define ALLOCBUF(type, name, bufname)   LocalTensor<type> name = bufname.template Get<type>()
#define _ENQUE(type, qname, var)        qname.template EnQue<type>(var)
#define ENQUE(type, name)               _ENQUE(type, q_##name, name)
#define _DEQ(type, qname, name)         name = qname.template DeQue<type>()
#define _DEQ_DEFINE(type, qname, name)  LocalTensor<type> name = qname.template DeQue<type>()
#define _GET(type, bname, var)          var = bname.template Get<type>()
#define _GET_DEFINE(type, bname, var)   LocalTensor<type> var = bname.template Get<type>()
#define _ALLOC(type, qname, var)        var = qname.template AllocTensor<type>()
#define _ALLOC_DEFINE(type, qname, var) LocalTensor<type> var = qname.template AllocTensor<type>()
#define _FREE(qname, var)               qname.FreeTensor(var)
#define DEQUE(type, name)               _DEQ_DEFINE(type, q_##name, name)
#define GET(type, name)                 _GET_DEFINE(type, bf_##name, name)
#define ALLOC(type, name)               _ALLOC_DEFINE(type, q_##name, name)

#define FREE(var)                       _FREE(q_##var, var)

// Init function stuff
#define _SET_GLOBAL(type, var, add, of)  var.SetGlobalBuffer((__gm__ type *)add + (of))
#define SET_GLOBAL(type, add, offset)   _SET_GLOBAL(type, gm_##add, add, offset)
#define __INIT_QUEUE(p, tp, qn, bn, l)  p InitBuffer(qn, (bn), ((l) * sizeof(tp)))
#define INIT_QUEUE_1(type, qname, len)  __INIT_QUEUE(p->, type, qname, 1, len)
#define INIT_QUEUE_2(type, qname, len)  __INIT_QUEUE(p->, type, qname, 2, len)
#define INIT_QUEUE_3(type, qname, len)  __INIT_QUEUE(p->, type, qname, 3, len)
#define INIT_QUEUE_N(type, q, bn, len)  __INIT_QUEUE(p->, type, q, (bn), (len))
#define __INIT_BUFF(p, tp, bn, l)       p InitBuffer(bn, ((l) * sizeof(tp)))
#define INIT_BUFF_N(type, bfname, len)  __INIT_BUFF(p->, type, bfname, len)

// bitwise hack
#define TREAT_AS(type)                  template ReinterpretCast<type>()
#define CAST(type, name)                reinterpret_cast<type>(name)

// tiling struct stuff
#define TI                              this->ti
#define TI_COPY_VAR(var_name)           .var_name = tiling_data.var_name
#define _TI_COPY_ARR(ti, arr_name, it)  ti.arr_name[it] = tiling_data.arr_name[it]
#define TI_COPY_ARR_I(arr_name)         _TI_COPY_ARR(ti, arr_name, i)
#define _COPY_ARR(ti, arr_name, it)     arr_name[it] = ti.arr_name[it]
#define COPY_ARR_I(arr_name)            _COPY_ARR(ti, arr_name, i)
#define DEF_VAR_FROM_TI(var_name)       auto var_name = TI.var_name
#define DEF_CVAR_FROM_TI(var_name)      const auto var_name = TI.var_name

// algorithms
#define CEIL(x, align_num)              (((x) + (align_num) - 1) / (align_num) * (align_num))
#define FLOOR(x, align_num)             ((x) / (align_num) * (align_num))

// Alias
using namespace AscendC;
using std::is_same_v;

// tiling structs
struct tilingInfo {
    uint32_t maxLength, tileLength, reminder;
};
/* ******************************************************************************************************* */
template<class DATA, int tilingKey, class tilingStruct> class DataCopyHelper { };                           // DataCopyHelper base template
/* ******************************************************************************************************* */
template<class DATA, class tilingStruct> class DataCopyHelper<DATA, 2, tilingStruct> {                      // DataCopyHelper double buffer version
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_local;
    GlobalTensor<DATA> gm_x, gm_y;
    tilingStruct ti;
    __aicore__ inline void CopyIn(uint32_t prefix, uint32_t length) {
        ALLOC(DATA, local);
        DataCopy(local, gm_x[prefix], length);
        ENQUE(DATA, local);
    }
    __aicore__ inline void CopyOut(uint32_t prefix, uint32_t length) {
        DEQUE(DATA, local);
        DataCopy(gm_y[prefix], local, length);
        FREE(local);
    }
public:
    __aicore__ inline DataCopyHelper(GM_ADDR x, GM_ADDR y, const tilingStruct &ti, TPipe *p):ti(ti) {
        this->set_ptr(x, y);
        INIT_QUEUE_2(DATA, q_local, TI.tileLength);
    }
    __aicore__ inline DataCopyHelper(const tilingStruct &ti, TPipe *p):ti(ti) {
        INIT_QUEUE_2(DATA, q_local, TI.tileLength);
    }
    __aicore__ inline void set_ptr(GM_ADDR x, GM_ADDR y) {
        SET_GLOBAL(DATA, x, 0);
        SET_GLOBAL(DATA, y, 0);
    }
    __aicore__ inline void exec(GM_ADDR x, GM_ADDR y) {
        this->set_ptr(x, y);
        this->exec();
    }
    __aicore__ inline void exec() {
        uint32_t i;
        for(i = 0; i < TI.maxLength; i += TI.tileLength) {
            CopyIn(i, TI.tileLength);
            CopyOut(i, TI.tileLength);
        }
        CopyIn(i, TI.reminder);
        CopyOut(i, TI.reminder);
    }
};
/* ******************************************************************************************************* */
template<class DATA, class INDICES, int tilingKey, class tilingStruct> class MyGather { };                  // Gather base template
/* ******************************************************************************************************* */
template<class DATA, class INDICES, class tilingStruct> class MyGather<DATA, INDICES, 0, tilingStruct> {    // Gather DataCopy Version
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    DataCopyHelper<DATA, 2, tilingInfo> helper;
public:
    __aicore__ inline MyGather( GM_ADDR x, 
                                GM_ADDR indices,
                                GM_ADDR y,
                                const tilingStruct &ti,
                                TPipe *p):  x(x),
                                            indices(indices),
                                            y(y),
                                            ti(ti),
                                            helper({ TI.maxLength, TI.tileLength, TI.reminder }, p) { }
    __aicore__ inline void exec() {
        INDICES index, basePrefix = 0, indicesPrefix = 0, inPrefix = 0, outPrefix = 0;
        for (uint32_t bigBatch = 0; bigBatch < TI.batchNumber; ++bigBatch) {
            for (uint32_t indiceIdx = 0; indiceIdx < TI.indicesLength; ++indiceIdx) {
                index = *(CAST(__gm__ INDICES *, indices) + indicesPrefix + indiceIdx);
                inPrefix = basePrefix + index * TI.sliceLength;
                helper.exec(this->x + inPrefix * sizeof(DATA), this->y + outPrefix * sizeof(DATA));
                // SetFlag<HardEvent::MTE3_S>(0);
                // WaitFlag<HardEvent::MTE3_S>(0);
                // PipeBarrier<PIPE_MTE2>();
                PipeBarrier<PIPE_MTE3>();
                outPrefix += TI.sliceLength;
            }
            basePrefix += TI.batchLength;
            indicesPrefix += TI.indicesLength;
        }
    }
};
/* ******************************************************************************************************* */
template<class DATA, class INDICES, class tilingStruct> class MyGather<DATA, INDICES, 1, tilingStruct> {    // Gather ScalarCopy Version
    GM_ADDR x;
    GM_ADDR indices;
    GM_ADDR y;
    tilingStruct ti;
    __aicore__ inline void CopyScalar(__gm__ DATA *x, __gm__ DATA *y) {
        for(uint32_t i = 0; i < TI.sliceLength; ++i) {
            *(y + i) = *(x + i);
        }
    }
public:
    __aicore__ inline MyGather( GM_ADDR x, 
                                GM_ADDR indices,
                                GM_ADDR y,
                                const tilingStruct &ti):x(x),
                                                        indices(indices),
                                                        y(y),
                                                        ti(ti) { }
    __aicore__ inline void exec() {
        INDICES index, basePrefix = 0, indicesPrefix = 0, inPrefix = 0, outPrefix = 0;
        for (uint32_t bigBatch = 0; bigBatch < TI.batchNumber; ++bigBatch) {
            for (uint32_t indiceIdx = 0; indiceIdx < TI.indicesLength; ++indiceIdx) {
                index = *(CAST(__gm__ INDICES *, indices) + indicesPrefix + indiceIdx);
                inPrefix = basePrefix + index * TI.sliceLength;
                CopyScalar(CAST(__gm__ DATA *, this->x) + inPrefix, CAST(__gm__ DATA *, this->y) + outPrefix);
                outPrefix += TI.sliceLength;
            }
            basePrefix += TI.batchLength;
            indicesPrefix += TI.indicesLength;
        }
    }
};
/* ******************************************************************************************************* */
extern "C" __global__ __aicore__ void gather(GM_ADDR x1, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if(TILING_KEY_IS(0)) {      // DataCopy Ver
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataWithDataCopy, ti, tiling);
        TPipe p;
        MyGather<DTYPE_Y, DTYPE_INDICES, 0, decltype(ti)>(x1, indices, y, ti, &p).exec();
    }
    else if(TILING_KEY_IS(1)) { // ScalarCopy Ver
        GET_TILING_DATA_WITH_STRUCT(GatherTilingDataScalarCopy, ti, tiling);
        MyGather<DTYPE_Y, DTYPE_INDICES, 1, decltype(ti)>(x1, indices, y, ti).exec();
    }
}