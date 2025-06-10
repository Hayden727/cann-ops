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

template<int tilingKey> struct ReshapeTilingInfo { };

template<> struct ReshapeTilingInfo<1> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};
template<> struct ReshapeTilingInfo<2> { 
    uint32_t tileLength, tileNumber, reminder;
    // int32_t axis, numAxes;
};

template<class DATA, class SHAPE, int tilingKey> class MyReshape {};

template<class DATA, class SHAPE> class MyReshape<DATA, SHAPE, 1> {
    GlobalTensor<DATA> gm_x, gm_y;
    GlobalTensor<SHAPE> gm_shape;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> q_data;
    ReshapeTilingInfo<1> ti;
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        ALLOC(DATA, data);
        DataCopy(data, gm_x[offset], length);
        ENQUE(DATA, data);
    }
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        DEQUE(DATA, data);
        DataCopy(gm_y[offset], data, length);
        FREE(data);
    }
public:
    __aicore__ inline MyReshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, const ReshapeTilingInfo<1> &ti, TPipe *p):ti(ti) {
        SET_GLOBAL(DATA, x, 0);
        SET_GLOBAL(SHAPE, shape, 0);
        SET_GLOBAL(DATA, y, 0);
        INIT_QUEUE_2(DATA, q_data, TI.tileLength);
    }
    __aicore__ inline void exec() {
        for(uint32_t i = 0; i < TI.tileNumber; i++) {
            CopyIn(i * TI.tileLength, TI.tileLength);
            CopyOut(i * TI.tileLength, TI.tileLength);
        }
        const uint32_t miniBatch = 32 / sizeof(DATA);
        const uint32_t lastLength = CEIL(TI.reminder, miniBatch);
        CopyIn(TI.tileNumber * TI.tileLength, lastLength);
        CopyOut(TI.tileNumber * TI.tileLength, lastLength);
    }
};
extern "C" __global__ __aicore__ void reshape(GM_ADDR x, GM_ADDR shape, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) {
        ReshapeTilingInfo<1> ti{
            TI_COPY_VAR(tileLength),
            TI_COPY_VAR(tileNumber),
            TI_COPY_VAR(reminder),
        };
        TPipe p;
        MyReshape<DTYPE_Y, DTYPE_SHAPE, 1>(x, shape, y, ti, &p).exec();
    }
}