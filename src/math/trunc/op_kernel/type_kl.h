/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef TYPE_KUNLUN_H
#define TYPE_KUNLUN_H

/// @brief 判断REAL是否为NaN/Inf
/// @param REAL 浮点数
#define IS_NAN_INF(REAL) !(float(REAL)*0==0)

/// @brief 判断T1是否为T2类型。(可在编译期给出结果)
/// @param T1 类型 1
/// @param T2 类型 2
#define IS_TYPE(T1, T2) std::is_same<T1,T2>::value

/// @brief 判断VAR是否为TYPE类型。(可在编译期给出结果)
/// @param VAR 变量
/// @param T 类型
#define IS_TYPE_VAR(VAR, T) _Generic(VAR,\
    T: true,\
    default: false\
)

/// @brief 判断是否为浮点型。(可在编译期给出结果)
/// @param T 类型
#define IS_FLOAT(T) (IS_TYPE(T,half)||IS_TYPE(T,float)||IS_TYPE(T,double))

/// @brief 判断是否为浮点型。(可在编译期给出结果)
/// @param VAR 变量
#define IS_FLOAT_VAR(VAR) (IS_TYPE_VAR(VAR,half)||IS_TYPE_VAR(VAR,float)||IS_TYPE_VAR(VAR,double))

/// @brief 判断是否为整型。(可在编译期给出结果)
/// @param T 类型
#define IS_INTEGER(T) (IS_TYPE(T,int8_t)||IS_TYPE(T,int16_t)||IS_TYPE(T,int32_t)||IS_TYPE(T,int64_t)||\
                    IS_TYPE(T,uint8_t)||IS_TYPE(T,uint16_t)||IS_TYPE(T,uint32_t)||IS_TYPE(T,uint64_t))

/// @brief 判断是否为整型。(可在编译期给出结果)
/// @param VAR 变量
#define IS_INTEGER_VAR(VAR) (IS_TYPE_VAR(VAR,int8_t)||IS_TYPE_VAR(VAR,int16_t)||IS_TYPE_VAR(VAR,int32_t)||IS_TYPE_VAR(VAR,int64_t)||\
                        IS_TYPE_VAR(VAR,uint8_t)||IS_TYPE_VAR(VAR,uint16_t)||IS_TYPE_VAR(VAR,uint32_t)||IS_TYPE_VAR(VAR,uint64_t))


#endif// TYPE_KUNLUN_H
