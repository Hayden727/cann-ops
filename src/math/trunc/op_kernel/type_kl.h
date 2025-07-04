/* 
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef TYPE_KUNLUN_H
#define TYPE_KUNLUN_H

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
#endif// TYPE_KUNLUN_H
