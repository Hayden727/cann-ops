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

/// @brief 判断T1是否为T2类型。(可在编译期给出结果)
/// @param T1 类型 1
/// @param T2 类型 2
#define IS_TYPE(T1, T2) std::is_same<T1,T2>::value
#endif// TYPE_KUNLUN_H
