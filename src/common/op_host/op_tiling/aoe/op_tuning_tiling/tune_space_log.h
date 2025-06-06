/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_TUNING_TILING_TUNE_SPACE_LOG_H
#define OP_TUNING_TILING_TUNE_SPACE_LOG_H

#include <cstdint>
#include <memory>
#include "slog.h"
#include "mmpa_api.h"

namespace OpTuneSpace {
using Status = uint32_t;
constexpr Status SUCCESS = 0;
constexpr Status FAILED = 1;
constexpr Status TILING_NUMBER_EXCEED = 2;

constexpr int TUNE_MODULE = static_cast<int>(TUNE);
#define TUNE_SPACE_LOGD(format, ...) \
do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_DEBUG, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while (0)

#define TUNE_SPACE_LOGI(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_INFO, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGW(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_WARN, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGE(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_ERROR, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_LOGV(format, ...) \
    do {DlogSub(TUNE_MODULE, "TUNE_SPACE", DLOG_EVENT, "[Tid:%d]" #format"\n", mmGetTid(), ##__VA_ARGS__);} while(0)

#define TUNE_SPACE_MAKE_SHARED(execExpr0, execExpr1) \
    do {                                            \
        try {                                       \
            (execExpr0);                            \
        } catch (const std::bad_alloc &) {          \
            TUNE_SPACE_LOGE("Make shared failed");    \
            execExpr1;                              \
        }                                           \
    } while (false)
} // namespace OpTuneSpace
#endif