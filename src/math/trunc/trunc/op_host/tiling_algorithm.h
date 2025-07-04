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
#ifndef DATA_QUANTITY_TILING_ALGORITHM_H
#define DATA_QUANTITY_TILING_ALGORITHM_H

#include <cstdint>
#include <cassert>
#include <tuple>
#include <functional>

namespace kunlun {
    namespace tiling {
         * @tparam T
         * @param totalLen
         * @param coreNum
         * @param formerLen
         * @param formerNum
         * @param tailLen
         * @return
         */
        static std::tuple<uint64_t, uint64_t, uint64_t>
        sokmak(uint64_t totalLen, uint64_t coreNum, uint64_t factor, uint64_t align)
        {
            // TODO: uint64_t overflow should be considered
            assert(totalLen > 0 && "require totalLen > 0");
            assert(coreNum > 0 && "require coreNum > 0");
            assert(factor > 0 && "require factor > 0");
            assert(align > 0 && "require align > 0");

            uint64_t formerLen = 0;
            uint64_t formerNum = 0;
            uint64_t tailLen = totalLen;
            uint64_t bestObjective = totalLen;
            if (align % factor == 0) {
                uint64_t step = align / factor;
                for (uint64_t n = coreNum - 1; n != 0; --n) {
                    for (uint64_t f = 0; f <= totalLen; f += step) {
                        if (totalLen < f * n) { // detect underflow of t
                            break;
                        }
                        uint64_t t = totalLen - f * n;
                        uint64_t obj = std::max(f, t);
                        if (obj < bestObjective) {
                            formerLen = f;
                            formerNum = n;
                            tailLen = t;
                            bestObjective = obj;
                        }
                    }
                }
            } else {
                for (uint64_t n = coreNum - 1; n != 0; --n) {
                    for (uint64_t f = 0; f <= totalLen; ++f) {
                        if (totalLen < f * n) {
                            break;
                        }
                        uint64_t t = totalLen - f * n;
                        if ((f * factor) % align == 0) {
                            uint64_t obj = std::max(f, t);
                            if (obj < bestObjective) {
                                formerLen = f;
                                formerNum = n;
                                tailLen = t;
                                bestObjective = obj;
                            }
                        }
                    }
                }
            }

            // 合并 tailLen 到 formerLen
            // 1. f, n, t = [x, y, x] -> [x, y+1, 0]
            if (formerLen == tailLen) {
                tailLen = 0;
                ++formerNum;
            }
            // 2. f, n, t = [0, 0, x] -> [x, 1, 0] if x 满足对齐要求
            if (formerLen == 0 && (tailLen * factor) % align == 0) {
                formerLen = tailLen;
                formerNum = 1;
                tailLen = 0;
            }
            return std::make_tuple(formerLen, formerNum, tailLen);
        }
        using Int = uint32_t;
        using WorkloadFunc = std::function<Int(Int)>;
        /**
         * input: total, coreNum, factor, align
         *      total, coreNum, factor, align > 0

            output: formerLen, formerNum, tailLen

            constraints:
                [bound]
                0 <= formerLen <= total
                0 <= formerNum <= coreNum - 1
                0 <= tailLen <= total

                [equality]
                formerLen * formerNum + tailLen = total
                formerLen * factor = k * align, where k is a non-negative integer

            objective: minimize max{formerLen, tailLen}

            [x, 39, x]   obj: max{x, x} = x     [x, 40, 0]   obj : max{x, 0} = x   [x, 39, x] -> [x, 40, 0]
            [0, 0, x]   obj: max{0, x} = x      [x, 1, 0]   obj: max{x, 0} = x    [0, 0, x] ->  [x, 1, 0]  if x 满足对齐要求
         */
        static std::tuple<uint64_t, uint64_t, uint64_t, uint32_t, uint32_t>
        sokmakWithWorkload(uint64_t totalLen, uint64_t coreNum, uint64_t factor, uint64_t align, WorkloadFunc workload){
            assert(totalLen > 0 && "require totalLen > 0");
            assert(coreNum > 0 && "require coreNum > 0");
            assert(factor > 0 && "require factor > 0");
            assert(align > 0 && "require align > 0");

            uint64_t formerLen = 0;
            uint64_t formerNum = 0;
            uint64_t tailLen = totalLen;
            uint64_t bestObjective = workload(totalLen);
            if (align % factor == 0) {
                uint64_t step = align / factor;
                for (uint64_t n = coreNum - 1; n != 0; --n) {
                    for (uint64_t f = 0; f <= totalLen; f += step) {
                        if (totalLen < f * n) { // detect underflow of t
                            break;
                        }
                        uint64_t t = totalLen - f * n;
                        uint64_t obj = std::max(workload(f), workload(t));
                        if (obj < bestObjective) {
                            formerLen = f;
                            formerNum = n;
                            tailLen = t;
                            bestObjective = obj;
                        }
                    }
                }
            } else {
                for (uint64_t n = coreNum - 1; n != 0; --n) {
                    for (uint64_t f = 0; f <= totalLen; ++f) {
                        if (totalLen < f * n) {
                            break;
                        }
                        uint64_t t = totalLen - f * n;
                        if ((f * factor) % align == 0) {
                            uint64_t obj = std::max(workload(f), workload(t));
                            if (obj < bestObjective) {
                                formerLen = f;
                                formerNum = n;
                                tailLen = t;
                                bestObjective = obj;
                            }
                        }
                    }
                }
            }

            // 合并 tailLen 到 formerLen
            // 1. f, n, t = [x, y, x] -> [x, y+1, 0]
            if (formerLen == tailLen) {
                tailLen = 0;
                ++formerNum;
            }
            // 2. f, n, t = [0, 0, x] -> [x, 1, 0] if x 满足对齐要求
            if (formerLen == 0) {
                formerLen = tailLen;
                formerNum = 1;
                tailLen = 0;
            }
            return std::make_tuple(formerLen, formerNum, tailLen, workload(formerLen), workload(tailLen));
        }
    }
}

#endif //DATA_QUANTITY_TILING_ALGORITHM_H

