#ifndef LOG_TILING_H
#define LOG_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, ubPartDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreTailDataNum);
  TILING_DATA_FIELD_DEF(uint64_t, smallCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, bigCoreLoopNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum);
  TILING_DATA_FIELD_DEF(float, base);
  TILING_DATA_FIELD_DEF(float, scale);
  TILING_DATA_FIELD_DEF(float, shift);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Log, LogTilingData)
} // namespace optiling
#endif // LOG_TILING_H